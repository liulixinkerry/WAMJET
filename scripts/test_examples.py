"""Focused CPU tests of the shared benchmark and reference flow."""
import contextlib
import json
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]
# Load patched modules before patch.dict snapshots sys.modules. Otherwise a later
# test can patch a stale package attribute while run() imports a fresh module.
import wamjet.capture
import wamjet.instrument
import wamjet.preflight
from benchmark import Adapter, run


class Power:
    def __init__(self):
        self.stop_flag = threading.Event()

    def start(self):
        pass

    def join(self, timeout=None):
        pass

    def summary(self):
        return {}


class ExampleTests(unittest.TestCase):
    @contextlib.contextmanager
    def fake_device(self):
        fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(synchronize=lambda: None))
        fake_dynamo = types.ModuleType("torch._dynamo.utils")
        fake_dynamo.counters = {}

        def measure(infer, **kwargs):
            infer()
            return {"trusted": True, "median_ms": 1.0, "steady_n": 3, "n_calls": 3,
                    "node": {"torch": "test"}}

        def capture(infer, path, **kwargs):
            Path(path).write_text('{"traceEvents": []}')
            return measure(infer, **kwargs)

        with patch.dict(sys.modules, {"torch": fake_torch, "torch._dynamo.utils": fake_dynamo}), \
                patch("wamjet.preflight.fingerprint", return_value={"torch": "test"}), \
                patch("wamjet.preflight.hazards", return_value=[]), \
                patch("wamjet.capture.measure", side_effect=measure), \
                patch("wamjet.capture.capture", side_effect=capture), \
                patch("wamjet.instrument.PowerSampler", Power):
            yield

    def test_timing_defaults_skip_diagnostics_and_allow_opt_in(self):
        with tempfile.TemporaryDirectory() as td, self.fake_device():
            build = lambda: Adapter(lambda: np.array([1.0]), {}, {})
            with patch("wamjet.capture.capture", side_effect=AssertionError("unexpected trace")), \
                    patch("wamjet.instrument.PowerSampler", side_effect=AssertionError("unexpected power")), \
                    patch("wamjet.preflight.hazards", side_effect=AssertionError("unexpected import probe")):
                self.assertEqual(run(build, tag="timing", out_dir=td), 0)
            self.assertFalse(Path(td, "timing_trace.json").exists())
            self.assertNotIn("power", json.loads(Path(td, "timing.result.json").read_text()))
            self.assertEqual(run(build, tag="profile", out_dir=td,
                                 profile=True, sample_power=True), 0)
            self.assertTrue(Path(td, "profile_trace.json").exists())
            self.assertIn("power", json.loads(Path(td, "profile.result.json").read_text()))

    def test_invalid_outputs_cannot_pass_final_task_acceptance(self):
        with tempfile.TemporaryDirectory() as td, self.fake_device():
            ref = f"{td}/stock"
            def build(value=1.0, dtype=np.float32):
                return Adapter(lambda: np.array([value], dtype=dtype), {}, {})
            run(build, tag="baseline", out_dir=td, reference=ref, record_reference=True)
            for candidate in (lambda: build(float("nan")), lambda: build(dtype=np.float64)):
                self.assertEqual(run(candidate, tag="invalid", out_dir=td, reference=ref,
                                     maximum_tier="C", task_evaluation={"passed": True}), 1)

    def test_missing_reference_fails_before_build(self):
        with tempfile.TemporaryDirectory() as td, self.fake_device():
            def build():
                self.fail("a missing reference must be detected before loading")
            with self.assertRaises(FileNotFoundError):
                run(build, tag="candidate", out_dir=td, reference=f"{td}/missing")

    def test_reference_checks_and_failed_evidence_are_shared(self):
        with tempfile.TemporaryDirectory() as td, self.fake_device():
            reference = f"{td}/stock"
            tolerance = {"max_abs": 1e-4, "rel_l2": 1e-4}
            def build(prompt="open", value=1.0):
                return Adapter(lambda: np.array([value]),
                               {"image": np.array([1]), "prompt": prompt, "steps": 10}, {})
            self.assertEqual(run(build, tag="baseline", out_dir=td, reference=reference,
                                 record_reference=True, tier_a_tolerance=tolerance), 0)
            self.assertEqual(run(lambda: build(value=1.00001), tag="candidate", out_dir=td,
                                 reference=reference, maximum_tier="A"), 0)
            verdict = json.loads(Path(td, "candidate.equivalence.json").read_text())
            self.assertEqual(verdict["tier_a_tolerance"], tolerance)
            self.assertEqual(verdict["tier"], "A")
            self.assertEqual(run(lambda: build("close"), tag="wrong-input", out_dir=td,
                                 reference=reference), 1)
            verdict = json.loads(Path(td, "wrong-input.equivalence.json").read_text())
            self.assertEqual(verdict["blocked"], "input mismatch")
            self.assertTrue(Path(td, "wrong-input.result.json").exists())
            with self.assertRaises(FileExistsError):
                run(build, tag="overwrite", out_dir=td, reference=reference, record_reference=True)

    def test_stateful_floor_samples_share_the_same_start(self):
        with tempfile.TemporaryDirectory() as td, self.fake_device():
            state = {"calls": 0}
            def build():
                def infer():
                    state["calls"] += 1
                    return np.array([float(state["calls"])])
                def reset():
                    state["calls"] = 0
                return Adapter(infer, {"prompt": "open"}, {}, reset)
            self.assertEqual(run(build, tag="baseline", out_dir=td, reference=f"{td}/stock",
                                 record_reference=True), 0)
            evidence = json.loads(Path(td, "baseline.reference.json").read_text())
            self.assertEqual(evidence["floor"]["max_abs"], 0.0)
            self.assertEqual(state["calls"], 1)  # reset again before timing

    def test_lossless_phase_rejects_unvalidated_noise_floor_only_match(self):
        from wamjet import equivalence

        with tempfile.TemporaryDirectory() as td, self.fake_device():
            values = iter([1.0, 1.1, 1.0])
            floor = equivalence.noise_floor(lambda _: np.array([next(values)]), None, repeats=3)
            reference = f"{td}/stock"
            equivalence.save_reference(reference, np.array([1.0]), floor=floor,
                                       inputs={"prompt": "open"})
            def build():
                return Adapter(lambda: np.array([1.05]), {"prompt": "open"}, {})

            self.assertEqual(run(build, tag="unvalidated-b", out_dir=td,
                                 reference=reference), 1)
            self.assertEqual(run(build, tag="validated-b", out_dir=td,
                                 reference=reference,
                                 task_evaluation={"passed": True}), 0)
            self.assertEqual(run(build, tag="exact", out_dir=td, reference=reference,
                                 maximum_tier="A"), 1)
            smoke = json.loads(Path(td, "exact.smoke.json").read_text())
            self.assertFalse(smoke["ok"])
            self.assertEqual(smoke["validation"]["tier"], "B")

    def test_tier_c_requires_task_evidence_and_keeps_input_gate(self):
        with tempfile.TemporaryDirectory() as td, self.fake_device():
            reference = f"{td}/stock"
            def build(value=1.0, prompt="open"):
                return Adapter(lambda: np.array([value]), {"prompt": prompt}, {})
            run(build, tag="baseline", out_dir=td, reference=reference, record_reference=True)
            for task, expected in ((None, 1), ({"passed": False}, 1), ({"passed": True}, 0)):
                with self.subTest(task=task):
                    self.assertEqual(run(lambda: build(2.0), tag="approx", out_dir=td,
                                         reference=reference, maximum_tier="C",
                                         task_evaluation=task), expected)
                    smoke = json.loads(Path(td, "approx.smoke.json").read_text())
                    result = json.loads(Path(td, "approx.result.json").read_text())
                    self.assertEqual(smoke["ok"], expected == 0)
                    self.assertEqual(result["equivalence"]["tier"], "C")
                    self.assertEqual(result["numerical_policy"], {"maximum_tier": "C"})
                    self.assertEqual(smoke.get("task_evaluation"), task)
                    self.assertEqual(result.get("task_evaluation"), task)
            self.assertEqual(run(lambda: build(2.0, "close"), tag="wrong-input", out_dir=td,
                                 reference=reference, maximum_tier="C",
                                 task_evaluation={"passed": True}), 1)

    def test_invalid_tier_fails_before_build(self):
        with tempfile.TemporaryDirectory() as td, self.fake_device():
            with self.assertRaises(ValueError):
                run(lambda: self.fail("must validate policy before loading"),
                    tag="bad-policy", out_dir=td, maximum_tier="D")

    def test_screening_can_continue_without_task_evidence_but_keeps_failed_gate(self):
        with tempfile.TemporaryDirectory() as td, self.fake_device():
            reference = f"{td}/stock"
            def build(value=1.0, prompt="open"):
                return Adapter(lambda: np.array([value]), {"prompt": prompt}, {})
            run(build, tag="baseline", out_dir=td, reference=reference, record_reference=True)
            self.assertEqual(run(lambda: build(2.0), tag="screen", out_dir=td,
                                 reference=reference, maximum_tier="C", screening=True), 0)
            smoke = json.loads(Path(td, "screen.smoke.json").read_text())
            result = json.loads(Path(td, "screen.result.json").read_text())
            self.assertFalse(smoke["ok"])
            self.assertFalse(result["numerical_accepted"])
            self.assertTrue(result["screening_passed"])
            self.assertEqual(result["validation_status"], "provisional")
            self.assertEqual(result["equivalence"]["tier"], "C")
            self.assertNotIn("task_evaluation", result)
            startup = json.loads(Path(td, "screen.startup.json").read_text())
            self.assertEqual(result["startup"], startup)
            self.assertGreaterEqual(startup["build_seconds"], 0)
            self.assertGreaterEqual(startup["first_inference_seconds"], 0)
            self.assertGreaterEqual(result["run_seconds"], startup["build_seconds"])
            # Ordinary validation still fails until real task evidence is supplied.
            self.assertEqual(run(lambda: build(2.0), tag="final", out_dir=td,
                                 reference=reference, maximum_tier="C"), 1)
            for candidate, policy in ((lambda: build(2.0, "close"), {}),
                                      (lambda: build(float("nan")), {}),
                                      (lambda: build(2.0), {"maximum_tier": "A"}),
                                      (lambda: build(2.0), {"task_evaluation": {"passed": False}})):
                self.assertEqual(run(candidate, tag="bad-screen", out_dir=td,
                                     reference=reference, screening=True,
                                     **{"maximum_tier": "C", **policy}), 1)
            with patch("wamjet.capture.measure", return_value={"trusted": False, "median_ms": 1.0}):
                self.assertEqual(run(lambda: build(2.0), tag="untrusted", out_dir=td,
                                     reference=reference, screening=True, maximum_tier="C"), 1)

    def test_screening_requires_existing_original_reference(self):
        with tempfile.TemporaryDirectory() as td, self.fake_device():
            for kwargs in ({}, {"reference": f"{td}/stock", "record_reference": True}):
                with self.assertRaises(ValueError):
                    run(lambda: self.fail("screening needs a reference before loading"),
                        tag="screen", out_dir=td, screening=True, **kwargs)


if __name__ == "__main__":
    unittest.main()
