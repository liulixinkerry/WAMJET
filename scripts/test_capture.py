"""CPU-only regression for warmup/profile execution-mode comparability."""
import contextlib
import pathlib
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from wamjet.capture import capture, measure


class FakeTimer(contextlib.AbstractContextManager):
    def __init__(self):
        self.ms = []

    def __exit__(self, *args):
        self.ms.append(1.0)


class FakeProfiler(contextlib.AbstractContextManager):
    def __exit__(self, *args):
        pass

    def export_chrome_trace(self, path):
        pathlib.Path(path).write_text('{"traceEvents": []}')


class CaptureTests(unittest.TestCase):
    def test_slow_compile_does_not_abort_by_default(self):
        class SlowFirstCall(FakeTimer):
            def __exit__(self, *args):
                self.ms.append(600_000.0 if not self.ms else 1.0)

        with patch("wamjet.capture.Timer", SlowFirstCall), \
                patch("wamjet.capture.node_fingerprint", return_value={}):
            result = measure(lambda: None, warmup=5, progress=False)
            self.assertTrue(result["trusted"])
            self.assertEqual(result["warmup_calls"], 1)
            with self.assertRaises(RuntimeError):
                measure(lambda: None, warmup=5, progress=False, stall_s=300)

    def test_unstable_timing_is_returned_for_diagnosis_by_default(self):
        class UnstableTimer(FakeTimer):
            def __exit__(self, *args):
                self.ms.append(2.0 ** len(self.ms))

        with patch("wamjet.capture.Timer", UnstableTimer), \
                patch("wamjet.capture.node_fingerprint", return_value={}):
            result = measure(lambda: None, warmup=5, progress=False)
            self.assertFalse(result["trusted"])
            self.assertEqual(len(result["latency_series"]), 5)
            with self.assertRaises(RuntimeError):
                measure(lambda: None, warmup=5, progress=False, strict=True)

    def test_measure_needs_no_profiler_and_keeps_all_timings(self):
        seen = []
        with patch("wamjet.capture.Timer", FakeTimer), \
                patch("wamjet.capture.node_fingerprint", return_value={"gpu": "test"}), \
                patch.dict(sys.modules, {"torch": types.SimpleNamespace()}):
            result = measure(lambda: seen.append(True), warmup=5, progress=False)
        self.assertEqual(len(seen), 5)
        self.assertEqual(result["latency_series"], [1.0] * 5)
        self.assertTrue(result["trusted"])
        self.assertNotIn("trace", result)

    def test_warmup_and_profile_preserve_callers_mode(self):
        for caller_inference_mode in (False, True):
            with self.subTest(caller_inference_mode=caller_inference_mode):
                state = {"inference_mode": caller_inference_mode}

                @contextlib.contextmanager
                def inference_mode():
                    previous = state["inference_mode"]
                    state["inference_mode"] = True
                    try:
                        yield
                    finally:
                        state["inference_mode"] = previous

                torch = types.SimpleNamespace(
                    cuda=types.SimpleNamespace(synchronize=lambda: None),
                    inference_mode=inference_mode,
                    profiler=types.SimpleNamespace(
                        ProfilerActivity=types.SimpleNamespace(CPU=0, CUDA=1),
                        profile=lambda **kwargs: FakeProfiler()))
                seen = []
                with tempfile.TemporaryDirectory() as td, \
                        patch.dict(sys.modules, {"torch": torch}), \
                        patch("wamjet.capture.Timer", FakeTimer), \
                        patch("wamjet.capture.node_fingerprint", return_value={}):
                    result = capture(lambda: seen.append(state["inference_mode"]),
                                     str(pathlib.Path(td) / "trace.json"),
                                     warmup=3, calls=2, progress=False)
                self.assertEqual(seen, [caller_inference_mode] * 5)
                self.assertTrue(result["trusted"])


if __name__ == "__main__":
    unittest.main()
