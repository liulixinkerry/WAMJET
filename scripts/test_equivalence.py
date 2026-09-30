"""Self-test for the G2 reference round-trip.

Runs on the host only: `fingerprint` is stubbed so the test never initializes a
CUDA context, which would otherwise perturb any measurement sharing the device.

Usage:  python scripts/test_equivalence.py
"""
import pathlib
import json
import sys
import tempfile
from collections import UserDict

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from wamjet import equivalence, preflight  # noqa: E402

STACK = {"host": "n1", "torch": "2.13.0", "cuda": "13.0", "gpu": "H100", "sm": "90"}
preflight.fingerprint = lambda: dict(STACK)

fails = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}{'  -- ' + detail if detail and not cond else ''}")
    if not cond:
        fails.append(name)


rng = np.random.default_rng(0)
base = {"action": rng.standard_normal((1, 16, 7)).astype(np.float32)}
inputs = {"obs": rng.standard_normal((1, 3, 224, 224)).astype(np.float32)}
floor = {"max_abs": 1e-3, "rel_l2": 1e-4, "repeats": 3, "deterministic": False}

with tempfile.TemporaryDirectory() as td:
    ref = f"{td}/ref"
    equivalence.save_reference(ref, base, floor=floor, inputs=inputs, note="baseline")

    r = equivalence.check_reference(ref, base, inputs=inputs)
    check("identical outputs classify tier A", r.get("tier") == "A", str(r))

    nested_ref = f"{td}/nested"
    nested = {"action": base["action"], "done": False,
              "state": {"value": np.array([2.0])}}
    equivalence.save_reference(nested_ref, nested)
    for label, candidate in (
        ("scalar", {**nested, "done": True}),
        ("nested", {**nested, "state": {"value": np.array([99.0])}}),
    ):
        r = equivalence.check_reference(nested_ref, candidate)
        check(f"changed {label} outputs cannot pass as exact", r.get("tier") == "C", str(r))
    r = equivalence.check_reference(ref, {"action": base["action"].astype(np.float64)}, inputs=inputs)
    check("output dtype changes block acceptance", bool(r.get("blocked")), str(r))
    r = equivalence.check_reference(ref, {**base, "new_output": np.array([1])}, inputs=inputs)
    check("extra output keys block acceptance", r.get("blocked") == "extra outputs", str(r))
    r = equivalence.check_reference(ref, {"action": np.full_like(base["action"], np.nan)}, inputs=inputs)
    check("nonfinite outputs cannot pass with task evidence",
          bool(r.get("blocked")) and not equivalence.is_accepted(r, "C", {"passed": True}), str(r))
    check("saved nonfinite metrics cannot pass with task evidence",
          not equivalence.is_accepted({"tier": "C", "diffs": {"action": {"max_abs": float("nan")}}},
                                      "C", {"passed": True}))
    with np.load(nested_ref + ".npz", allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    meta = json.loads(str(arrays["__meta__"]))
    del meta["output_dtypes"]
    arrays["__meta__"] = np.array(json.dumps(meta))
    np.savez(f"{td}/old-output.npz", **arrays)
    r = equivalence.check_reference(f"{td}/old-output", nested)
    check("old output references require a fresh original reference",
          r.get("blocked") == "legacy output reference", str(r))

    near = {"action": base["action"] + 1e-5}
    r = equivalence.check_reference(ref, near, inputs=inputs)
    check("drift inside the floor classifies tier B", r.get("tier") == "B", str(r))
    check("tier B needs paired task evidence",
          not equivalence.is_accepted(r, "B")
          and equivalence.is_accepted(r, "B", {"passed": True}), str(r))

    lossless_ref = f"{td}/lossless"
    equivalence.save_reference(
        lossless_ref, base, floor=floor, inputs=inputs,
        tier_a_tolerance={"max_abs": 2e-5, "rel_l2": 2e-5},
    )
    r = equivalence.check_reference(lossless_ref, near, inputs=inputs)
    check("declared lossless numerical drift classifies tier A",
          r.get("tier") == "A" and not r["diffs"]["action"]["bit_identical"], str(r))
    check("lossless tier A needs no task-quality override",
          equivalence.is_accepted(r, "A"), str(r))

    far = {"action": base["action"] + 5e-1}
    r = equivalence.check_reference(ref, far, inputs=inputs)
    check("drift beyond the floor classifies tier C", r.get("tier") == "C", str(r))
    check("tier C states what it owes", bool(r.get("owes")), str(r))

    r = equivalence.check_reference(ref, base, inputs={"obs": inputs["obs"] + 1.0})
    check("a different input blocks rather than tiering",
          r.get("blocked") == "input mismatch", str(r))

    r = equivalence.check_reference(ref, {"other": base["action"]}, inputs=inputs)
    check("a missing output key blocks", r.get("blocked") == "missing outputs", str(r))

    # A reference recorded with no floor cannot certify a non-zero drift.
    ref2 = f"{td}/ref2"
    equivalence.save_reference(ref2, base, inputs=inputs)
    r = equivalence.check_reference(ref2, near, inputs=inputs)
    check("no floor means a moved output cannot be called a reorder",
          r.get("tier") == "C", str(r))

    # Stack drift must not be silently ignored.
    preflight.fingerprint = lambda: dict(STACK, torch="2.14.0", cuda="13.1")
    r = equivalence.check_reference(ref, base, inputs=inputs)
    check("a moved stack raises an advisory", bool(r.get("advisories")), str(r))
    check("a moved stack cannot pass final acceptance", not equivalence.is_accepted(r), str(r))
    preflight.fingerprint = lambda: dict(STACK)

    # Bare tensors, not just mappings.
    ref3 = f"{td}/ref3"
    equivalence.save_reference(ref3, base["action"], floor=floor, inputs=inputs)
    r = equivalence.check_reference(ref3, base["action"], inputs=inputs)
    check("a bare array round-trips", r.get("tier") == "A", str(r))

    # Hugging Face BatchFeature and similar policy outputs implement Mapping
    # through UserDict rather than subclassing the builtin dict.
    ref4 = f"{td}/ref4"
    mapping_output = UserDict(base)
    mapping_inputs = UserDict(inputs)
    equivalence.save_reference(ref4, mapping_output, floor=floor, inputs=mapping_inputs)
    r = equivalence.check_reference(ref4, mapping_output, inputs=mapping_inputs)
    check("a UserDict mapping round-trips", r.get("tier") == "A", str(r))

    # Absolute error and relative error can rank outputs differently.
    multi = {"large": np.array([1000.0]), "small": np.array([0.001])}
    ref5 = f"{td}/multi"
    per_key = {key: {"max_abs": 1.0, "rel_l2": 0.01} for key in multi}
    equivalence.save_reference(ref5, multi, floor={"per_key": per_key})
    r = equivalence.check_reference(
        ref5, {"large": np.array([1000.5]), "small": np.array([0.002])}
    )
    check("the worst tier wins across output scales",
          r.get("tier") == "C" and r["worst_key"] == "small", str(r))

    conditioned = {"image": np.array([1]), "prompt": "open door", "steps": 10,
                   "context": {"history": [None, True, (1.0, "task")]}}
    typed_ref = f"{td}/typed"
    equivalence.save_reference(typed_ref, base, inputs=conditioned)
    for label, changed in (
        ("prompt", {**conditioned, "prompt": "close door"}),
        ("scalar option", {**conditioned, "steps": 1}),
        ("nested input", {**conditioned, "context": {"history": [None, False, (1.0, "task")]}}),
    ):
        r = equivalence.check_reference(typed_ref, base, inputs=changed)
        check(f"a changed {label} blocks comparison", r.get("blocked") == "input mismatch", str(r))
    check("input mapping order does not change the hash",
          equivalence.digest(conditioned) == equivalence.digest(dict(reversed(list(conditioned.items())))))
    check("input scalar types have distinct hashes",
          len({equivalence.digest(x) for x in (True, 1, 1.0, "1")}) == 4)
    check("list and tuple input types have distinct hashes",
          equivalence.digest([1, 2]) != equivalence.digest((1, 2)))
    raw = np.array([1065353216], dtype=np.int32)
    check("array dtype changes the hash even with identical bytes",
          equivalence.digest(raw) != equivalence.digest(raw.view(np.float32)))
    try:
        equivalence.digest({"image": raw, "unsupported": object()})
    except TypeError:
        check("unsupported nested input is rejected", True)
    else:
        check("unsupported nested input is rejected", False)

    with np.load(typed_ref + ".npz") as saved:
        contents = {key: saved[key] for key in saved.files}
    metadata = json.loads(str(contents["__meta__"]))
    metadata.pop("input_digest_version")
    contents["__meta__"] = np.array(json.dumps(metadata))
    np.savez(f"{td}/legacy.npz", **contents)
    r = equivalence.check_reference(f"{td}/legacy", base, inputs=conditioned)
    check("legacy input hashes require a new reference",
          r.get("blocked") == "legacy input digest" and "re-record" in r["note"], str(r))

    levels = iter((1.0, 2.0, 0.0))
    multi_floor = equivalence.noise_floor(
        lambda _: {"action": np.array([1.0]), "video": np.array([next(levels)])}, None
    )
    check("noise floors retain per-output measurements",
          multi_floor["per_key"]["action"]["max_abs"] == 0.0
          and multi_floor["max_abs"] == 1.0, str(multi_floor))
    multi_base = {"action": np.array([1.0]), "video": np.array([1.0])}
    multi_changed = {**multi_base, "action": np.array([1.5])}
    equivalence.save_reference(f"{td}/separate", multi_base, floor=multi_floor)
    r = equivalence.check_reference(f"{td}/separate", multi_changed)
    check("video noise cannot certify deterministic action drift",
          r.get("tier") == "C" and r["worst_key"] == "action", str(r))
    legacy_floor = {key: value for key, value in multi_floor.items() if key != "per_key"}
    check("direct classification cannot use a pooled output floor",
          equivalence.classify(equivalence.compare(np.array([1.0]), np.array([1.5])),
                               legacy_floor)["tier"] == "C")
    equivalence.save_reference(f"{td}/pooled", multi_base, floor=legacy_floor)
    r = equivalence.check_reference(f"{td}/pooled", multi_changed)
    check("a legacy pooled multi-output floor cannot certify drift",
          r.get("tier") == "C", str(r))

    limits = {"per_key": {"action": {"max_abs": 1e-6, "rel_l2": 1e-6},
                          "video": {"max_abs": 1.0, "rel_l2": 1.0}}}
    equivalence.save_reference(f"{td}/limits", multi_base, tier_a_tolerance=limits)
    for candidate, expected in (({**multi_base, "video": np.array([1.5])}, "A"),
                                (multi_changed, "C")):
        r = equivalence.check_reference(f"{td}/limits", candidate)
        check(f"per-output tolerances classify {expected} without pooling",
              r.get("tier") == expected, str(r))

# A deterministic build has a floor of exactly zero; the tier B test degenerates
# there and a ratio against it is meaningless.
zero = {"max_abs": 0.0, "rel_l2": 0.0, "repeats": 3, "deterministic": True}
moved = {"max_abs": 1.86e-2, "rel_l2": 1.16e-2, "bit_identical": False}
v = equivalence.classify(moved, zero)
check("a zero floor still classifies tier C", v["tier"] == "C", str(v))
check("a zero floor reports no ratio", "x the noise floor" not in v["note"], v["note"])
check("a zero floor says the floor cannot certify a reorder",
      "cannot certify" in v["note"], v["note"])
v = equivalence.classify(moved, {"max_abs": 1e-3, "rel_l2": 1e-4, "repeats": 3})
check("a non-zero floor still reports a ratio", "x the noise floor" in v["note"], v["note"])

# noise_floor must accept whatever the policy returns. A policy usually returns a
# mapping; comparing the raw object worked for a bare tensor and raised on a dict,
# so the floor could only be measured for models whose return type happened to match.
counter = {"n": 0}


def jittery(_):
    counter["n"] += 1
    return {"action": base["action"] + counter["n"] * 1e-6}


f = equivalence.noise_floor(jittery, None, repeats=3)
check("noise_floor accepts a mapping output", f["max_abs"] > 0, str(f))
check("noise_floor reports which keys it compared", f.get("keys") == ["action"], str(f))
f2 = equivalence.noise_floor(lambda _: base["action"], None, repeats=3)
check("noise_floor still accepts a bare array", f2["deterministic"] is True, str(f2))

shared_floor = np.zeros(1)


def reused_output(_):
    shared_floor[:] += 1
    return {"action": shared_floor}


f3 = equivalence.noise_floor(reused_output, None, repeats=3)
check("noise floor snapshots reused output buffers",
      not f3["deterministic"] and f3["max_abs"] == 2.0, str(f3))

# Captured graphs and caches can be correct after warmup while returning a stale
# first result or retaining state from a different input. A named A -> B -> A
# sequence checks each result against stock and checks the repeated A against its
# own earlier output.
supports_sequence = hasattr(equivalence, "check_sequence")
check("equivalence exposes a sequence check", supports_sequence)
if supports_sequence:
    input_a = {"obs": np.array([1.0, 2.0], dtype=np.float32)}
    input_b = {"obs": np.array([3.0, 5.0], dtype=np.float32)}

    def sequence_policy(x):
        return {"action": x["obs"] * 2.0}

    with tempfile.TemporaryDirectory() as td:
        ref_a = f"{td}/a"
        ref_b = f"{td}/b"
        equivalence.save_reference(ref_a, sequence_policy(input_a), floor=zero, inputs=input_a)
        equivalence.save_reference(ref_b, sequence_policy(input_b), floor=zero, inputs=input_b)
        cases = [
            {"name": "A", "inputs": input_a, "reference": ref_a},
            {"name": "B", "inputs": input_b, "reference": ref_b},
            {"name": "A", "inputs": input_a, "reference": ref_a},
        ]

        good = equivalence.check_sequence(sequence_policy, cases)
        check("A-B-A sequence passes", good["ok"], str(good))
        check("sequence records its pattern", good["pattern"] == ["A", "B", "A"],
              str(good))
        check("every sequence output matches stock",
              all(r.get("tier") == "A" for r in good["comparisons"]), str(good))
        check("A replay is bit-identical",
              len(good["replays"]) == 1 and good["replays"][0]["bit_identical"],
              str(good))

        tolerant_cases = []
        for case in cases:
            tolerant_ref = f"{td}/tolerant-{case['name']}"
            equivalence.save_reference(
                tolerant_ref, sequence_policy(case["inputs"]), inputs=case["inputs"],
                tier_a_tolerance={"max_abs": 2e-5, "rel_l2": 2e-5},
            )
            tolerant_cases.append({**case, "reference": tolerant_ref})
        tolerant_calls = {"n": 0}

        def tolerant_policy(x):
            tolerant_calls["n"] += 1
            return {"action": x["obs"] * 2.0 + tolerant_calls["n"] * 1e-6}

        tolerant = equivalence.check_sequence(tolerant_policy, tolerant_cases)
        check("A-B-A replay accepts declared lossless numerical drift",
              tolerant["ok"] and tolerant["replays"][0]["lossless"]
              and not tolerant["replays"][0]["bit_identical"], str(tolerant))

        state = {"calls": 0}

        def stale_on_replay(x):
            state["calls"] += 1
            if state["calls"] == 3:
                return {"action": x["obs"] * 0.0}
            return sequence_policy(x)

        stale = equivalence.check_sequence(stale_on_replay, cases)
        check("a stale repeated input fails the sequence", not stale["ok"], str(stale))
        check("a stale repeated input fails replay identity",
              not stale["replays"][0]["bit_identical"], str(stale))

        # CUDA graphs commonly return the same static output allocation on
        # every replay. The checker must snapshot A rather than retain a view
        # that B and the final broken A can mutate underneath it.
        shared = np.empty_like(input_a["obs"])
        shared_state = {"calls": 0}

        def stale_shared_output(x):
            shared_state["calls"] += 1
            if shared_state["calls"] == 3:
                shared.fill(0.0)
            else:
                np.multiply(x["obs"], 2.0, out=shared)
            return {"action": shared}

        aliased = equivalence.check_sequence(stale_shared_output, cases)
        check("reused output storage cannot hide a stale replay",
              not aliased["replays"][0]["bit_identical"], str(aliased))

        wrong_reference = [cases[0], {**cases[1], "reference": ref_a}, cases[2]]
        wrong = equivalence.check_sequence(sequence_policy, wrong_reference)
        check("a wrong per-input reference fails the sequence", not wrong["ok"], str(wrong))
        check("the input mismatch remains visible",
              wrong["comparisons"][1].get("blocked") == "input mismatch", str(wrong))

        def tier_c_policy(x):
            return {"action": x["obs"] * 2.01}

        default_c = equivalence.check_sequence(tier_c_policy, cases)
        check("tier C sequence fails closed by default", not default_c["ok"], str(default_c))
        unproved_c = equivalence.check_sequence(tier_c_policy, cases, maximum_tier="C")
        check("tier C sequence still requires task evidence", not unproved_c["ok"], str(unproved_c))
        passed_c = equivalence.check_sequence(
            tier_c_policy, cases, maximum_tier="C", task_evaluation={"passed": True}
        )
        check("tier C sequence passes with explicit policy and task evidence",
              passed_c["ok"] and passed_c["replays"][0]["bit_identical"], str(passed_c))
        check("passing task evidence preserves the measured tier",
              all(item["tier"] == "C" for item in passed_c["comparisons"]), str(passed_c))
        shared_state["calls"] = 0
        stale_c = equivalence.check_sequence(
            stale_shared_output, cases, maximum_tier="C", task_evaluation={"passed": True}
        )
        check("task evidence never excuses a stale replay", not stale_c["ok"], str(stale_c))

# Optional CPU tensor coverage: importing torch does not require a GPU allocation.
try:
    import torch
except ImportError:
    print("  skip  torch output checks (torch is not installed)")
else:
    a = torch.tensor([1.0], dtype=torch.float64)
    r = equivalence.compare(a, a + 1e-9)
    check("float64 tensor differences survive comparison",
          not r["bit_identical"] and r["max_abs"] > 0, str(r))
    with tempfile.TemporaryDirectory() as td:
        for dtype in (torch.float64, torch.float32, torch.bfloat16):
            output = {"action": torch.tensor([1.25], dtype=dtype)}
            ref = f"{td}/{dtype}"
            equivalence.save_reference(ref, output)
            r = equivalence.check_reference(ref, output)
            check(f"{dtype} reference round-trips exactly", r.get("tier") == "A", str(r))
        r = equivalence.check_reference(ref, {"action": output["action"].float()})
        check("bfloat16 to float32 cannot hide an output dtype change",
              r.get("blocked") == "output dtype mismatch", str(r))

print()
if fails:
    print(f"  {len(fails)} failure(s): {', '.join(fails)}")
    sys.exit(1)
print("  all equivalence reference checks passed")
