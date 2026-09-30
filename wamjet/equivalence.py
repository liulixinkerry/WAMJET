"""Compare original and candidate outputs across optimization trials.

A reports numerical agreement within declared limits (exact by default), B drift
within the baseline noise floor, and C larger drift. These are output checks;
Tier-A method eligibility also requires mathematical equivalence. Approximate
methods need paired task evaluation. References preserve output values and dtypes.

`graph_diff` compares kernel-name histograms, not graph identity or latency.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Mapping

INPUT_DIGEST_VERSION = 2


# ---------------------------------------------------------------- numerics --

def compare(a, b) -> dict:
    """Difference between two output tensors (torch or numpy)."""
    import numpy as np
    if hasattr(a, "detach") and hasattr(b, "detach") and a.dtype != b.dtype:
        raise ValueError(f"dtype mismatch {a.dtype} vs {b.dtype}")
    a, b = _numpy(a), _numpy(b)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    if a.dtype != b.dtype:
        raise ValueError(f"dtype mismatch {a.dtype} vs {b.dtype}")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("nonfinite outputs")
    identical = a.tobytes() == b.tobytes()
    dtype = np.result_type(a.dtype, b.dtype, np.float64)
    a, b = a.astype(dtype), b.astype(dtype)
    d = np.abs(a - b)
    na = float(np.linalg.norm(a))
    return {
        "max_abs": float(d.max()) if d.size else 0.0,
        "mean_abs": float(d.mean()) if d.size else 0.0,
        "rel_l2": float(np.linalg.norm(a - b) / max(na, 1e-30)),
        "cosine": float(np.vdot(a.ravel(), b.ravel()).real /
                        (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)),
        "bit_identical": identical,
    }


def noise_floor(run, inputs, repeats: int = 3) -> dict:
    """Run the UNMODIFIED model repeatedly on identical seeded input.

    Whatever spread comes back is the floor: a candidate that stays under it is
    indistinguishable from the baseline talking to itself.
    """
    # Policies may return a reused host buffer; preserve each observed value.
    if repeats < 2:
        raise ValueError("noise_floor needs at least two repeats")
    outs = [{k: a.copy() for k, a in _arrays(run(inputs)).items()}
            for _ in range(repeats)]
    keys = sorted(outs[0])
    if not keys or any(set(o) != set(keys) for o in outs[1:]):
        raise ValueError("noise_floor needs the same nonempty output keys on every repeat")
    per_key = {}
    for key in keys:
        diffs = [compare(outs[0][key], o[key]) for o in outs[1:]]
        per_key[key] = {
            "repeats": repeats,
            "max_abs": max(d["max_abs"] for d in diffs),
            "rel_l2": max(d["rel_l2"] for d in diffs),
            "deterministic": all(d["bit_identical"] for d in diffs),
        }
    return {
        "repeats": repeats,
        "keys": keys,
        "per_key": per_key,
        # Aggregates are for display; they cannot certify an individual output.
        "max_abs": max(f["max_abs"] for f in per_key.values()),
        "rel_l2": max(f["rel_l2"] for f in per_key.values()),
        "deterministic": all(f["deterministic"] for f in per_key.values()),
    }


def classify(diff: dict, floor: dict | None = None,
             tier_a_tolerance: dict | None = None) -> dict:
    """Assign a numerical tier using one output's declared limits."""
    if diff["bit_identical"]:
        return {"tier": "A", "owes": None,
                "note": "zero measured drift"}
    if tier_a_tolerance is not None:
        limits = {k: float(tier_a_tolerance[k]) for k in ("max_abs", "rel_l2")}
        if any(v < 0 or not math.isfinite(v) for v in limits.values()):
            raise ValueError("Tier-A tolerances must be finite and non-negative")
        if all(diff[k] <= limit for k, limit in limits.items()):
            return {"tier": "A", "owes": None, "note": "within declared Tier-A tolerance"}
    if floor is not None and "per_key" in floor:
        floors = floor["per_key"]
        floor = next(iter(floors.values())) if len(floors) == 1 else None
    elif floor is not None and len(floor.get("keys", [])) > 1:
        floor = None
    if floor is None:
        return {"tier": "C", "owes": "G3 paired closed-loop eval",
                "note": "output moved and no noise floor was measured; "
                        "measure the floor before assuming this is kernel drift"}
    if diff["max_abs"] <= floor["max_abs"] and diff["rel_l2"] <= max(floor["rel_l2"], 1e-6):
        return {"tier": "B", "owes": "G3 paired task evaluation",
                "note": "within the baseline noise floor; task quality unvalidated"}
    # A model that is bit-reproducible on identical input has a floor of exactly
    # zero, and then the tier B test degenerates: every change that moves the
    # output at all is outside the floor, and a ratio against zero is a number
    # with no meaning. Say that instead of dividing.
    if floor["max_abs"] == 0.0:
        return {"tier": "C", "owes": "G3 paired closed-loop eval",
                "note": f"drift {diff['max_abs']:.3e} (rel_l2 {diff['rel_l2']:.3e}), "
                        f"against a noise floor of exactly zero over "
                        f"{floor.get('repeats', '?')} repeats. This build is "
                        "deterministic on this input, so the floor cannot certify a "
                        "reorder and no drift can reach tier B by that test. Judge the "
                        "magnitude on its own terms, or settle it at G3"}
    ratio = diff["max_abs"] / floor["max_abs"]
    return {"tier": "C", "owes": "G3 paired closed-loop eval",
            "note": f"drift {diff['max_abs']:.3e} is {ratio:.1f}x the noise floor "
                    f"{floor['max_abs']:.3e} -- this is a numerics change"}


def is_accepted(result: dict, maximum_tier: str = "B",
                task_evaluation: dict | None = None) -> bool:
    """Apply numerical policy; approximate tiers need passing task evidence.

    This does not change the measured tier or erase the G3 obligation from the
    artifact. Callers provide the paired task decision for this candidate.
    """
    tiers = {"A": 0, "B": 1, "C": 2}
    if (not isinstance(result, Mapping) or result.get("blocked") or result.get("advisories")
            or result.get("ok") is False or maximum_tier not in tiers):
        return False
    if any(not math.isfinite(d["max_abs"]) for d in result.get("diffs", {}).values()):
        return False
    if task_evaluation is not None and (not isinstance(task_evaluation, Mapping)
                                        or task_evaluation.get("passed") is not True):
        return False
    tier = result.get("tier")
    if tier not in tiers or tiers[tier] > tiers[maximum_tier]:
        return False
    if tier in ("B", "C"):
        return (isinstance(task_evaluation, Mapping)
                and task_evaluation.get("passed") is True)
    return not result.get("owes")


# ------------------------------------------------------------- references --
#
# `compare` needs both outputs in memory, but optimization work spans processes:
# the baseline runs, the source is edited, and the candidate runs in a fresh
# interpreter.  A gate that can only be used inside one process is a gate that
# does not get used, so the reference has to survive on disk.


def _numpy(value):
    """Preserve numeric precision; bfloat16 converts exactly to float32 for NumPy."""
    import numpy as np
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if str(value.dtype) == "torch.bfloat16":
            value = value.float()
        value = value.numpy()
    array = np.asarray(value)
    if array.dtype.kind not in "biufc":
        raise TypeError(f"unsupported output dtype {array.dtype}; select numeric outputs in the adapter")
    return array


def _leaves(outputs, path="") -> dict:
    """Flatten numeric outputs, including nested containers and scalar leaves."""
    if isinstance(outputs, Mapping):
        children = []
        for key, value in outputs.items():
            if not isinstance(key, str):
                raise TypeError("output mapping keys must be strings")
            children.append((key.replace("~", "~0").replace("/", "~1"), value))
    elif isinstance(outputs, (list, tuple)):
        children = [(str(i) if path else f"output{i}", value)
                    for i, value in enumerate(outputs)]
    else:
        return {path or "output": outputs}
    if not children:
        raise ValueError("output containers must be nonempty")
    return {key: leaf for name, value in children
            for key, leaf in _leaves(value, f"{path}/{name}" if path else name).items()}


def _arrays(outputs) -> dict:
    return {key: _numpy(value) for key, value in _leaves(outputs).items()}


def _dtypes(outputs) -> dict:
    # Retain the public tensor dtype even when NumPy needs a lossless conversion.
    return {key: str(value.dtype) if hasattr(value, "dtype") else str(_numpy(value).dtype)
            for key, value in _leaves(outputs).items()}


def digest(obj) -> str:
    """Typed content hash of nested inputs, including text and scalar options.

    Supports mappings, lists, tuples, Python scalars, NumPy values and dense
    tensors. Unsupported values are errors, never silently omitted.
    """
    import hashlib
    import struct
    import numpy as np

    def tagged(tag, *parts):
        h = hashlib.sha256(tag.encode() + b"\0")
        for part in parts:
            h.update(len(part).to_bytes(8, "big"))
            h.update(part)
        return h.digest()

    def visit(value):
        if value is None:
            return tagged("none")
        if isinstance(value, np.generic):
            if value.dtype.hasobject:
                raise TypeError("cannot hash object-valued NumPy inputs")
            return tagged("numpy-scalar", str(value.dtype).encode(), value.tobytes())
        if isinstance(value, bool):
            return tagged("bool", bytes([value]))
        if isinstance(value, int):
            return tagged("int", str(value).encode())
        if isinstance(value, float):
            return tagged("float", struct.pack("!d", value))
        if isinstance(value, str):
            return tagged("str", value.encode())
        if isinstance(value, bytes):
            return tagged("bytes", value)
        if hasattr(value, "detach"):
            import torch
            tensor = value.detach().cpu().contiguous()
            raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
            return tagged("tensor", str(tensor.dtype).encode(), visit(tuple(tensor.shape)), raw)
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise TypeError("cannot hash object-valued NumPy inputs")
            return tagged("ndarray", str(value.dtype).encode(), visit(value.shape), value.tobytes())
        if isinstance(value, Mapping):
            pairs = sorted((visit(key), visit(item)) for key, item in value.items())
            return tagged("mapping", *(key + item for key, item in pairs))
        if isinstance(value, (list, tuple)):
            return tagged("list" if isinstance(value, list) else "tuple",
                          *(visit(item) for item in value))
        raise TypeError(f"cannot hash input of type {type(value).__name__}")

    return visit(obj).hex()[:16]


def _output_limit(limits, key, keys):
    """Select one output's limits; flat mappings apply only to single outputs."""
    if limits is None:
        return None
    if "per_key" in limits:
        return limits["per_key"].get(key)
    return limits if len(keys) == 1 and len(limits.get("keys", keys)) == 1 else None


def save_reference(path: str, outputs, floor: dict | None = None,
                   inputs=None, note: str = "",
                   tier_a_tolerance: dict | None = None) -> dict:
    """Save original outputs, input identity and fixed numerical limits.

    Pass the model's actual `inputs`. `tier_a_tolerance` holds max_abs/rel_l2
    limits for one output, or a `per_key` mapping for multiple outputs.
    """
    import numpy as np
    from . import preflight
    arrays = _arrays(outputs)
    if any(not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("cannot save nonfinite reference outputs")
    meta = {
        "keys": sorted(arrays),
        "output_dtypes": _dtypes(outputs),
        "floor": floor,
        "tier_a_tolerance": tier_a_tolerance,
        "note": note,
        "input_digest": digest(inputs) if inputs is not None else None,
        "input_digest_version": INPUT_DIGEST_VERSION,
        "fingerprint": preflight.fingerprint(),
    }
    if not path.endswith(".npz"):
        path += ".npz"
    np.savez(path, __meta__=np.array(json.dumps(meta)), **arrays)
    return {"path": path, **meta}


def check_reference(path: str, outputs, inputs=None) -> dict:
    """Compare this process's outputs against a saved reference and tier them.

    Returns the worst tier across output keys. A recorded input digest that does
    not match blocks the comparison outright rather than reporting a tier: a
    difference measured against the wrong input says nothing about the change.
    """
    import numpy as np
    from . import preflight
    if not path.endswith(".npz"):
        path += ".npz"
    ref = np.load(path, allow_pickle=False)
    meta = json.loads(str(ref["__meta__"]))
    try:
        got = _arrays(outputs)
    except (TypeError, ValueError) as exc:
        return {"blocked": "invalid outputs", "note": str(exc)}

    if (meta.get("input_digest")
            and meta.get("input_digest_version") != INPUT_DIGEST_VERSION):
        return {"blocked": "legacy input digest",
                "note": "re-record this reference: its input hash did not cover "
                        "typed text, scalars and nested inputs"}
    if meta.get("input_digest") and inputs is not None:
        d = digest(inputs)
        if d != meta["input_digest"]:
            return {"blocked": "input mismatch",
                    "note": f"reference was recorded on input {meta['input_digest']} "
                            f"but this run used {d}; outputs from different inputs "
                            "are not comparable at any tier"}
    missing = set(meta["keys"]) - set(got)
    if missing:
        return {"blocked": "missing outputs",
                "note": f"reference holds {sorted(missing)} which this run did not return"}
    if set(got) != set(meta["keys"]):
        return {"blocked": "extra outputs", "note": "candidate output keys changed"}
    if "output_dtypes" not in meta:
        return {"blocked": "legacy output reference",
                "note": "re-record the original build: old references discarded output dtypes and nested values"}
    if _dtypes(outputs) != meta["output_dtypes"]:
        return {"blocked": "output dtype mismatch"}
    try:
        diffs = {k: compare(ref[k], got[k]) for k in meta["keys"]}
    except ValueError as exc:
        return {"blocked": "invalid outputs", "note": str(exc)}
    floor = meta.get("floor")
    tolerance = meta.get("tier_a_tolerance")
    tiers = {
        k: classify(d, _output_limit(floor, k, meta["keys"]),
                    _output_limit(tolerance, k, meta["keys"]))
        for k, d in diffs.items()
    }
    worst = max(diffs, key=lambda k: (tiers[k]["tier"], diffs[k]["max_abs"]))
    verdict = tiers[worst]

    drift = preflight.compare(meta["fingerprint"], preflight.fingerprint())
    advisories = []
    if not drift.get("comparable", True):
        advisories.append(
            "the stack moved since the reference was recorded, so bit-identity is "
            "not expected and a tier A result cannot be claimed from it; re-record "
            "the reference on this stack before reading the tier")
    return {"tier": verdict["tier"], "owes": verdict["owes"], "note": verdict["note"],
            "worst_key": worst, "diffs": diffs, "advisories": advisories,
            "floor": floor, "tier_a_tolerance": tolerance}


def check_sequence(run, cases, *, maximum_tier: str = "B",
                   task_evaluation: dict | None = None) -> dict:
    """Check named inputs against stock and replay them within saved tolerances.

    Each case has `name`, `inputs`, and `reference`; use A -> B -> A to check state.
    Repeated names must reproduce their first output within Tier-A limits, even
    when passing task evidence allows Tier B/C reference comparisons.
    """
    cases = list(cases)
    if len(cases) < 2:
        raise ValueError("check_sequence needs at least two named cases")

    pattern = []
    comparisons = []
    replays = []
    first_outputs = {}
    first_indices = {}
    ok = True

    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise TypeError(f"sequence case {index} must be a mapping")
        missing = {"name", "inputs", "reference"} - set(case)
        if missing:
            raise ValueError(f"sequence case {index} is missing {sorted(missing)}")
        name = str(case["name"])
        if not name:
            raise ValueError(f"sequence case {index} has an empty name")

        output = run(case["inputs"])
        reference = check_reference(
            str(case["reference"]), output, inputs=case["inputs"]
        )
        record = {"name": name, "index": index, **reference}
        comparisons.append(record)
        pattern.append(name)
        if not is_accepted(reference, maximum_tier, task_evaluation):
            ok = False

        arrays = _arrays(output)
        if name not in first_outputs:
            # CUDA graphs often replay into one static output allocation. Keep
            # the value observed here, not a view later calls can mutate.
            first_outputs[name] = {key: value.copy() for key, value in arrays.items()}
            first_indices[name] = index
            continue

        original = first_outputs[name]
        same_keys = set(original) == set(arrays)
        diffs = {}
        if same_keys:
            try:
                diffs = {key: compare(original[key], arrays[key]) for key in sorted(original)}
            except ValueError as exc:
                diffs = {"error": str(exc)}
        valid = bool(same_keys and diffs and "error" not in diffs)
        tolerance = comparisons[first_indices[name]].get("tier_a_tolerance")
        replay_lossless = valid and all(
            classify(diff, tier_a_tolerance=_output_limit(tolerance, key, original))["tier"] == "A"
            for key, diff in diffs.items())
        replays.append({
            "name": name,
            "first_index": first_indices[name],
            "index": index,
            "lossless": replay_lossless,
            "bit_identical": valid and all(d["bit_identical"] for d in diffs.values()),
            "diffs": diffs,
        })
        ok = ok and replay_lossless

    return {
        "ok": ok,
        "pattern": pattern,
        "comparisons": comparisons,
        "replays": replays,
    }


# ------------------------------------------------------------ graph identity --

def _kernels(path):
    ev = json.load(open(path))["traceEvents"]
    return Counter(e["name"] for e in ev
                   if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
                   and e.get("dur", 0) > 0)


def graph_diff(trace_a: str, trace_b: str) -> dict:
    """Compare kernel-name histograms, without inferring graph identity."""
    ka, kb = _kernels(trace_a), _kernels(trace_b)
    added = {k: v for k, v in (kb - ka).items()}
    removed = {k: v for k, v in (ka - kb).items()}
    return {
        "events_a": sum(ka.values()), "events_b": sum(kb.values()),
        "delta": sum(kb.values()) - sum(ka.values()),
        "same_kernel_histogram": not added and not removed,
        "added": dict(sorted(added.items(), key=lambda x: -x[1])[:15]),
        "removed": dict(sorted(removed.items(), key=lambda x: -x[1])[:15]),
    }


def main():
    p = argparse.ArgumentParser(description="Compare kernel-name histograms in two traces.")
    p.add_argument("trace_a")
    p.add_argument("trace_b")
    a = p.parse_args()
    r = graph_diff(a.trace_a, a.trace_b)
    print(f"  A {r['events_a']:>9,} device events")
    print(f"  B {r['events_b']:>9,} device events   (delta {r['delta']:+,})")
    if r["same_kernel_histogram"]:
        print("\n  SAME KERNEL HISTOGRAM. Names and counts match; shapes, launch "
              "parameters, ordering and dependencies may differ. This does not "
              "establish graph identity or latency parity.")
        return
    if r["delta"] == 0:
        print("\n  SAME KERNEL COUNT, DIFFERENT HISTOGRAM. Inspect the changed "
              "kernels and measure latency; counts alone do not quantify work.")
    if r["removed"]:
        print("\n  kernels only in A (removed):")
        for k, v in r["removed"].items():
            print(f"    {v:>7,}  {k[:100]}")
    if r["added"]:
        print("\n  kernels only in B (added):")
        for k, v in r["added"].items():
            print(f"    {v:>7,}  {k[:100]}")


if __name__ == "__main__":
    main()
