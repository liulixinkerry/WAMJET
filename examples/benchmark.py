"""Shared example runner. Adapters own loading, inputs, inference and state reset."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Callable


@dataclass
class Adapter:
    infer: Callable
    inputs: object
    metadata: dict
    reset: Callable | None = None


def sync_sites(infer):
    import torch
    import warnings
    from collections import Counter

    previous = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("warn")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            infer()
        sites = Counter(f"{w.filename}:{w.lineno}" for w in caught
                        if "synchron" in str(w.message).lower())
        return {"count": sum(sites.values()), "sites": dict(sites)}
    finally:
        torch.cuda.set_sync_debug_mode(previous)
        torch.cuda.synchronize()


def run(build, *, tag, out_dir, warmup=40, calls=2, reference="",
        record_reference=False, activities="both", probe_syncs=False, events=False,
        maximum_tier="B", task_evaluation=None, tier_a_tolerance=None, screening=False,
        profile=False, sample_power=False):
    """Benchmark an adapter; tier_a_tolerance applies when recording a reference."""
    run_started = time.perf_counter()
    from wamjet import equivalence, preflight
    from wamjet.capture import capture, measure
    from wamjet.instrument import PowerSampler
    import torch

    if maximum_tier not in ("A", "B", "C"):
        raise ValueError("maximum_tier must be A, B, or C")
    policy = {"numerical_policy": {"maximum_tier": maximum_tier}}
    if screening:
        policy["evaluation_mode"] = "screening"
    if task_evaluation is not None:
        policy["task_evaluation"] = task_evaluation

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def save(suffix, value):
        (out / f"{tag}.{suffix}.json").write_text(json.dumps(value, indent=2))

    # Catch a mistyped reference path before loading a large model.
    if record_reference and not reference:
        raise ValueError("REF_MODE=record requires REF")
    if screening and (not reference or record_reference):
        raise ValueError("screening requires an existing original reference in check mode")
    if reference:
        ref_path = Path(reference if reference.endswith(".npz") else reference + ".npz")
        if record_reference and ref_path.exists():
            raise FileExistsError(f"reference already exists: {ref_path}")
        if not record_reference and not ref_path.exists():
            raise FileNotFoundError(f"reference missing: {ref_path}; use REF_MODE=record on the baseline")

    fingerprint = preflight.fingerprint()
    save("env", {"fingerprint": fingerprint})
    print(f"[G-1] {fingerprint.get('host')} {fingerprint.get('gpu')} "
          f"torch {fingerprint.get('torch')}", flush=True)
    build_started = time.perf_counter()
    adapter = build()
    torch.cuda.synchronize()  # Include queued device transfers in build/load time.
    build_seconds = time.perf_counter() - build_started
    save("startup", {"build_seconds": build_seconds})
    print(f"[R0] adapter build/load: {build_seconds:.2f} s", flush=True)
    print(f"[cfg] {adapter.metadata}", flush=True)
    validator = (lambda output: equivalence.check_reference(
        reference, output, inputs=adapter.inputs)) if reference and not record_reference else None
    smoke = preflight.smoke(adapter.infer, validate=validator,
                           maximum_tier=maximum_tier, task_evaluation=task_evaluation)
    startup = {"build_seconds": build_seconds, "first_inference_seconds": smoke["seconds"]}
    save("startup", startup)
    save("smoke", {**smoke, **policy, "fresh_process": True, "calls_before": 0})
    if not smoke["ok"] and "validation" not in smoke:
        print(f"[G-1] {smoke['error']}", flush=True)
        return 1
    numeric = smoke.get("validation")
    if numeric is not None:
        save("equivalence", numeric)
        print(f"[G2] {numeric}", flush=True)
    elif record_reference:
        def sample(_):
            if adapter.reset:
                adapter.reset()
            output = adapter.infer()
            torch.cuda.synchronize()
            return output

        floor = equivalence.noise_floor(sample, None, repeats=3)
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        saved = equivalence.save_reference(reference, sample(None), floor=floor,
                                           inputs=adapter.inputs, note=tag,
                                           tier_a_tolerance=tier_a_tolerance)
        save("reference", saved)
        print(f"[G2] reference saved; floor max_abs={floor['max_abs']:.3e}", flush=True)

    # Reference sampling must not change the measured sequence's starting state.
    if adapter.reset:
        adapter.reset()
    power = PowerSampler() if sample_power else None
    if power:
        power.start()
    started = time.perf_counter()
    try:
        if profile:
            rec = capture(adapter.infer, str(out / f"{tag}_trace.json"),
                          warmup=warmup, calls=calls, strict=False, activities=activities)
        else:
            rec = measure(adapter.infer, warmup=warmup, strict=False)
    finally:
        if power:
            power.stop_flag.set()
            power.join(timeout=6)
    counters = getattr(sys.modules.get("torch._dynamo.utils"), "counters", {})
    rec.update({**policy, "adapter": adapter.metadata, "fingerprint": fingerprint,
                "startup": startup,
                "graph_breaks": dict(counters.get("graph_break", {})),
                "dynamo_stats": dict(counters.get("stats", {})),
                "total_s": time.perf_counter() - started})
    if power:
        rec["power"] = power.summary()
    if numeric is not None:
        rec["equivalence"] = numeric
    if probe_syncs:
        rec["sync_sites"] = sync_sites(adapter.infer)
    if events:
        from wamjet.events import time_sync_cost
        rec["events"] = time_sync_cost(adapter.infer, warmup=5, reps=20)
    accepted = numeric is None or equivalence.is_accepted(numeric, maximum_tier, task_evaluation)
    if screening:
        # Screening may measure approximate drift without G3, but never convert it
        # into a passing correctness gate. Preserve the original smoke/G2 verdicts.
        diffs = (numeric or {}).get("diffs", {})
        valid_outputs = bool(diffs) and all(
            math.isfinite(diff["max_abs"]) for diff in diffs.values())
        tier_allowed = (numeric or {}).get("tier") in {
            "A": ("A",), "B": ("A", "B"), "C": ("A", "B", "C")}[maximum_tier]
        screening_passed = bool(
            rec["trusted"] and valid_outputs and tier_allowed
            and not numeric.get("blocked") and not numeric.get("advisories")
            and (task_evaluation is None or task_evaluation.get("passed") is True))
        rec.update(screening_passed=screening_passed, numerical_accepted=accepted,
                   validation_status="provisional" if screening_passed else "failed-screening")
    rec["run_seconds"] = time.perf_counter() - run_started
    save("result", rec)
    print(f"[G0] {tag}: {rec['median_ms']:.1f} ms; trusted={rec['trusted']}", flush=True)
    if screening:
        print(f"[screening] passed={screening_passed}; {rec['validation_status']}", flush=True)
        return 0 if screening_passed else 1
    return 0 if rec["trusted"] and accepted else 1


def revision(tree):
    return subprocess.run(["git", "-C", tree, "rev-parse", "HEAD"],
                          capture_output=True, text=True, timeout=10).stdout.strip()


def main(build, *, prefix, root, warmup=40, calls=2):
    mode = os.getenv("REF_MODE", "check")
    if mode not in ("record", "check"):
        raise ValueError("REF_MODE must be record or check")
    task_path = os.getenv("TASK_EVALUATION", "")
    task_evaluation = json.loads(Path(task_path).read_text()) if task_path else None
    tolerance_path = os.getenv("TIER_A_TOLERANCE", "")
    tier_a_tolerance = json.loads(Path(tolerance_path).read_text()) if tolerance_path else None
    return run(build, tag=os.getenv("TAG", prefix.lower()),
               out_dir=os.getenv("OUT_DIR", os.path.join(root, "ledger")),
               warmup=int(os.getenv("WARMUP", str(warmup))),
               calls=int(os.getenv("PROF_CALLS", str(calls))),
               reference=os.getenv("REF", os.getenv(f"{prefix}_REF", "")),
               maximum_tier=os.getenv("MAXIMUM_TIER", "B"),
               tier_a_tolerance=tier_a_tolerance,
               screening=os.getenv("SCREENING", "false").lower() == "true",
               profile=os.getenv("PROFILE", "false").lower() in ("1", "true"),
               sample_power=os.getenv("POWER", "false").lower() in ("1", "true"),
               task_evaluation=task_evaluation,
               record_reference=mode == "record",
               activities=os.getenv("ACTIVITIES", os.getenv(f"{prefix}_ACTIVITIES", "both")),
               probe_syncs=os.getenv(f"{prefix}_SYNC_DEBUG", "false").lower() == "true",
               events=os.getenv(f"{prefix}_EVENTS", "false").lower() == "true")
