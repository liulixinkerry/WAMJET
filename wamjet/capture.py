"""Synchronized endpoint timing and optional diagnostic profiling.

`measure` collects a latency series without a profiler. `capture` also writes a
short trace and its metadata. Check profiler inflation before using scheduling
metrics; overlapping kernel durations do not establish elapsed GPU busy time.
Shape/module recording is opt-in because it can add substantial overhead.
"""
from __future__ import annotations

import json
import os

from .instrument import Timer, node_fingerprint, steady


def measure(infer, warmup: int = 40, *, strict: bool = False,
            progress: bool = True, stall_s: float | None = None) -> dict:
    """Measure synchronized endpoint calls and their steady suffix, without profiling."""
    if warmup < 3:
        raise ValueError("measure needs at least three calls to assess steady timing")
    t = Timer()
    for i in range(warmup):
        with t:
            infer()
        if progress:
            # Print every call. A quiet warmup and a hung process look identical
            # from outside, and a GPU can read 100% busy while a kernel spin-waits
            # on an event that never fires -- so emit a heartbeat you can watch.
            print(f"[capture] call {i + 1}/{warmup}  {t.ms[-1]:8.1f} ms", flush=True)
        if stall_s is not None and t.ms[-1] > stall_s * 1e3:
            raise RuntimeError(
                f"call {i + 1} took {t.ms[-1] / 1e3:.0f}s (> stall_s={stall_s:.0f}s). "
                "Increase stall_s if this workload needs longer calls or compilation.")
    s = steady(t.ms)

    if not s["trusted"]:
        msg = (f"Timing is not steady over {s['n_calls']} calls. Inspect the latency "
               "series for compilation or runtime variability before comparing speedups.")
        if strict:
            raise RuntimeError(msg)
        print("WARNING: " + msg)

    return {"node": node_fingerprint(), "warmup": warmup,
            **{k: v for k, v in s.items() if k != "calls"}, "latency_series": s["calls"]}


def capture(infer, path: str, warmup: int = 40, calls: int = 2,
            record_shapes: bool = False, with_modules: bool = False,
            strict: bool = False, progress: bool = True,
            stall_s: float | None = None, activities: str = "both") -> dict:
    """Measure the endpoint, then collect an optional diagnostic trace and metadata.

    Both measurement and profiling retain the caller's execution mode.
    """
    import torch

    s = measure(infer, warmup, strict=strict, progress=progress, stall_s=stall_s)

    # `activities` decides which questions this trace can answer.
    #   "both" -> CPU + CUDA. Gives cpu_op spans, so you get sync attribution,
    #             compiled-region membership and per-op ownership. But CPU-side
    #             instrumentation grows with launch count and can inflate timings
    #             even with record_shapes and with_modules both off. Compare with
    #             unprofiled endpoint timing before using wall-derived metrics.
    #   "cuda" -> CUDA only. No cpu_op spans, so no attribution. This can reduce
    #             overhead, but still needs the same inflation check before
    #             using idle / GPU-busy / inter-kernel gap measurements.
    act = ([torch.profiler.ProfilerActivity.CUDA] if activities == "cuda"
           else [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
    import time as _t
    _p0 = _t.perf_counter()
    with torch.profiler.profile(activities=act, record_shapes=record_shapes,
                                with_modules=with_modules) as prof:
        for _ in range(calls):
            infer()
        torch.cuda.synchronize()
    profiled_ms = (_t.perf_counter() - _p0) * 1e3 / calls
    prof.export_chrome_trace(path)

    # The instrument must not distort what it measures. Compare the profiled
    # per-call wall against the steady-state median we just established.
    inflation = profiled_ms / s["median_ms"] if s["median_ms"] else float("inf")
    if inflation > 1.25:
        print(f"[capture] *** PROFILER INFLATION {inflation:.1f}x "
              f"({s['median_ms']:.1f} ms steady -> {profiled_ms:.1f} ms profiled). "
              f"Kernel DURATIONS remain valid; every wall-derived metric (idle, "
              f"GPU-busy %, launch lead, sync starvation) is an artifact of the "
              f"instrument on this trace. Re-capture with record_shapes=False and "
              f"with_modules=False before trusting them. ***", flush=True)

    rec = {"trace": path, "calls": calls, "warmup": warmup,
           "profiled_ms_per_call": profiled_ms, "profiler_inflation": inflation,
           "wall_metrics_trustworthy": inflation <= 1.25,
           "record_shapes": record_shapes, "with_modules": with_modules,
           "activities": activities,
           **s}
    meta = path + ".meta.json"
    with open(meta, "w") as fh:
        json.dump(rec, fh, indent=1)
    print(f"[capture] {s['median_ms']:.1f} ms median over {s['steady_n']} steady calls; "
          f"cliff at {s['warmup_calls']}")
    print(f"[capture] wrote {path} ({os.path.getsize(path)/1e6:.1f} MB) and {meta}")
    return rec
