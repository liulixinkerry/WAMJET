"""Time a hot path when the profiler cannot: CUDA events, near-zero overhead.

Profiler overhead scales with recorded events and can dominate a compiled WAM
call. When that happens, wall-derived trace timings describe the instrument.

CUDA events sidestep it. `torch.cuda.Event(enable_timing=True)` records a
timestamp *in the stream*, costs a few microseconds, and reports true device
elapsed time between two points. You lose per-kernel attribution and gain the
ability to measure at all.

Use it to answer the questions the trace cannot:

  * how long a PHASE takes on the device (encode / prefill / each denoise step)
  * what a suspected sync actually costs -- time the region with and without it
  * whether in-stream span and wall time diverge at the endpoint boundary

    from wamjet.events import phase, report_phases
    with phase("text_encode"): model.encode(prompt)
    with phase("denoise"):     model.denoise(x)
    report_phases()

Phases nest; a phase re-entered many times (a denoise step) accumulates, and
both its total and its per-entry mean are reported.
"""
from __future__ import annotations

import statistics as st
from collections import defaultdict
from contextlib import contextmanager

_records: dict[str, list[float]] = defaultdict(list)
_pending: list = []


@contextmanager
def phase(name: str, enabled: bool = True):
    """Time a region on the device. Events are recorded in-stream; nothing syncs
    until `report_phases()`, so timing one region does not serialise the next."""
    if not enabled:
        yield
        return
    import torch
    if not torch.cuda.is_available():
        yield
        return
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    try:
        yield
    finally:
        end.record()
        _pending.append((name, start, end))


def report_phases(reset: bool = True, wall_ms: float | None = None) -> dict:
    """Synchronise once, then resolve every recorded pair.

    Deliberately one sync at the end rather than one per phase -- syncing per
    phase would itself drain the launch queue and change what you are measuring,
    which is the exact failure this module exists to avoid.
    """
    import torch
    torch.cuda.synchronize()
    for name, start, end in _pending:
        _records[name].append(start.elapsed_time(end))
    _pending.clear()

    out = {}
    for name, xs in _records.items():
        out[name] = {"n": len(xs), "total_ms": sum(xs),
                     "mean_ms": st.mean(xs), "max_ms": max(xs)}
    total = sum(v["total_ms"] for v in out.values())
    print(f"  {'ms total':>10} {'n':>6} {'ms each':>9}  {'% of phases':>11}  phase")
    for name, v in sorted(out.items(), key=lambda x: -x[1]["total_ms"]):
        print(f"  {v['total_ms']:>10.2f} {v['n']:>6} {v['mean_ms']:>9.3f}  "
              f"{v['total_ms']/total*100:>10.1f}%  {name}")
    if wall_ms:
        print(f"\n  phases sum to {total:.2f} ms of a {wall_ms:.2f} ms call "
              f"-> {total/wall_ms*100:.0f}% accounted, "
              f"{wall_ms - total:.2f} ms ({(wall_ms-total)/wall_ms*100:.0f}%) outside any phase.")
        print("  Time outside every phase is host-side work or a stall -- that is where")
        print("  a sync costs you, and it is invisible to a per-kernel breakdown.")
    if reset:
        _records.clear()
    return out


def time_sync_cost(infer, warmup: int = 5, reps: int = 20) -> dict:
    """Measure the gap between device time and wall time for one callable.

    device_ms is what the GPU spent; wall_ms is what the caller waited. The
    difference is stall -- syncs, host work, launch gaps -- without needing a
    trace at all, and without the instrument distorting the thing it measures.
    """
    import time
    import torch
    for _ in range(warmup):
        infer()
    torch.cuda.synchronize()

    walls, devs = [], []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        start.record()
        infer()
        end.record()
        torch.cuda.synchronize()
        walls.append((time.perf_counter() - t0) * 1e3)
        devs.append(start.elapsed_time(end))

    w, d = st.median(walls), st.median(devs)
    return {"wall_ms": w, "device_span_ms": d, "stall_ms": w - d,
            "stall_pct": (w - d) / w * 100, "reps": reps,
            "label": "outside-stream host time -- NOT GPU idle",
            "note": "device_span is the in-stream elapsed time across the whole call, so it "
                    "INCLUDES gaps between kernels. wall - device_span is therefore host-side "
                    "time outside the stream: the launch prologue and any CPU work that ran "
                    "before the first kernel or after the last."}
