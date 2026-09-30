"""Gate G1, attribution half -- which line of the model owns the time?

`wamjet.pipeline` answers "is the pipeline healthy". This answers "where is
the time going", and it is the tool you actually plan an optimization from.

A kernel bucket is not an answer. Module and operation ownership connects device
time to source and makes the next experiment concrete.

The link is in the trace already. Every device kernel carries a `correlation`
id back to the CUDA runtime call that launched it, and that runtime call sits
inside a stack of `cpu_op` spans on the launching thread. The innermost span is
the aten op; an enclosing `nn.Module: ...` span, present when the trace was
captured `with_modules=True`, is the layer. Walking that stack turns device time
into source ownership.

Usage:
    python -m wamjet.attribute TRACE.json [--calls N] [--by module|op|shape]
"""
from __future__ import annotations

import argparse
import heapq
import json
from collections import defaultdict

DEVICE_CATS = ("kernel", "gpu_memcpy", "gpu_memset")


def _launch_stacks(cpu_ops, launches):
    """Sweep each thread once, retaining only spans active at each launch.

    Sort equal-start spans outermost first. Index by correlation so kernels
    sharing a graph launch reuse its stack, regardless of device-event order.
    """
    per = defaultdict(list)
    for i, e in enumerate(cpu_ops):
        per[e.get("tid")].append((e["ts"], 0, -e["dur"], i, e))
    for i, e in enumerate(launches.values()):
        per[e.get("tid")].append((e["ts"], 1, 0, i, e))
    stacks = {}
    for events in per.values():
        active, ends = {}, []
        for ts, is_launch, _, i, e in sorted(events):
            while ends and ends[0][0] < ts:
                _, expired = heapq.heappop(ends)
                del active[expired]
            if is_launch:
                stacks[e["args"]["correlation"]] = tuple(active.values())
            else:
                end = ts + e["dur"]
                active[i] = (ts, end, e["name"], e)
                heapq.heappush(ends, (end, i))
    return stacks


def attribute(path: str, calls: int = 1) -> dict:
    ev = json.load(open(path))["traceEvents"]
    dev = [e for e in ev if e.get("cat") in DEVICE_CATS and e.get("dur", 0) > 0]
    # BOTH APIs: torch launches through cuda_runtime, Triton through cuda_driver.
    # Reading only one loses ~75% of the kernels in a compiled model.
    rt = {e["args"]["correlation"]: e for e in ev
          if e.get("cat") in ("cuda_runtime", "cuda_driver")
          and "correlation" in e.get("args", {})}
    cpu = [e for e in ev if e.get("cat") == "cpu_op" and e.get("dur", 0) > 0]
    stacks = _launch_stacks(cpu, rt)

    by_module = defaultdict(lambda: [0.0, 0])
    by_op = defaultdict(lambda: [0.0, 0])
    by_shape = defaultdict(lambda: [0.0, 0])
    unattributed = [0.0, 0]

    for k in dev:
        d = k["dur"]
        stack = stacks.get(k.get("args", {}).get("correlation"))
        if not stack:
            unattributed[0] += d
            unattributed[1] += 1
            continue

        # Innermost aten op = what ran; deepest nn.Module = where it lives.
        aten = next((s[2] for s in reversed(stack) if s[2].startswith("aten::")),
                    stack[-1][2])
        mods = [s[2] for s in stack if s[2].startswith("nn.Module:")]
        owner = mods[-1] if mods else aten          # deepest module, else the op

        by_module[owner][0] += d
        by_module[owner][1] += 1
        by_op[aten][0] += d
        by_op[aten][1] += 1

        inner = next((s[3] for s in reversed(stack) if s[2].startswith("aten::")), None)
        shapes = (inner or {}).get("args", {}).get("Input Dims")
        if shapes:
            by_shape[f"{aten} {shapes}"][0] += d
            by_shape[f"{aten} {shapes}"][1] += 1

    total = sum(e["dur"] for e in dev)
    def pack(d):
        return [{"name": k, "ms": v[0] / 1e3 / calls, "kernels": v[1] // calls,
                 "pct": v[0] / total * 100}
                for k, v in sorted(d.items(), key=lambda x: -x[1][0])]

    return {
        "trace": path, "calls": calls,
        "device_ms_per_call": total / 1e3 / calls,
        "unattributed_ms_per_call": unattributed[0] / 1e3 / calls,
        "unattributed_kernels": unattributed[1] // calls,
        "by_module": pack(by_module),
        "by_op": pack(by_op),
        "by_shape": pack(by_shape),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trace")
    p.add_argument("--calls", type=int, default=1)
    p.add_argument("--by", default="module", choices=["module", "op", "shape"])
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--json", help="write the full record for the ledger")
    a = p.parse_args()

    r = attribute(a.trace, a.calls)
    print(f"{r['trace']}  ({r['calls']} call(s))")
    print(f"  device time     {r['device_ms_per_call']:.1f} ms/call")
    if r["unattributed_kernels"]:
        print(f"  unattributed    {r['unattributed_ms_per_call']:.1f} ms/call "
              f"({r['unattributed_kernels']} kernels) -- launched outside any "
              f"traced cpu_op")
    key = {"module": "by_module", "op": "by_op", "shape": "by_shape"}[a.by]
    rows = r[key][:a.top]
    if not rows:
        print(f"\n  nothing to report by {a.by}."
              + (" Re-capture with with_modules=True." if a.by == "module" else
                 " Re-capture with record_shapes=True." if a.by == "shape" else ""))
        return
    print(f"\n  device time by {a.by} (per call, top {len(rows)}):")
    for x in rows:
        print(f"    {x['ms']:>8.2f} ms  {x['pct']:>5.1f}%  {x['kernels']:>6,} k  {x['name'][:88]}")
    if a.json:
        json.dump(r, open(a.json, "w"), indent=1)
        print(f"\n  wrote {a.json}")


if __name__ == "__main__":
    main()
