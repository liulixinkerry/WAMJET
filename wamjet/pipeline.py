"""WAMJET gate G1 -- pipeline health from one torch-profiler chrome trace.

A WAM call can lose time when the CPU stops feeding the GPU. This reads one
trace and reports five scheduling signals that can be compared across variants:

  1. syncs         device->host reads that drain the launch queue
  2. graph breaks  aten CPU time outside any compiled region
  3. fragmentation how many compiled regions one call enters
  4. queue restart launch lead -- how far ahead of the GPU the CPU is running
  5. idle          inter-kernel gap, which is what the other four produce

The central rule this tool encodes: *a defect only counts if it costs GPU
idle.*  Eager dispatch behind a 30 ms launch lead is free; the same dispatch
behind a 2 us lead is the whole problem.  So every defect is scored by the
device idle that overlaps it, not by its own CPU duration.

Usage:  python -m wamjet.pipeline TRACE.json [--calls N] [--json OUT.json]
"""
from __future__ import annotations

import argparse
import bisect
import json
import re
import statistics as st
from collections import defaultdict

DEVICE_CATS = ("kernel", "gpu_memcpy", "gpu_memset")

# Kernels that occupy the GPU while WAITING rather than computing. A collective
# spins on its peers, a barrier spins on a flag -- both hold SMs and both count
# as "GPU busy" in every utilisation metric, including this tool's own. Time in
# them is not throughput, and a run can look 100% busy while making no progress.
WAIT_KERNELS = ("nccl", "ncclDevKernel", "AllReduce", "AllGather", "ReduceScatter",
                "Broadcast", "barrier", "spin", "Wait", "sync_", "cuStreamWait")

# cpu_ops that read a device value into Python, or whose output shape depends on
# device data. Both force the CPU to wait for the GPU. These are the stalls you own.
IMPLICIT_SYNC = (
    "aten::item", "aten::_local_scalar_dense", "aten::equal", "aten::is_nonzero",
    "aten::nonzero", "aten::unique", "aten::_unique2", "aten::unique_consecutive",
    "aten::masked_select", "aten::bincount", "aten::allclose", "aten::index_put_impl_",
)
# CUDA calls that block the launching thread. BOTH APIs: torch uses the runtime
# API, Triton launches through the driver API, and a trace contains both.
BLOCKING_CALL = (
    "cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize",
    "cuStreamSynchronize", "cuCtxSynchronize", "cuEventSynchronize",
    "cudaMemcpy", "cuMemcpyDtoH", "cuMemcpyHtoD",       # the non-Async forms only
    "cudaHostAlloc", "cudaMallocHost", "cudaFreeHost",  # allocator calls that sync
    "cudaFree", "cuMemFree",
)
_ASYNC = ("cudaMemcpyAsync", "cuMemcpyDtoHAsync", "cuMemcpyHtoDAsync")
# every way a kernel gets launched -- missing the driver API loses ~75% of them
LAUNCH_NAMES = ("cudaLaunchKernel", "cuLaunchKernel", "cuLaunchKernelEx",
                "cudaGraphLaunch", "cuGraphLaunch", "cudaLaunchKernelExC",
                "cudaMemcpyAsync", "cuMemcpyDtoHAsync", "cuMemcpyHtoDAsync",
                "cudaMemsetAsync")
from .fusion import is_vendor          # one definition of "vendor", not two (H12)

REGION_RE = re.compile(r"Torch-Compiled Region|CompiledFunction|Torch-Dynamo", re.I)


def _merge(intervals):
    """Sort and merge (start, end) pairs; returns a disjoint list."""
    if not intervals:
        return []
    iv = sorted(intervals)
    out = [list(iv[0])]
    for s, e in iv[1:]:
        if s > out[-1][1]:
            out.append([s, e])
        else:
            out[-1][1] = max(out[-1][1], e)
    return [(a, b) for a, b in out]


class Cover:
    """Merged intervals with O(log n) containment and overlap queries."""

    def __init__(self, intervals):
        self.iv = _merge(intervals)
        self.starts = [a for a, _ in self.iv]
        self.total = sum(b - a for a, b in self.iv)

    def contains(self, t):
        i = bisect.bisect_right(self.starts, t) - 1
        return i >= 0 and t <= self.iv[i][1]

    def overlap(self, s, e):
        """Length of self intersected with [s, e]."""
        i = bisect.bisect_right(self.starts, s) - 1
        if i < 0:
            i = 0
        tot = 0.0
        while i < len(self.iv) and self.iv[i][0] < e:
            tot += max(0.0, min(e, self.iv[i][1]) - max(s, self.iv[i][0]))
            i += 1
        return tot


def bucket(name: str) -> str:
    n = name.lower()
    if any(k in n for k in ("flash", "fmha", "sdpa", "attention", "attn", "cudnn")):
        return "attention"
    if any(k in n for k in ("nvjet", "gemm", "cutlass", "cublas", "xmma", "gemv", "s16816")):
        return "GEMM"
    if "memcpy" in n or "memset" in n:
        return "memcpy"
    if "triton_poi" in n or "elementwise" in n:
        return "pointwise"
    if "triton_red" in n or "triton_per" in n or "reduce" in n or "norm" in n:
        return "reduction"
    if "triton" in n:
        return "triton other"
    return "other"


def analyse(path: str, calls: int = 1) -> dict:
    # Cross-check against the G0 record `capture` wrote beside the trace. If the
    # profiled window is far larger than the measured steady-state call, the wall
    # clock in this trace belongs to the instrument, not the model -- and every
    # metric derived from it is an artifact that still looks entirely plausible.
    truth = {"trusted": False}
    try:
        meta = json.load(open(path + ".meta.json"))
        truth = {"trusted": meta.get("trusted") is True
                            and meta.get("calls", calls) == calls
                            and isinstance(meta.get("median_ms"), (int, float))
                            and meta["median_ms"] > 0}
        if truth["trusted"] and meta.get("median_ms"):
            truth.update({"steady_ms_per_call": meta["median_ms"],
                          "record_shapes": meta.get("record_shapes"),
                          "with_modules": meta.get("with_modules"),
                          "wall_metrics_trustworthy": meta.get("wall_metrics_trustworthy", True)})
    except Exception:
        pass

    ev = json.load(open(path))["traceEvents"]

    dev = [e for e in ev if e.get("cat") in DEVICE_CATS and e.get("dur", 0) > 0]
    if not dev:
        raise SystemExit(f"{path}: no device events -- was CUDA activity recorded?")
    cpu = [e for e in ev if e.get("cat") == "cpu_op"]
    # Triton launches through the driver API and torch through the runtime API;
    # a trace of a compiled model contains both and either alone is a biased sample.
    rt = [e for e in ev if e.get("cat") in ("cuda_runtime", "cuda_driver")]

    busy = Cover([(e["ts"], e["ts"] + e["dur"]) for e in dev])
    t0 = min(e["ts"] for e in dev)
    t1 = max(e["ts"] + e["dur"] for e in dev)
    span = t1 - t0

    # ---- 4. queue restart: how far ahead of the GPU is the CPU launching? ----
    launch = {e["args"]["correlation"]: e for e in rt if "correlation" in e.get("args", {})}
    paired = [(k, l) for k in dev
              if (l := launch.get(k.get("args", {}).get("correlation"))) is not None]
    lead_at = sorted((l["ts"] + l.get("dur", 0), k["ts"] - (l["ts"] + l.get("dur", 0)))
                     for k, l in paired)
    leads = sorted(x[1] for x in lead_at)
    launch_coverage = len(paired) / len(dev)

    def pct(p):
        return leads[min(len(leads) - 1, int(len(leads) * p))] if leads else None

    # ---- 5. inter-kernel gap on the busiest stream ----
    by_stream = defaultdict(list)
    for e in dev:
        by_stream[e.get("args", {}).get("stream", 0)].append((e["ts"], e["ts"] + e["dur"]))
    main = sorted(max(by_stream.values(), key=len))
    gaps = sorted(g for i in range(len(main) - 1)
                  if (g := main[i + 1][0] - main[i][1]) > 0)

    # ---- 1. syncs, scored by the GPU idle they actually cause ----
    def score_syncs(events):
        out = defaultdict(lambda: {"count": 0, "cpu_ms": 0.0, "gpu_idle_ms": 0.0})
        for e in events:
            d = e.get("dur", 0)
            r = out[e["name"]]
            r["count"] += 1
            r["cpu_ms"] += d / 1e3
            r["gpu_idle_ms"] += (d - busy.overlap(e["ts"], e["ts"] + d)) / 1e3
        return dict(sorted(out.items(), key=lambda x: -x[1]["gpu_idle_ms"]))

    sync_ops = [e for e in cpu if e["name"] in IMPLICIT_SYNC]
    block_ops = [e for e in rt if e["name"].startswith(BLOCKING_CALL)
                 and not e["name"].startswith(_ASYNC)]
    implicit = score_syncs(sync_ops)
    explicit = score_syncs(block_ops)
    d2h = [e for e in dev if "DtoH" in e.get("name", "")]

    # ---- queue drain candidates: observed idle after a sync returns ----
    # Other streams may still be working, and nested CPU/runtime sync spans can
    # describe the same wait. Intersect with actual idle and union intervals
    # before totaling. Adjacency identifies a candidate cause, not proof of one.
    kstarts = sorted(e["ts"] for e in dev)
    idle = Cover([(a[1], b[0]) for a, b in zip(busy.iv, busy.iv[1:])])

    def idle_time(intervals):
        return sum(idle.overlap(s, e) for s, e in _merge(list(intervals)))

    med_lead = st.median(leads) if leads else 0.0
    drains = []
    for e in sync_ops + block_ops:
        end = e["ts"] + e.get("dur", 0)
        i = bisect.bisect_right(kstarts, end)
        if i >= len(kstarts):
            continue
        nxt = kstarts[i]
        starve = idle.overlap(end, nxt)
        drains.append({"op": e["name"], "ts": e["ts"], "starve_us": starve,
                       "interval": (end, nxt),
                       "post_lead_us": nxt - end})
    # Retain candidates whose observed idle exceeds normal kernel turnaround.
    thresh = max(5.0, st.median(gaps) * 2 if gaps else 5.0)
    drained = [d for d in drains if d["starve_us"] > thresh]
    by_op = defaultdict(list)
    for d in drained:
        by_op[d["op"]].append(d)
    drain_rows = [{"op": op, "count": len(ds),
                   "starve_ms": idle_time(d["interval"] for d in ds) / 1e3,
                   "worst_us": max(d["starve_us"] for d in ds)}
                  for op, ds in by_op.items()]

    # ---- 2 & 3. compiled-region coverage and fragmentation ----
    regions = [e for e in cpu if e.get("dur", 0) > 0 and REGION_RE.search(e["name"])]
    reg = Cover([(e["ts"], e["ts"] + e["dur"]) for e in regions])

    # DEVICE-TIME coverage: the honest measure. Attribute each kernel to the
    # region its launch sits in.
    dev_in = dev_out = 0.0
    n_in = n_out = 0
    out_by = defaultdict(lambda: [0, 0.0])
    for k, l in paired:
        if reg.contains(l["ts"]):
            dev_in += k["dur"]; n_in += 1
        else:
            dev_out += k["dur"]; n_out += 1
            r = out_by[k["name"]]
            r[0] += 1; r[1] += k["dur"] / 1e3
    dev_tot = dev_in + dev_out

    # ---- 2b. concurrency: does anything actually run at the same time? ----
    # Sum of kernel durations over the union of their intervals. A factor of
    # 1.000 with more than one stream live means the streams are serialized:
    # work is *placed* concurrently and never *runs* concurrently. Overlapping
    # two independent paths costs no arithmetic, so this is the cheapest
    # remaining lever whenever it is available.
    kern = [e for e in dev if e.get("cat") == "kernel"]
    streams = defaultdict(lambda: [0, 0.0])
    for e in kern:
        row = streams[e.get("args", {}).get("stream", 0)]
        row[0] += 1
        row[1] += e["dur"] / 1e3
    kern_sum = sum(e["dur"] for e in kern)
    kern_union = Cover([(e["ts"], e["ts"] + e["dur"]) for e in kern]).total

    # ---- 2c. fallbacks: ops the compiler declined, inside a compiled region --
    # A kernel launched from inside a compiled region that the compiler did not
    # generate, and that is not a closed vendor library it deliberately calls,
    # is a FALLBACK: the compiler refused the op and emitted an eager kernel
    # inside a region believed to be compiled.
    #
    # Nothing else catches this. It is not a graph break, so `fullgraph=True`
    # passes; the region entry count is unchanged; and under CUDA-graph replay
    # it is hidden from a latency reading. The usual cause is a dtype or op the
    # backend cannot code-generate -- complex arithmetic is the common one.
    fb = defaultdict(lambda: [0, 0.0])
    fb_us = 0.0
    for k, l in paired:
        if k.get("cat") != "kernel" or not reg.contains(l["ts"]):
            continue
        nm = k["name"]
        if "triton" in nm.lower() or is_vendor(nm):
            continue
        f = fb[nm]
        f[0] += 1
        f[1] += k["dur"] / 1e3
        fb_us += k["dur"]

    kern_ms_call = kern_sum / 1e3 / calls
    kern_per_call = len(kern) / calls

    eager_us = in_us = 0.0
    eager_ops = defaultdict(lambda: [0, 0.0, 0.0])   # count, cpu_ms, gpu_idle_ms
    for e in cpu:
        if not e["name"].startswith("aten::"):
            continue
        d = e.get("dur", 0)
        if reg.contains(e["ts"]):
            in_us += d
        else:
            eager_us += d
            r = eager_ops[e["name"]]
            r[0] += 1
            r[1] += d / 1e3
            r[2] += (d - busy.overlap(e["ts"], e["ts"] + d)) / 1e3

    buckets = defaultdict(float)
    for e in dev:
        buckets[bucket(e["name"])] += e["dur"]

    # split busy into productive and waiting
    def is_wait(e):
        n = e["name"]
        if any(w.lower() in n.lower() for w in WAIT_KERNELS):
            return True
        # A kernel holding one or two blocks for a long time is usually polling.
        # But NOT if it is a GEMM or attention kernel: a vendor GEMM on a small
        # matrix legitimately launches a couple of blocks and computes for a
        # while. Applying the heuristic to those flagged two real cutlass GEMMs
        # as spin-waits -- the same mistake as trying to fuse into cuDNN.
        if bucket(n) in ("GEMM", "attention"):
            return False
        g = e.get("args", {}).get("grid") or [0]
        blocks = g[0] if isinstance(g, list) and g else 0
        return bool(blocks and blocks <= 2 and e["dur"] > 1000)

    waiting = [e for e in dev if is_wait(e)]
    wait_cover = Cover([(e["ts"], e["ts"] + e["dur"]) for e in waiting])
    wait_by_name = defaultdict(lambda: [0, 0.0])
    for e in waiting:
        wait_by_name[e["name"]][0] += 1
        wait_by_name[e["name"]][1] += e["dur"] / 1e3

    if truth.get("steady_ms_per_call"):
        infl = (span / 1e3 / calls) / truth["steady_ms_per_call"]
        truth["profiler_inflation"] = infl
        truth["wall_metrics_trustworthy"] = truth["wall_metrics_trustworthy"] and infl <= 1.25

    # Summed durations double-count overlapping streams. Only a trustworthy
    # trace can supply elapsed busy time; inflation leaves this quantity unknown.
    idle_ms_call = us_per_gap = None
    if truth.get("wall_metrics_trustworthy"):
        idle_ms_call = (span - busy.total) / 1e3 / calls
        if kern_per_call:
            us_per_gap = idle_ms_call * 1e3 / kern_per_call

    return {
        "trace": path,
        "calls": calls,
        "ground_truth": truth,
        "device_events": len(dev),
        "device_events_per_call": len(dev) / calls,
        "span_ms": span / 1e3,
        "busy_ms": busy.total / 1e3,
        "busy_pct": busy.total / span * 100,
        "waiting_ms": wait_cover.total / 1e3,
        "waiting_pct_of_busy": wait_cover.total / busy.total * 100 if busy.total else 0.0,
        "productive_pct": (busy.total - wait_cover.total) / span * 100,
        "waiting_kernels": [{"kernel": k, "count": v[0], "ms": v[1]} for k, v in
                            sorted(wait_by_name.items(), key=lambda x: -x[1][1])[:6]],
        "idle_ms_per_call": (span - busy.total) / 1e3 / calls,
        "gap": {
            "count": len(gaps),
            "total_ms_per_call": sum(gaps) / 1e3 / calls,
            "median_us": st.median(gaps) if gaps else 0.0,
            "p99_us": gaps[int(len(gaps) * 0.99)] if gaps else 0.0,
        },
        "launch_lead_us": {
            "n": len(leads),
            "median": st.median(leads) if leads else None,
            "p01": pct(0.01),
            "frac_under_5us": sum(1 for x in leads if x < 5) / len(leads) if leads else None,
            "coverage": launch_coverage,
        },
        "sync": {
            "implicit": implicit,
            "explicit": explicit,
            "implicit_count": sum(v["count"] for v in implicit.values()),
            "implicit_gpu_idle_ms": sum(v["gpu_idle_ms"] for v in implicit.values()),
            "blocking_count": sum(v["count"] for v in explicit.values()),
            "d2h_copies": len(d2h),
        },
        "queue_drain": {
            "checked": len(drains),
            "drained": len(drained),
            "threshold_us": thresh,
            "median_lead_us": med_lead,
            "starve_ms_per_call": idle_time(d["interval"] for d in drained) / 1e3 / calls,
            "worst_starve_us": max((d["starve_us"] for d in drains), default=0.0),
            "by_op": sorted(drain_rows, key=lambda x: -x["starve_ms"]),
            "basis": "union of observed idle after syncs; per-op rows may overlap",
        },
        "compiled": {
            # Headline: fraction of DEVICE time launched from inside a region.
            "device_coverage_pct": dev_in / dev_tot * 100 if dev_tot else None,
            "device_ms_outside": dev_out / 1e3,
            "kernels_outside_per_call": n_out / calls,
            "top_uncompiled_kernels": [
                {"kernel": k, "count": v[0], "ms": v[1]}
                for k, v in sorted(out_by.items(), key=lambda x: -x[1][1])[:6]],
            "distinct_regions": len({e["name"] for e in regions}),
            "entries_per_call": len(regions) / calls,
            "cpu_ms_in_regions": in_us / 1e3,
            "cpu_ms_eager": eager_us / 1e3,
            "coverage_pct": in_us / (in_us + eager_us) * 100 if (in_us + eager_us) else None,
            "eager_gpu_idle_ms": sum(v[2] for v in eager_ops.values()),
        },
        "concurrency": {
            "factor": (kern_sum / kern_union) if kern_union else None,
            "streams_live": len(streams),
            "kernel_ms_sum_per_call": kern_ms_call,
            "kernel_ms_union_per_call": kern_union / 1e3 / calls,
            "by_stream": [{"stream": k, "kernels_per_call": v[0] / calls,
                           "ms_per_call": v[1] / calls}
                          for k, v in sorted(streams.items(), key=lambda x: -x[1][0])],
        },
        "fallbacks": {
            "ms_per_call": fb_us / 1e3 / calls,
            "pct_of_in_region_device_time": (fb_us / dev_in * 100) if dev_in else None,
            "kernels": [{"kernel": k, "count_per_call": v[0] / calls,
                         "ms_per_call": v[1] / calls}
                        for k, v in sorted(fb.items(), key=lambda x: -x[1][1])[:8]],
        },
        "launch_pressure": {
            "kernels_per_call": kern_per_call,
            "idle_ms_per_call": idle_ms_call,
            "us_per_gap": us_per_gap,
            "basis": "observed trace idle / kernel count" if idle_ms_call is not None else
                     "unavailable -- no trustworthy trace timing",
        },
        "top_eager_ops": [
            {"op": k, "count": v[0], "cpu_ms": v[1], "gpu_idle_ms": v[2]}
            for k, v in sorted(eager_ops.items(), key=lambda x: -x[1][2])[:10]
        ],
        "device_ms_by_bucket": dict(sorted(((k, v / 1e3) for k, v in buckets.items()),
                                           key=lambda x: -x[1])),
    }


def _inflation_proof(r: dict) -> list[str]:
    """Structural leads that do not infer scheduling limits from timestamps."""
    out = []
    fb = r.get("fallbacks") or {}
    if fb.get("ms_per_call", 0) > 0.05:
        top = fb["kernels"][0] if fb["kernels"] else None
        out.append(
            f"POSSIBLE COMPILER FALLBACK: {fb['ms_per_call']:.2f} ms/call inside compiled regions "
            f"runs in kernels not recognized as compiler-generated or vendor kernels"
            + (f" (top: {top['kernel'][:48]}, {top['count_per_call']:.0f}/call)" if top else "")
            + f" -- {fb['pct_of_in_region_device_time']:.0f}% of in-region device time. "
            "Check the compiler log and generated code before treating these as fallbacks.")
    return out


def verdict(r: dict) -> list[str]:
    """The numbers, turned into the next action -- or into 'stop scheduling'."""
    out = []
    gt = r.get("ground_truth") or {}
    if gt.get("trusted") is False:
        return ["MEASUREMENT UNVERIFIED: G0 is untrusted or its call count differs. "
                "Establish a comparable steady baseline before using scheduling timings."] + _inflation_proof(r)
    if gt.get("profiler_inflation") and not gt.get("wall_metrics_trustworthy"):
        dev = sum(r["device_ms_by_bucket"].values()) / r["calls"]
        steady = gt["steady_ms_per_call"]
        return [
            f"MEASUREMENT VOID for scheduling metrics: profiler inflation "
            f"{gt['profiler_inflation']:.1f}x. Re-capture with record_shapes=False and "
            f"with_modules=False before drawing any conclusion about idle or syncs. "
            f"Concurrency is void here too: inflation spreads kernel timestamps apart, "
            f"which drives the factor toward 1.0 whether or not the model overlaps.",
            f"Device events total {dev:.1f} ms/call of summed durations against a "
            f"{steady:.1f} ms measured call. Overlapping work is counted repeatedly; "
            f"this ratio cannot establish GPU-busy percentage or elapsed idle time.",
        ] + _inflation_proof(r)
    cov = r["launch_lead_us"].get("coverage")
    if cov is not None and cov < 0.98:
        out.append(f"TRACE INCOMPLETE: only {cov*100:.0f}% of kernels could be matched to a "
                   f"launch call, so the launch-lead figures are a biased subset. Capture "
                   f"with both cuda_runtime and cuda_driver activities recorded.")
    lead, sync, comp = r["launch_lead_us"], r["sync"], r["compiled"]
    idle_pc = r["idle_ms_per_call"]
    covered = lead["median"] is not None and lead["median"] > 100  # CPU well ahead

    dr = r["queue_drain"]
    if dr["drained"]:
        top = dr["by_op"][0]
        out.append(
            f"QUEUE DRAIN CANDIDATES: {dr['drained']} of {dr['checked']} sync points were "
            f"followed by observed GPU idle, totaling {dr['starve_ms_per_call']:.2f} ms/call "
            f"after merging overlaps (worst {dr['worst_starve_us']/1e3:.2f} ms; "
            f"largest per-op total: {top['op']}). Test whether removing these syncs "
            f"reduces the idle; timing adjacency alone does not establish causation.")
    if sync["implicit_count"]:
        w = next(iter(sync["implicit"]))
        out.append(
            f"SYNC ({sync['implicit_count']} implicit reads of a device value, "
            f"{sync['implicit_gpu_idle_ms']:.1f} ms of idle inside them; worst {w}). "
            f"Even when cheap in isolation they cap how far ahead the CPU can run, "
            f"so they gate every other lever.")
    if lead["frac_under_5us"] and lead["frac_under_5us"] > 0.10:
        out.append(
            f"LAUNCH-BOUND: {lead['frac_under_5us']*100:.0f}% of kernels start <5 us after "
            f"their launch returns; the GPU is waiting on the CPU. Widen the compiled region "
            f"or capture a CUDA graph.")
    dcov = comp["device_coverage_pct"]
    if dcov is not None and dcov < 95:
        top = comp["top_uncompiled_kernels"]
        out.append(
            f"COVERAGE: {dcov:.1f}% of device time is launched from inside a compiled "
            f"region; {comp['device_ms_outside']:.1f} ms is not"
            + (f" (worst: {top[0]['count']:,}x {top[0]['kernel'][:44]})" if top else "") + ".")
    if comp["entries_per_call"] > 20 and not covered:
        out.append(
            f"FRAGMENTED: {comp['entries_per_call']:.0f} compiled-region entries per call "
            f"across {comp['distinct_regions']} distinct graphs. Fewer, larger regions "
            f"pipeline better; check for a graph break splitting one logical region.")
    if r["waiting_pct_of_busy"] > 5:
        out.append(
            f"BUSY BUT NOT WORKING: {r['waiting_pct_of_busy']:.0f}% of GPU-busy time "
            f"({r['waiting_ms']:.1f} ms) is spent in kernels that spin-wait rather than "
            f"compute, so real utilisation is {r['productive_pct']:.1f}%, not "
            f"{r['busy_pct']:.1f}%. A utilisation number alone cannot tell these apart.")
    if r["busy_pct"] < 90:
        out.append(
            f"IDLE: {100 - r['busy_pct']:.1f}% of the window has no kernel resident "
            f"({idle_pc:.1f} ms/call over {r['gap']['count']:,} gaps, median "
            f"{r['gap']['median_us']:.2f} us). Test compilation, graph capture or fusion "
            f"against the endpoint; gap size alone does not establish the cause.")
    out += _inflation_proof(r)

    conc = r.get("concurrency") or {}
    if (conc.get("factor") is not None and conc.get("streams_live", 0) > 1
            and conc["factor"] < 1.02):
        out.append(
            f"SERIALIZED: {conc['streams_live']} streams are live but the concurrency "
            f"factor is {conc['factor']:.3f} -- the union of kernel intervals equals their "
            f"sum, so nothing ever runs at the same time. Work is placed on separate "
            f"streams and still executes end to end. If any two paths are independent, "
            f"overlapping them removes wall time at no arithmetic cost.")

    if not out:
        out.append(
            f"CLEAN: {r['busy_pct']:.1f}% GPU-busy, no implicit syncs, "
            f"{comp['device_coverage_pct'] or 0:.0f}% compiled device coverage. "
            f"No scheduling defect was identified in this trace; inspect remaining "
            f"device work before choosing the next experiment.")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trace")
    p.add_argument("--calls", type=int, default=1,
                   help="inferences captured in this trace, for per-call normalisation")
    p.add_argument("--json", help="also write the raw record here, for the ledger")
    a = p.parse_args()

    r = analyse(a.trace, a.calls)
    c, s, ll = r["compiled"], r["sync"], r["launch_lead_us"]
    print(f"{r['trace']}  ({r['calls']} call(s))")
    gt = r["ground_truth"]
    if gt.get("profiler_inflation") and not gt["wall_metrics_trustworthy"]:
        print(f"\n  {'*'*72}\n  *** PROFILER INFLATION {gt['profiler_inflation']:.1f}x: "
              f"steady call is {gt['steady_ms_per_call']:.1f} ms but the profiled window is "
              f"{r['span_ms']/r['calls']:.1f} ms/call.\n"
              f"  *** Kernel DURATIONS and device-time buckets below are VALID.\n"
              f"  *** Idle, GPU-busy %, launch lead and sync starvation are ARTIFACTS of the\n"
              f"  *** instrument on this trace -- do not act on them. Re-capture with\n"
              f"  *** record_shapes=False, with_modules=False.\n  {'*'*72}\n")
    print(f"  device events   {r['device_events']:>10,}  ({r['device_events_per_call']:,.0f}/call)")
    print(f"  GPU busy        {r['busy_pct']:>9.1f}%  ({r['busy_ms']:.1f} of {r['span_ms']:.1f} ms)")
    if r["waiting_ms"] > 0.05:
        print(f"  of which WAITING{r['waiting_pct_of_busy']:>9.1f}%  ({r['waiting_ms']:.1f} ms "
              f"spin-waiting, not computing) -> productive {r['productive_pct']:.1f}%")
        for w in r["waiting_kernels"][:3]:
            print(f"                    {w['count']:>6} x {w['kernel'][:52]:<52} {w['ms']:.1f} ms")
    print(f"  idle            {r['idle_ms_per_call']:>9.1f} ms/call over "
          f"{r['gap']['count']:,} gaps (median {r['gap']['median_us']:.2f} us, "
          f"p99 {r['gap']['p99_us']:.1f} us)")
    if ll["median"] is not None:
        print(f"  launch lead     {ll['median']/1e3:>9.2f} ms median, "
              f"p01 {ll['p01']:.1f} us, {ll['frac_under_5us']*100:.1f}% under 5 us "
              f"({ll['coverage']*100:.0f}% of kernels matched)")
    conc, fb, lp = r["concurrency"], r["fallbacks"], r["launch_pressure"]
    if conc["factor"] is not None:
        print(f"  concurrency     {conc['factor']:>9.3f}x  ({conc['streams_live']} stream(s) live, "
              f"{conc['kernel_ms_sum_per_call']:.1f} ms of kernels in "
              f"{conc['kernel_ms_union_per_call']:.1f} ms of wall)")
    if lp["us_per_gap"] is not None:
        print(f"  per-gap         {lp['us_per_gap']:>9.2f} us  ({lp['idle_ms_per_call']:.1f} ms idle "
              f"over {lp['kernels_per_call']:,.0f} kernels, from the {lp['basis']})")
    if fb["ms_per_call"] > 0:
        print(f"  fallbacks       {fb['ms_per_call']:>9.2f} ms/call inside compiled regions "
              f"({fb['pct_of_in_region_device_time']:.0f}% of in-region device time)")
        for k in fb["kernels"][:3]:
            print(f"                    {k['count_per_call']:>6,.0f} x {k['kernel'][:52]:<52} "
                  f"{k['ms_per_call']:.2f} ms")
    print(f"  sync points     {s['implicit_count'] + s['blocking_count']:>10}  "
          f"({s['implicit_count']} implicit reads, {s['blocking_count']} blocking calls, "
          f"{s['d2h_copies']} DtoH copies)")
    for k, v in list(s["implicit"].items())[:4]:
        print(f"                    {v['count']:>6} x {k:<34} idle {v['gpu_idle_ms']:.1f} ms")
    for k, v in list(s["explicit"].items())[:4]:
        print(f"                    {v['count']:>6} x {k:<34} idle {v['gpu_idle_ms']:.1f} ms")
    d = r["queue_drain"]
    print(f"  queue drain     {d['drained']:>10} of {d['checked']} syncs followed by idle "
          f"(> {d['threshold_us']:.1f} us), {d['starve_ms_per_call']:.2f} ms/call union; "
          f"worst {d['worst_starve_us']/1e3:.2f} ms")
    for x in d["by_op"][:4]:
        print(f"                    {x['count']:>6} x {x['op']:<34} "
              f"starve {x['starve_ms']:>7.2f} ms, worst {x['worst_us']/1e3:.2f} ms")
    if d["by_op"]:
        print("                    Per-op totals may overlap; sync attribution is a hypothesis.")
    print(f"  compiled        {c['device_coverage_pct'] or 0:>9.1f}% of DEVICE time, "
          f"{c['entries_per_call']:.0f} region entries/call, "
          f"{c['distinct_regions']} distinct graphs")
    print(f"                    {c['device_ms_outside']:.1f} ms of kernels launched outside "
          f"({c['kernels_outside_per_call']:,.0f}/call):")
    for u in c["top_uncompiled_kernels"][:3]:
        print(f"                    {u['ms']:>8.1f} ms  {u['count']:>7,} x  {u['kernel'][:62]}")
    print(f"                    [aten-CPU view: {c['coverage_pct'] or 0:.1f}% -- biased LOW, "
          f"see note] eager {c['cpu_ms_eager']:.1f} ms costing "
          f"{c['eager_gpu_idle_ms']:.1f} ms idle")
    for e in r["top_eager_ops"][:5]:
        print(f"                    {e['cpu_ms']:>7.1f} ms cpu / {e['gpu_idle_ms']:>6.1f} ms idle "
              f"{e['count']:>6} x {e['op']}")
    print("  device time by bucket:")
    tot = sum(r["device_ms_by_bucket"].values()) or 1
    for k, v in r["device_ms_by_bucket"].items():
        print(f"                    {v:>9.1f} ms  {v/tot*100:>5.1f}%  {k}")
    print("\n  VERDICT")
    for line in verdict(r):
        print(f"    - {line}")

    if a.json:
        json.dump(r, open(a.json, "w"), indent=1)
        print(f"\n  wrote {a.json}")


if __name__ == "__main__":
    main()
