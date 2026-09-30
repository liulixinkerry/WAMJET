"""Gate G1, bandwidth half -- where would fusion cut memory traffic?

Once the queue is full and the pipeline is clean, what is left is memory
traffic. A model at >90% occupancy can still spend a third of its device time
in kernels that are moving bytes, not doing arithmetic -- and the fix is not a
faster kernel, it is *one fewer pass over the same tensor*.

Every fusion removes a full round trip: the producer's write of an intermediate
plus the consumer's read of it. This memory traffic can dominate arithmetic.
Use the current workload's trace to select candidates, then measure their effect
on endpoint latency.

This finds three things, all derivable from the trace with no extra input:

  1. FUSION CANDIDATES. Ordered kernel pairs that essentially always run
     back-to-back. If A is almost always followed by B, the intermediate
     between them is written and immediately re-read, and a fused kernel
     would keep it in registers.

  2. TWO-PASS PATTERNS. Inductor names its kernels after the ops it merged, so
     the names say what is being recomputed. A reduction kernel followed by a
     pointwise kernel sharing op tokens is the classic compute-a-statistic-
     then-apply-it shape -- `amax` then `div/clamp` is dynamic quantization,
     and static scaling deletes the first pass outright.

  3. UNDERFILLED LAUNCHES. Occupancy and grid size, weighted by device time.
     A memory-bound kernel that cannot fill the machine is wasting bandwidth it
     never asked for.

Bandwidth in absolute GB/s needs tensor sizes, which a trace only carries when
captured with `record_shapes=True` (opt in with `wamjet.capture`). Where they
are missing this reports structure rather than inventing numbers.

Usage:  python -m wamjet.fusion TRACE.json [--calls N] [--top N]
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict

# Inductor encodes the merged op set in the kernel name; pull the tokens out.
_NOISE = {"fused", "triton", "poi", "red", "per", "view", "t", "to", "copy",
          "unsqueeze", "squeeze", "split", "select", "slice", "arange", "stack"}
_TOK = re.compile(r"[a-z][a-z0-9]*")
# tokens that mean "this kernel computed a statistic over the tensor"
_REDUCE_TOK = {"amax", "amin", "sum", "mean", "var", "norm", "max", "min", "prod", "rsqrt"}
# vendor kernels are closed: you cannot fuse anything into cuDNN or cuBLAS.
_VENDOR = ("cudnn", "nvjet", "cutlass", "cublas", "sm90_", "sm100_", "xmma", "gemvx")


def is_vendor(name: str) -> bool:
    """Closed kernels you cannot fuse into: cuDNN, cuBLAS, cutlass, nvjet.

    NOT `at::native::*` -- those are PyTorch's own eager kernels, and they are
    exactly what a compiler fuses. An earlier version excluded anything
    containing "::" without "triton", which swallowed every eager kernel and
    reported a 0.0 ms fusable ceiling on a trace that was nothing but fusable
    work. The `::` test was standing in for "vendor" and meant "C++ symbol".
    """
    n = name.lower()
    return any(v in n for v in _VENDOR)


def ops_in(name: str) -> set[str]:
    if "triton_" not in name:
        return set()
    body = name.rsplit("fused_", 1)[-1]
    return {t for t in _TOK.findall(body) if t not in _NOISE and len(t) > 2}


def analyse(path: str, calls: int = 1, min_share: float = 0.8) -> dict:
    ev = json.load(open(path))["traceEvents"]
    dev = [e for e in ev if e.get("cat") == "kernel" and e.get("dur", 0) > 0]
    if not dev:
        raise SystemExit(f"{path}: no kernels")

    # order on the busiest stream -- that is the dependency chain
    per_stream = defaultdict(list)
    for e in dev:
        per_stream[e.get("args", {}).get("stream", 0)].append(e)

    total = sum(e["dur"] for e in dev)
    kt, kn, first_args = defaultdict(float), Counter(), {}
    for e in dev:
        kt[e["name"]] += e["dur"]
        kn[e["name"]] += 1
        first_args.setdefault(e["name"], e.get("args", {}))

    # Chain adjacency within EVERY stream, not only the busiest one. CUDA-graph
    # capture splits work across a capture stream and a side stream, so analysing
    # one stream drops the kernels on the other -- while `kn` below still counts
    # them. The `share` test then compares a numerator that saw one stream against
    # a denominator that counted both, every candidate fails "always adjacent",
    # and the tool reports a 0.0 fusable ceiling for a build carrying the same
    # fusable material as its uncaptured twin. Adjacency is only a dependency
    # signal within a stream, so pairing across streams would be wrong; pairing
    # within each of them is the fix.
    pairs = Counter()
    pair_ms = defaultdict(float)
    for kernels in per_stream.values():
        seq = sorted(kernels, key=lambda e: e["ts"])
        for a, b in zip(seq, seq[1:]):
            pairs[(a["name"], b["name"])] += 1
            pair_ms[(a["name"], b["name"])] += a["dur"] + b["dur"]

    cands = []
    for (a, b), n in pairs.items():
        share = min(n / kn[a], n / kn[b])
        if share < min_share or n < 4:
            continue
        if is_vendor(a) or is_vendor(b):
            continue                      # closed kernels: nothing to fuse into
        oa, ob = ops_in(a), ops_in(b)
        # A reduction feeding a pointwise kernel is a statistic computed and then
        # applied -- the canonical two-pass shape, and the one worth deleting.
        two_pass = bool(oa & _REDUCE_TOK) and "_poi_" in b
        # A ranking estimate, not a speedup bound: fusion can reduce traffic in
        # both kernels, or introduce overhead that erases the expected benefit.
        estimate = min(kt[a] / kn[a], kt[b] / kn[b]) * n / 1e3 / calls
        cands.append({
            "producer": a, "consumer": b, "count_per_call": n // calls,
            "pair_ms_per_call": pair_ms[(a, b)] / 1e3 / calls,
            "ranking_estimate_ms_per_call": estimate,
            "pct": estimate / total * 1e3 * calls * 100,
            "always_adjacent": round(share, 3),
            "shared_ops": sorted(oa & ob),
            "two_pass": two_pass,
        })
    cands.sort(key=lambda x: -x["ranking_estimate_ms_per_call"])

    occ = []
    for name, ms in sorted(kt.items(), key=lambda x: -x[1])[:40]:
        a = first_args[name]
        g = a.get("grid") or [0]
        o = a.get("est. achieved occupancy %")
        # the profiler reports 0 for kernels it cannot measure (every vendor
        # kernel). A running kernel cannot truly have 0% occupancy, so 0 means
        # "not reported" -- treating it as a finding invents a bottleneck.
        occ.append({"kernel": name, "ms_per_call": ms / 1e3 / calls,
                    "count_per_call": kn[name] // calls,
                    "occupancy_pct": o if o else None,
                    "grid": g[0] if isinstance(g, list) else g,
                    "block": (a.get("block") or [0])[0]})

    # all copies, not just host<->device: static-buffer marshalling shows up as DtoD
    mc = [e for e in ev if e.get("cat") == "gpu_memcpy" and e.get("args", {}).get("bytes")]
    by_kind = defaultdict(float)
    for e in mc:
        kind = e["name"].split("(")[0].replace("Memcpy", "").strip() or "?"
        by_kind[kind] += e["args"]["bytes"]
    return {
        "trace": path, "calls": calls,
        "device_ms_per_call": total / 1e3 / calls,
        "distinct_kernels": len(kt),
        "fusion_candidates": cands,
        "occupancy": occ,
        "memcpy_bytes_per_call": sum(e["args"]["bytes"] for e in mc) / calls,
        "memcpy_count_per_call": len(mc) // calls,
        "memcpy_by_kind_mb_per_call": {k: v / 1e6 / calls for k, v in
                                       sorted(by_kind.items(), key=lambda x: -x[1])},
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trace")
    p.add_argument("--calls", type=int, default=1)
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--json")
    a = p.parse_args()
    r = analyse(a.trace, a.calls)

    print(f"{r['trace']}  ({r['calls']} call(s))")
    print(f"  device time     {r['device_ms_per_call']:.1f} ms/call across "
          f"{r['distinct_kernels']} distinct kernels")
    print(f"  memcpy traffic  {r['memcpy_bytes_per_call']/1e6:.1f} MB/call in "
          f"{r['memcpy_count_per_call']} copies  "
          + ", ".join(f"{k} {v:.0f} MB" for k, v in
                      list(r["memcpy_by_kind_mb_per_call"].items())[:3]))

    print(f"\n  FUSION CANDIDATES -- fusable kernel pairs that always run back to back.")
    print("  'estimate' ranks pairs by the smaller kernel's runtime; it is not a")
    print("  speedup bound or a measured saving. Confirm shared data and benchmark")
    print("  the fused endpoint. Closed vendor kernels are excluded.\n")
    print(f"    {'estimate':>8} {'pair ms':>8} {'n':>5}  producer -> consumer")
    for c in r["fusion_candidates"][:a.top]:
        flag = "   <- TWO-PASS" if c["two_pass"] else ""
        print(f"    {c['ranking_estimate_ms_per_call']:>8.2f} {c['pair_ms_per_call']:>8.2f} "
              f"{c['count_per_call']:>5}  {c['producer'][:66]}{flag}")
        print(f"    {'':>23}  -> {c['consumer'][:66]}")
        if c["shared_ops"]:
            print(f"    {'':>23}     both touch: {', '.join(c['shared_ops'][:8])}")
    if r["fusion_candidates"]:
        print("\n    Pair estimates may share kernels; do not add them.")
    if any(c["two_pass"] for c in r["fusion_candidates"]):
        print(f"    A reduction feeding a pointwise kernel over the same tensor is a "
              f"statistic\n    computed and then applied. If that statistic can be held "
              f"static or precomputed,\n    the first pass disappears outright rather than "
              f"merely being fused.")

    known = [o for o in r["occupancy"] if o["occupancy_pct"] is not None]
    bad = [o for o in known if o["occupancy_pct"] < 30][:6]
    print(f"\n  UNDERFILLED LAUNCHES -- big kernels that cannot fill the machine")
    print(f"  ({len(known)} of {len(r['occupancy'])} top kernels report occupancy; "
          f"vendor kernels do not):")
    if bad:
        for o in bad:
            print(f"    {o['ms_per_call']:>8.2f} ms  occ {o['occupancy_pct']:>5.1f}%  "
                  f"grid {o['grid']:>7}  {o['kernel'][:62]}")
    else:
        print("    none below 30% among those that report -- launches are well filled.")

    if a.json:
        json.dump(r, open(a.json, "w"), indent=1)
        print(f"\n  wrote {a.json}")


if __name__ == "__main__":
    main()
