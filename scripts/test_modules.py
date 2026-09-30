"""Regression tests for the trace-analysis modules, on a synthetic trace.

These modules turn a profile into verdicts people act on. A synthetic trace with
known properties checks those verdicts against a controlled ground truth.

Portable on purpose. The recorded traces are hundreds of megabytes and are not
in the repository, so a test that needed one would not run.

Usage:  python scripts/test_modules.py
"""
import json
import contextlib
import io
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from wamjet import attribute, equivalence, fusion, instrument, paired, pipeline, preflight  # noqa: E402
from wamjet.instrument import steady  # noqa: E402

fails = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}{'  -- ' + detail if detail and not cond else ''}")
    if not cond:
        fails.append(name)


# Two calls. Each repeats an eager elementwise -> reduce pair a compiler could
# fuse, then runs a cuDNN kernel nothing fuses into, then a device-to-device copy.
# The pair repeats because `analyse` ignores any pair seen fewer than four times,
# which is a noise filter rather than a threshold worth testing against.
ELEM, RED, VENDOR, COPY = 100, 60, 200, 5     # microseconds
REPS = 3                                       # pair occurrences per call
events, t, corr = [], 1000, 0
for call in range(2):
    seq = [("at::native::elementwise_kernel", ELEM),
           ("at::native::reduce_kernel", RED)] * REPS
    seq.append(("sm90_xmma_cudnn_conv_fprop", VENDOR))
    for nm, dur in seq:
        corr += 1
        events.append({"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel",
                       "ts": t - 5, "dur": 2, "args": {"correlation": corr}})
        events.append({"ph": "X", "cat": "kernel", "name": nm, "ts": t, "dur": dur,
                       "args": {"stream": 7, "correlation": corr}})
        t += dur + 10
    events.append({"ph": "X", "cat": "gpu_memcpy", "name": "Memcpy DtoD", "ts": t,
                   "dur": COPY, "args": {"bytes": 1_000_000, "stream": 7}})
    t += 100

KERNEL_MS_PER_CALL = ((ELEM + RED) * REPS + VENDOR) / 1000.0

with tempfile.TemporaryDirectory() as td:
    path = f"{td}/synthetic_trace.json"
    json.dump({"traceEvents": events}, open(path, "w"))

    # -- vendor classification: the H12 regression ----------------------------
    check("at::native is NOT a vendor kernel (H12)",
          not fusion.is_vendor("at::native::elementwise_kernel"))
    check("a triton kernel is NOT a vendor kernel",
          not fusion.is_vendor("triton_poi_fused_add_0"))
    check("cuDNN IS a vendor kernel", fusion.is_vendor("sm90_xmma_cudnn_conv_fprop"))
    check("cuBLAS IS a vendor kernel", fusion.is_vendor("cublasLt_sgemm_128x64"))
    check("cutlass IS a vendor kernel", fusion.is_vendor("cutlass_80_tensorop_gemm"))

    # -- device time is the sum of kernel durations, divided by calls ----------
    # The two modules define "device time" differently ON PURPOSE, and the
    # difference is exactly the copies: `attribute` counts kernels plus memcpy
    # and memset, because it answers "where did the device's time go"; `fusion`
    # counts kernels only, because a copy is not something a compiler fuses.
    # Pinned here so the divergence stays deliberate -- two tools reporting one
    # metric name with two denominators is how H10 happened.
    a = attribute.attribute(path, calls=2)
    r = fusion.analyse(path, calls=2)
    check("fusion counts kernels only",
          abs(r["device_ms_per_call"] - KERNEL_MS_PER_CALL) < 1e-9,
          f"{r['device_ms_per_call']} vs {KERNEL_MS_PER_CALL}")
    check("attribute counts kernels plus copies",
          abs(a["device_ms_per_call"] - (KERNEL_MS_PER_CALL + COPY / 1000.0)) < 1e-9,
          f"{a['device_ms_per_call']} vs {KERNEL_MS_PER_CALL + COPY/1000.0}")
    check("memcpy bytes are counted per call",
          r["memcpy_bytes_per_call"] == 1_000_000, str(r["memcpy_bytes_per_call"]))

    # -- the fusable pair is found and explicitly ranked as an estimate -------
    cands = str(r["fusion_candidates"]).lower()
    check("the eager pair is offered as a fusion candidate",
          "elementwise" in cands and "reduce" in cands, cands[:120])
    check("no candidate proposes fusing into the vendor kernel",
          "cudnn" not in cands, cands[:120])
    pair = next(c for c in r["fusion_candidates"]
                if "elementwise" in c["producer"] and "reduce" in c["consumer"])
    check("fusion reports a ranking estimate without claiming a saving bound",
          "ranking_estimate_ms_per_call" in pair and "ceiling_ms_per_call" not in pair,
          str(pair))
    with contextlib.redirect_stdout(io.StringIO()) as output:
        old_argv = sys.argv
        sys.argv = ["fusion", path, "--calls", "2"]
        try:
            fusion.main()
        finally:
            sys.argv = old_argv
    check("fusion CLI explains that ranking estimates are not speedup bounds",
          "it is not a\n  speedup bound" in output.getvalue()
          and "you can never save more" not in output.getvalue(), output.getvalue())
    occupancy_counts = {x["kernel"]: x["count_per_call"] for x in r["occupancy"]}
    check("occupancy retains per-kernel counts",
          occupancy_counts["at::native::elementwise_kernel"] == REPS
          and occupancy_counts["sm90_xmma_cudnn_conv_fprop"] == 1, str(occupancy_counts))


# ---------------------------------------------------------------------------
# fusion across streams. CUDA-graph capture splits work over a capture stream
# and a side stream. Chaining adjacency on only the busiest one drops the other
# stream's kernels while the occurrence counts still include them, so `share`
# compares mismatched populations and every candidate fails "always adjacent" --
# no candidates despite fusable work in the trace.
ev3, t3, c3 = [], 1000, 0
for call in range(2):
    for _ in range(6):                       # filler makes stream 7 the busiest
        c3 += 1
        ev3.append({"ph": "X", "cat": "kernel", "name": "triton_poi_filler",
                    "ts": t3, "dur": 30, "args": {"stream": 7, "correlation": c3}})
        t3 += 40
    for _ in range(4):                       # the fusable pair lives on stream 13
        for nm, dur in (("at::native::producer_kernel", 90),
                        ("at::native::consumer_kernel", 50)):
            c3 += 1
            ev3.append({"ph": "X", "cat": "kernel", "name": nm, "ts": t3, "dur": dur,
                        "args": {"stream": 13, "correlation": c3}})
            t3 += dur + 5

with tempfile.TemporaryDirectory() as td:
    path3 = f"{td}/two_stream_trace.json"
    json.dump({"traceEvents": ev3}, open(path3, "w"))
    r3 = fusion.analyse(path3, calls=2)
    cands3 = str(r3["fusion_candidates"])
    check("a pair on a non-busiest stream is still found",
          "producer_kernel" in cands3 and "consumer_kernel" in cands3, cands3[:130])
    # Assert on the producer->consumer candidate itself. The reverse adjacency
    # is also a legitimate candidate on an alternating chain, so the total is
    # the sum of both and is not the quantity under test here.
    pc = [c for c in r3["fusion_candidates"]
          if c["producer"].endswith("producer_kernel")
          and c["consumer"].endswith("consumer_kernel")]
    check("the pair is offered exactly once", len(pc) == 1, str(r3["fusion_candidates"])[:130])
    if pc:
        check("its estimate retains occurrences from the secondary stream",
              abs(pc[0]["ranking_estimate_ms_per_call"] - (50 * 8 / 2) / 1000.0) < 1e-9,
              f"{pc[0]['ranking_estimate_ms_per_call']} vs {(50*8/2)/1000.0}")

# ---------------------------------------------------------------------------
# pipeline: a compiled region containing a compiler-generated kernel, a vendor
# kernel the compiler deliberately calls, and an eager kernel it declined. Only
# the last is a fallback.
FALLBACK_US = 200        # above the 0.05 ms/call noise floor the verdict applies
ev2, t2, c2 = [], 1000, 0
region_start = t2 - 20
for call in range(2):
    for nm, dur in (("triton_poi_fused_mul_0", 50),
                    ("cublasLt_sgemm_128x64", 70),
                    ("at::native::elementwise_kernel<complex>", FALLBACK_US)):
        c2 += 1
        ev2.append({"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel",
                    "ts": t2 - 5, "dur": 2, "args": {"correlation": c2}})
        ev2.append({"ph": "X", "cat": "kernel", "name": nm, "ts": t2, "dur": dur,
                    "args": {"stream": 7, "correlation": c2}})
        t2 += dur + 10
# One region span covering every launch above.
ev2.append({"ph": "X", "cat": "cpu_op", "name": "Torch-Compiled Region: 0/0",
            "ts": region_start, "dur": t2 - region_start, "tid": 1})

with tempfile.TemporaryDirectory() as td:
    path2 = f"{td}/compiled_trace.json"
    json.dump({"traceEvents": ev2}, open(path2, "w"))
    pr = pipeline.analyse(path2, calls=2)

    fb = pr["fallbacks"]
    check("the declined eager kernel is reported as a fallback",
          abs(fb["ms_per_call"] - FALLBACK_US / 1000.0) < 1e-9,
          f"{fb['ms_per_call']} vs {FALLBACK_US/1000.0}")
    names = str(fb["kernels"])
    check("a triton kernel is not called a fallback", "triton" not in names, names[:100])
    check("a vendor kernel the compiler calls is not a fallback",
          "cublas" not in names.lower(), names[:100])

    conc = pr["concurrency"]
    check("concurrency is 1.0 when kernels never overlap",
          abs(conc["factor"] - 1.0) < 1e-6, str(conc["factor"]))
    check("one live stream is reported as one", conc["streams_live"] == 1,
          str(conc["streams_live"]))

    # No G0 record beside a synthetic trace, so launch pressure must decline to
    # guess rather than derive a gap from the profiled window (H11).
    lp = pr["launch_pressure"]
    check("launch pressure refuses to compute without a trusted wall",
          lp["us_per_gap"] is None and "unavailable" in lp["basis"], str(lp))

    v = " ".join(pipeline.verdict(pr))
    check("a trace without G0 cannot make scheduling claims",
          "MEASUREMENT UNVERIFIED" in v, v[:150])
    check("the fallback reaches the verdict", "COMPILER FALLBACK" in v, v[:150])
    check("no serialization verdict on a single stream", "SERIALIZED" not in v, v[:150])

# A sync returning while another stream is active must not invent starvation.
# Nested CPU/runtime events can describe one gap, which the total counts once.
def device_event(ts, dur, stream=1):
    return {"cat": "kernel", "name": "triton_poi_add", "ts": ts, "dur": dur,
            "args": {"stream": stream}}


def sync_event(cat, name, ts, dur):
    return {"cat": cat, "name": name, "ts": ts, "dur": dur}


with tempfile.TemporaryDirectory() as td:
    path = f"{td}/sync_idle.json"
    cases = [
        ("continuous work", [device_event(0, 1000), device_event(500, 10, 2),
                              device_event(1000, 1000),
                              sync_event("cuda_runtime", "cudaStreamSynchronize", 50, 50)],
         0.0),
        ("partly busy after sync", [device_event(0, 1000), device_event(1100, 100),
                                    device_event(1201, 100), device_event(1302, 100),
                                    sync_event("cuda_runtime", "cudaStreamSynchronize", 50, 50)],
         0.1),
        ("nested syncs", [device_event(0, 100), device_event(1000, 100),
                           device_event(1101, 100), device_event(1202, 100),
                           sync_event("cpu_op", "aten::item", 80, 40),
                           sync_event("cpu_op", "aten::_local_scalar_dense", 85, 25),
                           sync_event("cuda_runtime", "cudaStreamSynchronize", 90, 10)],
         0.9),
    ]
    for name, trace_events, expected_idle_ms in cases:
        pathlib.Path(path).write_text(json.dumps({"traceEvents": trace_events}))
        pathlib.Path(path + ".meta.json").write_text(json.dumps(
            {"trusted": True, "calls": 1, "median_ms": 2.0}))
        r = pipeline.analyse(path)
        actual = r["queue_drain"]["starve_ms_per_call"]
        check(f"queue drain counts actual idle once: {name}",
              abs(actual - expected_idle_ms) < 1e-9, str(r["queue_drain"]))
        check(f"queue drain is bounded by observed total idle: {name}",
              actual <= r["idle_ms_per_call"] + 1e-9, str(r["queue_drain"]))
    check("sync adjacency is presented as a candidate cause",
          any("adjacency alone does not establish causation" in v
              for v in pipeline.verdict(r)))

# Stable measurements must actually contain a stable suffix, even on short runs.
check("a stable short measurement is trusted", steady([10.0] * 12)["trusted"])
unstable = steady([10.0, 100.0, 10.0] * 4)
check("an unstable short measurement is not trusted", not unstable["trusted"], str(unstable))

for base_keys, cand_keys in ((range(50), range(10)), (range(10), range(50))):
    try:
        paired.pair({str(k): True for k in base_keys}, {str(k): True for k in cand_keys})
    except ValueError:
        rejected = True
    else:
        rejected = False
    check("different episode sets block the comparison", rejected)
check("matching episode sets remain comparable",
      paired.pair({"a": True, "b": False}, {"a": False, "b": True})["n_paired"] == 2)

original_fingerprint = preflight.fingerprint
preflight.fingerprint = lambda: {"gpu": "test", "sm": "90", "triton": "test"}
try:
    check("measurement and comparability use one environment fingerprint",
          instrument.node_fingerprint() == preflight.fingerprint())
finally:
    preflight.fingerprint = original_fingerprint

with tempfile.TemporaryDirectory() as td:
    path = f"{td}/overlap.json"
    overlap = [
        {"cat": "kernel", "name": "triton_poi_add", "ts": 1000, "dur": 1000,
         "args": {"stream": 1}},
        {"cat": "kernel", "name": "triton_poi_mul", "ts": 1000, "dur": 1000,
         "args": {"stream": 2}},
    ]
    pathlib.Path(path).write_text(json.dumps({"traceEvents": overlap}))
    meta = {"median_ms": 1.1, "calls": 1, "trusted": True}
    pathlib.Path(path + ".meta.json").write_text(json.dumps(meta))
    r = pipeline.analyse(path)
    lp = r["launch_pressure"]
    check("overlapping streams never produce negative idle",
          lp["idle_ms_per_call"] is None or lp["idle_ms_per_call"] >= 0, str(lp))
    check("kernel count alone cannot establish a launch-count bound",
          "LAUNCH-COUNT BOUND" not in " ".join(pipeline.verdict(r)))

    # An inflated trace can count work, but cannot recover elapsed busy time.
    overlap[1]["ts"] = 10000
    pathlib.Path(path).write_text(json.dumps({"traceEvents": overlap}))
    r = pipeline.analyse(path)
    check("inflation leaves elapsed idle unknown",
          r["launch_pressure"]["idle_ms_per_call"] is None, str(r["launch_pressure"]))
    check("inflation verdict does not infer busy percent from kernel sums",
          "genuinely idle" not in " ".join(pipeline.verdict(r)))

    meta["trusted"] = False
    pathlib.Path(path + ".meta.json").write_text(json.dumps(meta))
    r = pipeline.analyse(path)
    check("an untrusted G0 record cannot supply a trusted wall",
          r["ground_truth"].get("steady_ms_per_call") is None, str(r["ground_truth"]))

    other = f"{td}/reordered.json"
    changed = [dict(e, ts=20000 - e["ts"], dur=2 * e["dur"]) for e in overlap]
    pathlib.Path(other).write_text(json.dumps({"traceEvents": changed}))
    diff = equivalence.graph_diff(path, other)
    check("matching names are explicitly a histogram comparison",
          diff.get("same_kernel_histogram") is True and "identical" not in diff, str(diff))
    with contextlib.redirect_stdout(io.StringIO()) as output:
        old_argv = sys.argv
        sys.argv = ["equivalence", path, other]
        try:
            equivalence.main()
        finally:
            sys.argv = old_argv
    check("matching names do not dismiss measured speedups as noise",
          "IDENTICAL GRAPHS" not in output.getvalue() and "is noise" not in output.getvalue())

# Attribution must preserve nested ownership, thread isolation and shared graph
# launches when device events arrive in a different order from CPU launches.
with tempfile.TemporaryDirectory() as td:
    path = f"{td}/ownership.json"
    spans = [
        {"cat": "cpu_op", "name": "nn.Module: outer", "ts": 0, "dur": 100, "tid": 1},
        {"cat": "cpu_op", "name": "nn.Module: inner", "ts": 0, "dur": 20, "tid": 1},
        {"cat": "cpu_op", "name": "aten::add", "ts": 2, "dur": 8, "tid": 1,
         "args": {"Input Dims": [[4]]}},
        {"cat": "cpu_op", "name": "aten::mul", "ts": 30, "dur": 10, "tid": 1},
        {"cat": "cpu_op", "name": "aten::sub", "ts": 2, "dur": 8, "tid": 2},
    ]
    launches = [
        {"cat": "cuda_runtime", "ts": 5, "tid": 1, "args": {"correlation": 1}},
        {"cat": "cuda_driver", "ts": 35, "tid": 1, "args": {"correlation": 2}},
        {"cat": "cuda_runtime", "ts": 5, "tid": 2, "args": {"correlation": 3}},
    ]
    kernels = [{"cat": "kernel", "name": "kernel", "ts": 200 + i, "dur": dur,
                "args": {"correlation": corr}}
               for i, (corr, dur) in enumerate(((2, 20), (1, 10), (3, 30), (1, 5)))]
    pathlib.Path(path).write_text(json.dumps({"traceEvents": kernels + launches + spans}))
    r = attribute.attribute(path)
    owners = {v["name"]: v["ms"] for v in r["by_module"]}
    check("nested modules with equal start times use the deepest owner",
          owners.get("nn.Module: inner") == 0.015, str(owners))
    check("expired spans and other threads do not take ownership",
          owners.get("nn.Module: outer") == 0.02 and owners.get("aten::sub") == 0.03, str(owners))
    check("shared launch attribution includes every kernel",
          sum(row["kernels"] for row in r["by_op"]) == 4 and r["unattributed_kernels"] == 0)

print()
if fails:
    print(f"  {len(fails)} failure(s): {', '.join(fails)}")
    sys.exit(1)
print("  all module regression checks passed")
