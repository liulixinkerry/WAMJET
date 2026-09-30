"""Environment metadata and optional setup diagnostics.

`fingerprint` caches one snapshot per process without importing optional libraries.
For setup failures, probe selected modules/symbols before loading checkpoints:
    python -m wamjet.preflight --import-target package.module:RequiredClass

The full CLI additionally probes capabilities and linker hazards when needed.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
from importlib import metadata
from functools import lru_cache
import json
import os
import platform
import subprocess
import sys
import time


def probe_imports(targets, timeout=30):
    """Probe selected module[:symbol] imports using this interpreter, without loading weights.

    Each import gets a fresh process and a time limit. Retain its traceback rather
    than interpreting every import failure as a missing package. Import success alone
    does not establish GPU readiness or that the model can run.
    """
    script = (
        "import importlib, sys\n"
        "module, separator, symbol = sys.argv[1].partition(':')\n"
        "value = importlib.import_module(module)\n"
        "if separator:\n"
        "    for name in symbol.split('.'):\n"
        "        value = getattr(value, name)\n"
        "print('WAMJET_IMPORT_OK')\n"
    )
    rows = []
    for target in targets:
        started = time.perf_counter()
        try:
            process = subprocess.run([sys.executable, "-c", script, target],
                                     capture_output=True, text=True, timeout=timeout)
            ok = process.returncode == 0 and "WAMJET_IMPORT_OK" in process.stdout.splitlines()
            row = {"target": target, "ok": ok, "status": "imported" if ok else "failed",
                   "returncode": process.returncode,
                   "stdout": process.stdout, "stderr": process.stderr}
        except subprocess.TimeoutExpired as exc:
            def text(value):
                return value.decode(errors="replace") if isinstance(value, bytes) else value or ""
            row = {"target": target, "ok": False, "status": "timeout",
                   "stdout": text(exc.stdout), "stderr": text(exc.stderr)}
        rows.append({**row, "seconds": time.perf_counter() - started})
    return {"executable": sys.executable, "python": platform.python_version(),
            "host": platform.node(), "imports_ok": bool(rows) and all(r["ok"] for r in rows),
            "imports": rows}

# (label, module, symbol substring, minimum sm as (major, minor), note)
PRECISION_PATHS = [
    ("fp8-torchao", "torchao.quantization", "Float8DynamicActivationFloat8Weight", (8, 9),
     "Speedup depends on tensor shapes, backend and GPU. Measure quantization overhead"
     " and endpoint latency on the current workload."),
    ("nvfp4-torchao", "torchao.prototype.mx_formats", "NVFP4DynamicActivationNVFP4Weight", (10, 0),
     "Blackwell only. NOT in torchao.quantization -- looking there and concluding it is absent"
     " is the exact mistake this gate exists to prevent"),
    ("nvfp4-weightonly", "torchao.prototype.mx_formats", "NVFP4WeightOnlyConfig", (10, 0),
     "weight-only variant; no activation quantization pass"),
    ("mx-torchao", "torchao.prototype.mx_formats", "MXDynamicActivationMXWeight", (10, 0),
     "MXFP8/MXFP4 microscaling"),
    ("int8-torchao", "torchao.quantization", "Int8DynamicActivationInt8Weight", (7, 5), ""),
]


def _sm():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_capability()
    except Exception:
        pass
    return None


@lru_cache(maxsize=1)
def fingerprint() -> dict:
    """One environment snapshot per process, without importing optional libraries.

    Call fingerprint.cache_clear() after deliberately changing the in-process stack.
    """
    fp = {"host": platform.node(), "python": platform.python_version(),
          "executable": sys.executable}
    for m in ("torch", "triton", "torchao", "transformer_engine", "transformers",
              "modelopt", "numpy"):
        try:
            loaded = sys.modules.get(m)
            distribution = "nvidia-modelopt" if m == "modelopt" else m
            fp[m] = getattr(loaded, "__version__", None) or metadata.version(distribution)
        except metadata.PackageNotFoundError:
            fp[m] = None
    try:
        import torch
        fp["cuda"] = torch.version.cuda
        fp["cudnn"] = torch.backends.cudnn.version()
        fp["cuda_available"] = torch.cuda.is_available()
        if fp["cuda_available"]:
            cap = torch.cuda.get_device_capability()
            fp["gpu"] = torch.cuda.get_device_name(0)
            fp["sm"] = f"{cap[0]}{cap[1]}"
    except Exception:
        pass
    try:
        q = ("nvidia-smi --query-gpu=driver_version,memory.total,power.limit"
             " --format=csv,noheader")
        row = subprocess.run(q.split(), capture_output=True, text=True,
                             timeout=15).stdout.strip().splitlines()[0]
        fp["driver"], fp["vram"], fp["power_limit"] = [c.strip() for c in row.split(",")]
    except Exception:
        pass
    for k in ("TORCHINDUCTOR_CACHE_DIR", "CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH"):
        if k in os.environ:
            fp[k] = os.environ[k]
    return fp


def capabilities(deep: bool = False) -> dict:
    """What precision paths exist here, and which can actually run on this GPU.

    Two separate questions. A config that imports is not a config you can run:
    NVFP4 needs sm_100, so on an H100 it is present and useless.
    """
    sm = _sm()
    out = []
    for label, mod, sym, need, note in PRECISION_PATHS:
        rec = {"path": label, "module": mod, "requires_sm": f"{need[0]}{need[1]}",
               "note": note, "available": False, "runnable_here": False, "symbols": []}
        try:
            m = importlib.import_module(mod)
            hits = [n for n in dir(m) if sym in n]
            rec["available"] = bool(hits)
            rec["symbols"] = sorted(hits)[:4]
        except ImportError as e:
            rec["error"] = f"not installed: {type(e).__name__}: {str(e)[:60]}"
        except Exception as e:
            # A probe bug is not a missing feature. Say so loudly rather than
            # letting a broad except turn our own error into "unavailable" --
            # that is exactly how FP4 got recorded as blocked for weeks.
            rec["error"] = f"PROBE ERROR (not a missing feature): {type(e).__name__}: {e}"
            rec["probe_failed"] = True
        rec["runnable_here"] = bool(rec["available"] and sm and sm >= need)
        out.append(rec)

    # modelopt ships its own quantization recipes; enumerate rather than assume.
    # But importing modelopt.torch.quantization drags in Megatron and takes
    # minutes, and a gate that runs first every time has to be fast -- so this
    # one is opt-in. A preflight nobody runs protects nobody.
    mo = {"available": None, "fp4_configs": [], "probed": deep}
    if deep:
        try:
            import modelopt.torch.quantization as mtq
            cfgs = [n for n in dir(mtq) if n.endswith("_CFG")]
            mo = {"available": True, "probed": True, "total_configs": len(cfgs),
                  "fp4_configs": sorted(c for c in cfgs if "FP4" in c)}
        except Exception as e:
            mo = {"available": False, "probed": True,
                  "error": f"{type(e).__name__}: {str(e)[:60]}", "fp4_configs": []}
    else:
        try:
            mo["available"] = importlib.util.find_spec("modelopt") is not None
            if mo["available"]:
                mo["note"] = "installed; re-run with --deep to enumerate its quant configs"
        except Exception:
            pass

    return {"sm": f"{sm[0]}{sm[1]}" if sm else None, "precision_paths": out, "modelopt": mo}


def hazards() -> list[dict]:
    """Environment traps that make a probe lie rather than fail."""
    out = []

    # 1. self-re-exec: silently kills any script fed on stdin
    sp = os.path.join(sys.prefix, "lib",
                      f"python{sys.version_info.major}.{sys.version_info.minor}",
                      "site-packages")
    # Compare resolved paths, not strings: this cluster reaches the same venv via
    # /fast (a symlink to /lustre/fast/fast), so LD_LIBRARY_PATH and sys.prefix
    # spell the identical directory differently and a string compare reports a
    # present dir as missing.
    raw_ld = os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
    ld = {os.path.realpath(p) for p in raw_ld if p}
    missing = [p for p in (os.path.join(sp, "nvidia", d, "lib")
                           for d in ("cudnn", "cu13", "cu12"))
               if os.path.isdir(p) and os.path.realpath(p) not in ld]
    if missing:
        out.append({
            "hazard": "stdin-eating re-exec",
            "detail": f"venv CUDA lib dir(s) absent from LD_LIBRARY_PATH: "
                      f"{', '.join(os.path.basename(os.path.dirname(m)) for m in missing)}"
                      f" -- so code that re-execs to fix it fires on every process start.",
            "impact": "A script fed on stdin (`python - <<'EOF'`) restarts with stdin already "
                      "consumed and exits 0 with NO output -- which reads as a passing test. "
                      "Run probes from a real file, or `python -m`.",
            "fix": "export LD_LIBRARY_PATH=\"" + ":".join(missing) + ":$LD_LIBRARY_PATH\"",
        })

    # 2. a mismatched system CUDA ahead of the venv's own
    prefix_real = os.path.realpath(sys.prefix)
    venv_first = next((i for i, p in enumerate(raw_ld)
                       if prefix_real in os.path.realpath(p)), len(raw_ld))
    for i, p in enumerate(raw_ld):
        if p.startswith("/is/software") or ("cuda" in p and prefix_real not in os.path.realpath(p)):
            out.append({"hazard": "system CUDA on LD_LIBRARY_PATH",
                        "detail": f"{p} (position {i + 1}; first venv dir at "
                                  f"{venv_first + 1 if venv_first < len(raw_ld) else 'none'})",
                        "impact": ("It PRECEDES the venv's own nvidia/* dirs and wins the linker "
                                   "search -- cuDNN will fail with SUBLIBRARY_LOADING_FAILED."
                                   if i < venv_first else
                                   "Harmless here: the venv's own dirs come first, so they win "
                                   "the search. Keep it that way."),
                        "blocking": i < venv_first})
            break

    # 3. verify, do not assume, that TE's Linear is still untraceable
    try:
        import inspect
        from transformer_engine.pytorch.module import linear as te_linear
        src = inspect.getsource(te_linear)
        if "@no_torch_dynamo" in src:
            out.append({"hazard": "TransformerEngine Linear is untraceable",
                        "detail": "@no_torch_dynamo is still on Linear.forward",
                        "impact": "TE FP8 cannot live inside torch.compile(fullgraph=True); "
                                  "measure the graph-break cost and compare endpoint latency "
                                  "with compatible backends on the current workload."})
    except Exception:
        pass
    return out


def smoke(infer, timeout_note: str = "", validate=None, *,
          maximum_tier: str = "B", task_evaluation: dict | None = None) -> dict:
    """Run ONE real inference. A version check and an import are not enough.

    Learned the hard way: after a stack upgrade, every version looked fine and
    all four model modules imported cleanly -- and the first forward pass died
    on `fused_attn_fwd(): incompatible function arguments`, because the vendor
    library had changed a signature in the hot path. Imports resolve names;
    they do not call them. A preflight that stops at imports hands you false
    confidence and costs you a model load plus a compile before you find out.

    Pass the ALREADY-BUILT zero-argument `infer` -- the exact object you are
    about to measure. Build it yourself first; smoke-testing must not cost a
    second multi-GB checkpoint load.

    `validate`, when supplied, receives that first call's output after device
    synchronization. It must return a boolean or a serializable mapping. This
    lets a candidate prove its cold first output before a warmup can hide stale
    buffers or a graph-capture result that was never replayed.

    Tier B/C validations need passing paired task evidence within maximum_tier.

    An earlier version accepted either a builder or a built `infer` and told them
    apart with `callable()`. Both are callable, so it called the `infer`, got its
    `None` return, and then called *that* -- surfacing as a baffling
    "'NoneType' object is not callable" five minutes into a run. A convenience
    that guesses between two callables is a trap; this one takes exactly one
    kind of argument.
    """
    import time, traceback
    if not callable(infer):
        return {"ok": False, "seconds": 0.0,
                "error": f"smoke() needs a zero-argument callable, got {type(infer).__name__}",
                "where": [], "note": "Build the model first and pass the `infer` you are "
                                     "about to measure."}
    t0 = time.perf_counter()
    try:
        output = infer()
        try:
            import torch
            torch.cuda.synchronize()
        except Exception:
            pass
        result = {"ok": True, "seconds": time.perf_counter() - t0}
        if validate is None:
            return result

        validation = validate(output)
        if isinstance(validation, bool):
            valid = validation
            validation = {"ok": validation}
        elif isinstance(validation, dict):
            if any(key in validation for key in ("tier", "blocked", "owes")):
                from .equivalence import is_accepted
                valid = is_accepted(validation, maximum_tier, task_evaluation)
            else:
                valid = validation.get("ok") is True
        else:
            raise TypeError(
                "smoke output validator must return bool or dict, got "
                f"{type(validation).__name__}"
            )
        result["validation"] = validation
        if not valid:
            result["ok"] = False
            result["error"] = "first output validation failed"
        return result
    except BaseException as e:
        return {"ok": False, "seconds": time.perf_counter() - t0,
                "error": f"{type(e).__name__}: {e}",
                "where": traceback.format_exc().strip().splitlines()[-3:],
                "note": "The stack does not RUN. In practice this is one of three things: a "
                        "vendor API signature change in the hot path after an upgrade, a "
                        "dependency missing from this environment, or a config/weights path "
                        "that does not resolve here. All three cost a model load to discover "
                        "if you skip this check. " + timeout_note}


def check_cache(path: str) -> dict:
    """Is a saved compile cache still valid for this stack?"""
    side = path + ".env.json"
    now = fingerprint()
    keys = ("torch", "triton", "torchao", "cuda", "sm", "gpu")
    if not os.path.exists(side):
        return {"cache": path, "stamped": False,
                "verdict": "UNSTAMPED -- cannot tell which stack built it. Stamp it next time."}
    was = json.load(open(side))
    diff = {k: (was.get(k), now.get(k)) for k in keys if was.get(k) != now.get(k)}
    return {"cache": path, "stamped": True, "diff": diff, "valid": not diff,
            "verdict": ("valid for this stack" if not diff else
                        f"STALE -- {', '.join(diff)} changed. It will miss entirely; "
                        f"expect a full cold compile and do not read it as a regression.")}


def stamp_cache(path: str) -> str:
    side = path + ".env.json"
    json.dump(fingerprint(), open(side, "w"), indent=1)
    return side


# Blocking differences: the comparison is invalid regardless of effect size.
CRITICAL = ("gpu", "sm", "driver", "torch", "triton", "torchao", "cuda")
# Advisory: same hardware model and stack on a different physical node is
# comparable in principle. Thermals, co-tenancy and clock policy can still move a
# median, so a difference SMALLER than node-to-node spread should not be attributed
# to the change under test. Warn; do not refuse.
ADVISORY = ("host",)


def compare(baseline: dict, candidate: dict) -> dict:
    """Refuse a comparison across incompatible environments.

    L4 made executable: only measure what you can compare.
    """
    a = baseline.get("node", baseline)
    b = candidate.get("node", candidate)
    diff = {k: (a.get(k), b.get(k)) for k in set(a) | set(b)
            if k not in ("LD_LIBRARY_PATH",) and a.get(k) != b.get(k)}
    blocking = {k: v for k, v in diff.items() if k in CRITICAL}
    advisories = {k: v for k, v in diff.items() if k in ADVISORY}
    note = ""
    if advisories.get("host"):
        note = (f" ADVISORY -- host {advisories['host'][0]} -> {advisories['host'][1]}: "
                f"same hardware and stack, so the comparison stands, but a difference "
                f"smaller than node-to-node spread should not be attributed to the "
                f"change. For a small expected effect, run both arms back to back on "
                f"one node.")
    return {"comparable": not blocking, "blocking": blocking, "advisories": advisories,
            "other_differences": {k: v for k, v in diff.items()
                                  if k not in CRITICAL and k not in ADVISORY},
            "verdict": (("comparable" + note) if not blocking else
                        "NOT COMPARABLE -- " + "; ".join(
                            f"{k}: {v[0]} -> {v[1]}" for k, v in blocking.items()) +
                        ". Re-run the baseline on this stack rather than comparing." + note)}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json", help="write the record for the ledger")
    p.add_argument("--cache", help="also check a saved compile cache against this stack")
    p.add_argument("--against", help="a previously written env.json to compare with")
    p.add_argument("--deep", action="store_true",
                   help="also enumerate modelopt's quant configs (slow: pulls in Megatron)")
    p.add_argument("--import-target", action="append",
                   help="probe only this module[:symbol]; repeat for selected inference imports")
    p.add_argument("--import-timeout", type=float, default=30,
                   help="seconds allowed per selected import (default: 30)")
    a = p.parse_args()

    if a.import_target:
        if a.cache or a.against or a.deep:
            p.error("selected import probes run separately from --cache, --against and --deep")
        if a.import_timeout <= 0:
            p.error("--import-timeout must be positive")
        report = probe_imports(a.import_target, timeout=a.import_timeout)
        print(json.dumps(report, indent=2))
        if a.json:
            with open(a.json, "w") as fh:
                json.dump(report, fh, indent=2)
        return 0 if report["imports_ok"] else 1

    fp, cap, hz = fingerprint(), capabilities(a.deep), hazards()
    print("STACK")
    for k in ("host", "gpu", "sm", "driver", "python", "torch", "triton", "torchao",
              "transformer_engine", "transformers", "modelopt", "cuda", "cudnn"):
        if fp.get(k) is not None:
            print(f"  {k:20s} {fp[k]}")

    print(f"\nPRECISION PATHS  (this device is sm_{cap['sm']})")
    for r in cap["precision_paths"]:
        mark = "OK " if r["runnable_here"] else ("--?" if r["available"] else "   ")
        why = ("" if r["runnable_here"] else
               f"  needs sm_{r['requires_sm']}" if r["available"] else
               f"  UNAVAILABLE: {r.get('error', 'symbol not found in module')}")
        print(f"  {mark} {r['path']:20s} {r['module']:32s}{why}")
    mo = cap["modelopt"]
    if mo.get("available") and mo.get("probed"):
        print(f"  OK  modelopt             {mo['total_configs']} configs, "
              f"{len(mo['fp4_configs'])} FP4: {', '.join(mo['fp4_configs'][:4])}...")
    elif mo.get("available"):
        print(f"  ?   modelopt             {mo.get('note', 'installed')}")
    present_unrunnable = [r for r in cap["precision_paths"]
                          if r["available"] and not r["runnable_here"]]
    if present_unrunnable:
        print(f"\n  {len(present_unrunnable)} path(s) are INSTALLED but cannot run on this GPU. "
              f"That is a hardware\n  constraint, not a missing feature -- do not record them as "
              f"unavailable.")

    if hz:
        print(f"\nHAZARDS ({len(hz)})")
        for h in hz:
            print(f"  {'!' if h.get('blocking', True) else 'i'} {h['hazard']}")
            print(f"      {h['detail']}")
            print(f"      {h['impact']}")
            if h.get("fix"):
                print(f"      FIX: {h['fix']}")
    else:
        print("\nHAZARDS  none detected")

    if a.cache:
        c = check_cache(a.cache)
        print(f"\nCOMPILE CACHE\n  {c['verdict']}")
        for k, (was, now) in (c.get("diff") or {}).items():
            print(f"    {k}: {was} -> {now}")

    if a.against:
        try:
            with open(a.against) as fh:
                base = json.load(fh)
        except OSError as e:
            # NB: keep fingerprints on shared storage. /tmp is node-local, so a
            # baseline written on one node is simply absent on the next -- and
            # a missing baseline must never discard the report we already have.
            print(f"\nCOMPARABILITY\n  baseline unreadable ({e.strerror}: {a.against}). "
                  f"Keep env.json on shared storage, not node-local /tmp.")
        else:
            r = compare(base.get("fingerprint", base), fp)
            print(f"\nCOMPARABILITY vs {a.against}\n  {r['verdict']}")

    if a.json:
        json.dump({"fingerprint": fp, "capabilities": cap, "hazards": hz},
                  open(a.json, "w"), indent=1)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    raise SystemExit(main())
