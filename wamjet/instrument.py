"""WAMJET gate G0 -- is this latency number allowed to exist?

Three comparability rules are enforced here:

  1. Measure the compiled public endpoint. Eager and isolated-loop results do
     not represent the build being optimized.
  2. Get past the warmup cliff.  With `dynamic=False`, a growing KV cache mints
     a fresh Inductor graph per length.  Until the cache wraps, the model shows
     a false steady state before falling to a different level. A fixed trailing
     window can straddle that cliff and report a contaminated median.
  3. Compare on one compatible stack, back to back.

`steady()` finds the cliff instead of assuming a warmup count: the steady
regime is the longest suffix of the call series that holds a stable level.

Usage as a library:

    from wamjet.instrument import Timer, steady, report
    t = Timer()
    for _ in range(40):
        with t:
            policy.infer(obs)
    report(steady(t.ms), node=..., tag=...)
"""
from __future__ import annotations

import json
import statistics as st
import subprocess
import threading
import time


class Timer:
    """CUDA-synchronised wall clock. Sync inside the timer, never around it."""

    def __init__(self):
        self.ms: list[float] = []
        self._t0 = 0.0

    def __enter__(self):
        import torch
        torch.cuda.synchronize()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_):
        import torch
        torch.cuda.synchronize()
        self.ms.append((time.perf_counter() - self._t0) * 1e3)
        return False


def steady(ms: list[float], tol: float = 0.15, min_frac: float = 0.25) -> dict:
    """Longest stable suffix of a latency series, plus what was cut off.

    tol      -- a call belongs to the regime if it is within tol of the
                regime median. 15% is loose enough for real jitter and tight
                enough to reject a 4x warmup plateau.
    min_frac -- refuse to report if the stable suffix is shorter than this
                fraction of the run; that means the cliff never happened and
                the number is still contaminated.
    """
    n = len(ms)
    best = None
    for start in range(n - 1):                      # longest suffix wins
        w = ms[start:]
        if len(w) < 3:
            break
        m = st.median(w)
        if all(abs(x - m) <= tol * m for x in w):
            best = (start, w, m)
            break
    stable = best is not None
    if best is None:
        start, w = n - 3, ms[-3:]
        m = st.median(w)
        best = (start, w, m)
    start, w, m = best
    pre = ms[:start]
    return {
        "n_calls": n,
        "warmup_calls": start,
        "steady_n": len(w),
        "median_ms": m,
        "mean_ms": st.mean(w),
        "sd_ms": st.stdev(w) if len(w) > 1 else 0.0,
        "min_ms": min(w),
        "max_ms": max(w),
        "cv_pct": (st.stdev(w) / m * 100) if len(w) > 1 else 0.0,
        "cliff_ratio": (st.median(pre) / m) if pre else 1.0,
        "trusted": stable and len(w) >= max(3, int(n * min_frac)),
        "calls": ms,
    }


def node_fingerprint() -> dict:
    """Return the same environment record used for comparability checks."""
    from .preflight import fingerprint

    return fingerprint()


class PowerSampler(threading.Thread):
    """Poll power and SM clock to expose power-capped low-precision runs."""

    def __init__(self, period=0.1, index=0):
        super().__init__(daemon=True)
        self.period, self.index = period, index
        self.stop_flag, self.rows = threading.Event(), []

    def run(self):
        q = ["nvidia-smi", "--query-gpu=utilization.gpu,power.draw,clocks.sm",
             "--format=csv,noheader,nounits", "-i", str(self.index)]
        while not self.stop_flag.is_set():
            try:
                o = subprocess.run(q, capture_output=True, text=True, timeout=5).stdout
                self.rows.append((time.perf_counter(),
                                  *[float(x) for x in o.strip().split(",")]))
            except Exception:
                pass
            self.stop_flag.wait(self.period)

    def summary(self, t0=None, t1=None):
        r = [x for x in self.rows if (t0 is None or t0 <= x[0] <= t1)]
        if not r:
            return {}
        return {"util_mean_pct": st.mean(x[1] for x in r),
                "power_mean_w": st.mean(x[2] for x in r),
                "power_max_w": max(x[2] for x in r),
                "sm_clock_mean_mhz": st.mean(x[3] for x in r),
                "samples": len(r)}


def report(s: dict, tag: str = "run", extra: dict | None = None,
           out: str | None = None) -> dict:
    rec = {"tag": tag, "node": node_fingerprint(), **s, **(extra or {})}
    print(f"[{tag}] {s['median_ms']:.1f} ms median  "
          f"(mean {s['mean_ms']:.1f}, sd {s['sd_ms']:.1f}, cv {s['cv_pct']:.1f}%, "
          f"n={s['steady_n']}/{s['n_calls']})")
    if s["warmup_calls"]:
        print(f"       cliff at call {s['warmup_calls']}: warmup ran "
              f"{s['cliff_ratio']:.2f}x the steady level "
              f"-- a trailing-window median would have been contaminated")
    if not s["trusted"]:
        print(f"       *** UNTRUSTED: only {s['steady_n']} of {s['n_calls']} calls are "
              f"stable. The compile cliff has not happened yet. Run more calls. ***")
    if out:
        json.dump(rec, open(out, "w"), indent=1)
        print(f"       wrote {out}")
    return rec
