"""Rung 0 -- make iteration cheap, and measure it without lying.

Startup does not change inference latency. It sets how many experiments are
possible per day, which is why it is the first rung: an optimization campaign is
bounded by its iteration count long before it is bounded by ideas.

It is also the rung whose measurements are easiest to corrupt. Build time is
dominated by reading weights, so it is dominated by the **page cache**, which no
part of the build reports. The same tree measured back to back can differ
several fold purely on cache warmth -- a swing far larger than any change worth
making. A cold-vs-warm comparison will therefore confirm almost any hypothesis,
including a wrong one, and nothing in the timing output hints at it.

So this module measures cache residency directly rather than warning about it.
`residency` reports what fraction of each weight file the kernel already holds;
`warm` puts both arms in the same state, which is the only control available
without root. Compare cold-to-cold or warm-to-warm, and say which.

The second trap is attribution. The obvious defect is often not the cost: a
loader that reads a checkpoint into host memory before copying it to the device
is a textbook anti-pattern and can still be a small fraction of build time,
while parameter initialization -- which looks like nothing, and which no
profiler attributes to a line you wrote -- is the majority. Split by phase
before changing anything.

Usage:

    from wamjet.startup import Phase, residency, warm

    warm(ckpt, weights)                      # or measure cold; state which
    p = Phase()
    cfg   = p("config",        build_config)
    model = p("construct",     lambda: build(cfg))
    _     = p("task weights",  lambda: model.load_checkpoint(path))
    p.report()

    python -m wamjet.startup /path/to/*.safetensors
"""
from __future__ import annotations

import ctypes
import os
import time

_PAGE = os.sysconf("SC_PAGE_SIZE")
_PROT_READ = 1
_MAP_SHARED = 1


def residency(*paths: str) -> dict:
    """Fraction of each file the kernel currently holds in the page cache.

    Uses `mincore`, which reports residency per page without reading the file --
    reading it to find out would populate the cache being measured.
    """
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                          ctypes.c_int, ctypes.c_int, ctypes.c_long]
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                             ctypes.POINTER(ctypes.c_ubyte)]

    out = {}
    for path in paths:
        try:
            size = os.path.getsize(path)
            if size == 0:
                out[path] = {"bytes": 0, "resident": 1.0}
                continue
            fd = os.open(path, os.O_RDONLY)
            try:
                addr = libc.mmap(None, size, _PROT_READ, _MAP_SHARED, fd, 0)
                if addr in (None, 0, ctypes.c_void_p(-1).value):
                    out[path] = {"bytes": size, "resident": None,
                                 "error": "mmap failed"}
                    continue
                try:
                    pages = (size + _PAGE - 1) // _PAGE
                    vec = (ctypes.c_ubyte * pages)()
                    if libc.mincore(ctypes.c_void_p(addr), size, vec) != 0:
                        out[path] = {"bytes": size, "resident": None,
                                     "error": os.strerror(ctypes.get_errno())}
                        continue
                    hot = sum(1 for b in vec if b & 1)
                    out[path] = {"bytes": size, "resident": hot / pages}
                finally:
                    libc.munmap(ctypes.c_void_p(addr), size)
            finally:
                os.close(fd)
        except OSError as e:
            out[path] = {"bytes": None, "resident": None, "error": str(e)}

    known = [v["resident"] for v in out.values() if v.get("resident") is not None]
    total = sum(v["bytes"] for v in out.values() if v.get("bytes")) or 0
    hot_bytes = sum(v["bytes"] * v["resident"] for v in out.values()
                    if v.get("bytes") and v.get("resident") is not None)
    return {"files": out,
            "bytes": total,
            "resident": (hot_bytes / total) if total else None,
            "state": ("warm" if known and min(known) > 0.95 else
                      "cold" if known and max(known) < 0.05 else
                      "mixed" if known else "unknown")}


def warm(*paths: str) -> dict:
    """Read files so both arms of a comparison start from the same cache state.

    Warming is the only control available without root -- dropping the cache
    needs privilege that a scheduled job does not have. Warming both arms is
    equally valid as long as it is reported, and it is far cheaper to repeat.
    """
    t0 = time.perf_counter()
    read = 0
    for path in paths:
        with open(path, "rb", buffering=0) as f:
            while chunk := f.read(1 << 24):
                read += len(chunk)
    dt = time.perf_counter() - t0
    return {"bytes": read, "seconds": dt,
            "GB_per_s": (read / 1e9 / dt) if dt else None,
            "after": residency(*paths)["state"]}


class Phase:
    """Time an ordered sequence of build steps, with device memory after each."""

    def __init__(self, sync: bool = True):
        self.rows: list[tuple[str, float, float | None]] = []
        self._sync = sync

    def _synchronize(self):
        if not self._sync:
            return
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass

    def _allocated(self):
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / 1e9
        except Exception:
            pass
        return None

    def __call__(self, tag: str, fn):
        self._synchronize()
        t0 = time.perf_counter()
        result = fn()
        self._synchronize()
        self.rows.append((tag, time.perf_counter() - t0, self._allocated()))
        return result

    def report(self, cache_state: str | None = None) -> dict:
        total = sum(r[1] for r in self.rows)
        width = max((len(r[0]) for r in self.rows), default=10)
        print(f"\n  startup by phase   (page cache: {cache_state or 'not measured'})")
        for tag, dt, mem in self.rows:
            share = f"{100 * dt / total:4.0f}%" if total else "   -"
            gpu = f"  gpu {mem:5.1f} GB" if mem is not None else ""
            print(f"    {tag:{width}s}  {dt:7.1f}s  {share}{gpu}")
        print(f"    {'TOTAL':{width}s}  {total:7.1f}s")
        if cache_state is None:
            print("\n  Cache state was not recorded, so this total is not comparable "
                  "with another run. Call residency() on the weight files.")
        print("  One build is one sample. Host contention moves a startup total by "
              "multiples without touching cache state, so repeat before attributing a "
              "difference to a change.")
        return {"phases": [{"tag": t, "seconds": d, "gpu_GB": m} for t, d, m in self.rows],
                "total_seconds": total, "cache_state": cache_state}


def main():
    import argparse
    import json
    p = argparse.ArgumentParser(description="Report page-cache residency of weight files.")
    p.add_argument("paths", nargs="+")
    p.add_argument("--warm", action="store_true", help="read the files first")
    p.add_argument("--json", action="store_true")
    a = p.parse_args()

    if a.warm:
        w = warm(*a.paths)
        print(f"  warmed {w['bytes']/1e9:.1f} GB in {w['seconds']:.1f}s "
              f"({w['GB_per_s']:.2f} GB/s)")
    r = residency(*a.paths)
    if a.json:
        print(json.dumps(r, indent=2))
        return
    for path, v in r["files"].items():
        if v.get("resident") is None:
            print(f"    {'?':>6}  {os.path.basename(path)}  ({v.get('error')})")
        else:
            print(f"    {v['resident']*100:5.1f}%  {v['bytes']/1e9:6.2f} GB  "
                  f"{os.path.basename(path)}")
    if r["resident"] is not None:
        print(f"\n  {r['resident']*100:.1f}% of {r['bytes']/1e9:.1f} GB resident "
              f"-- state: {r['state']}")
        print("  Compare cold-to-cold or warm-to-warm only, and report which.")


if __name__ == "__main__":
    main()
