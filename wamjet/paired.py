"""Compare task outcomes on the same fixed initial states and seeds.

Report per-episode changes, McNemar's exact test, and the smallest one-sided
regression this sample could detect. Task screening informs optimization;
it does not control whether the campaign continues.

Input: one JSON per variant mapping a stable episode key to a boolean, e.g.
    {"task3/episode7": true, ...}
Keys must be identical across variants -- that is what makes it paired.

Usage:  python -m wamjet.paired baseline=A.json candidate=B.json
"""
from __future__ import annotations

import argparse
import json
from math import comb


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar on the discordant pairs only.

    b = baseline-only successes, c = candidate-only successes. Under the null
    each discordant pair is a fair coin, so this is a two-sided binomial test.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def min_detectable(n: int, alpha: float = 0.05) -> int:
    """Smallest number of one-sided discordant episodes that would be significant."""
    for d in range(1, n + 1):
        if mcnemar_exact(d, 0) <= alpha:
            return d
    return n + 1


def pair(base: dict, cand: dict) -> dict:
    base_keys, cand_keys = set(base), set(cand)
    if not base_keys or base_keys != cand_keys:
        raise ValueError("paired evaluation requires identical, nonempty episode sets: "
                         f"{len(base_keys - cand_keys)} missing from candidate, "
                         f"{len(cand_keys - base_keys)} extra in candidate")
    shared = sorted(base_keys)
    both = sum(1 for k in shared if base[k] and cand[k])
    bonly = sum(1 for k in shared if base[k] and not cand[k])
    conly = sum(1 for k in shared if not base[k] and cand[k])
    neither = len(shared) - both - bonly - conly
    p = mcnemar_exact(bonly, conly)
    return {
        "n_paired": len(shared),
        "base_successes": both + bonly,
        "cand_successes": both + conly,
        "base_rate": (both + bonly) / len(shared),
        "cand_rate": (both + conly) / len(shared),
        "both": both, "baseline_only": bonly, "candidate_only": conly, "neither": neither,
        "discordant": bonly + conly,
        "mcnemar_p": p,
        "regressed": p <= 0.05 and bonly > conly,
        "min_detectable_discordant": min_detectable(len(shared)),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("variants", nargs="+", metavar="name=results.json",
                   help="first is the baseline; the rest are candidates")
    a = p.parse_args()
    named = [v.split("=", 1) for v in a.variants]
    if len(named) < 2:
        raise SystemExit("need a baseline and at least one candidate")
    (bname, bpath), rest = named[0], named[1:]
    base = json.load(open(bpath))

    print(f"baseline: {bname}  ({sum(base.values())}/{len(base)})")
    for cname, cpath in rest:
        try:
            r = pair(base, json.load(open(cpath)))
        except ValueError as exc:
            p.error(str(exc))
        print(f"\n{cname} vs {bname}   ({r['n_paired']} paired episodes)")
        print(f"  {bname:<12} {r['base_successes']:>3}/{r['n_paired']}  "
              f"{r['base_rate']*100:.1f}%")
        print(f"  {cname:<12} {r['cand_successes']:>3}/{r['n_paired']}  "
              f"{r['cand_rate']*100:.1f}%")
        print(f"  pairing: {r['both']} both, {r['baseline_only']} baseline-only, "
              f"{r['candidate_only']} candidate-only, {r['neither']} neither")
        print(f"  McNemar exact p = {r['mcnemar_p']:.3f}"
              + ("   *** REGRESSION ***" if r["regressed"] else "   no detected change"))
        print(f"  power: this run could only have flagged a regression of "
              f"{r['min_detectable_discordant']}+ one-sided discordant episodes. "
              f"A smaller true change would not show up here.")


if __name__ == "__main__":
    main()
