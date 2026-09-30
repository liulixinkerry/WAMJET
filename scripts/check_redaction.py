"""Verify that a sandbox's harness copy does not contain the answers.

EVAL.md requires the findings an evaluation is meant to reproduce to be absent
from the sandbox. Checking that by hand is the step most likely to be skipped,
and a skipped redaction produces a run that looks successful and means nothing.

Matches against the harness copy only. The policy tree legitimately contains its
own identifiers -- the source is what the agent under test is meant to read.

Usage:  python scripts/check_redaction.py <sandbox-harness-dir> <answer-key.json>
"""
import json
import pathlib
import re
import sys

TEXT = {".md", ".py", ".txt", ".json", ".yaml", ".yml", ".sh", ".html"}
# Recorded traces, compile caches and nested checkouts are megabytes to gigabytes
# of machine output that hold no guidance. Scanning them turns a check that should
# take a second into one that times out, and a check that times out is not run.
SKIP = {".git", ".cache", ".refs", "ledger", "__pycache__", "node_modules", ".venv"}
MAX_BYTES = 2 << 20


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    root = pathlib.Path(sys.argv[1])
    key = json.load(open(sys.argv[2]))
    if not root.is_dir():
        print(f"not a directory: {root}")
        return 2

    files = [p for p in root.rglob("*")
             if p.is_file() and p.suffix.lower() in TEXT
             and not SKIP & set(p.parts)
             and p.stat().st_size <= MAX_BYTES]
    terms = [(f["id"], t, re.compile(re.escape(t), re.IGNORECASE))
             for f in key["findings"] for t in f["terms"]]
    # Result figures are answers too. A run told the target in advance is not
    # reproducing it, and the line that leaks it may name no symbol and no policy
    # -- it simply quotes the headline, which no term will ever match.
    terms += [("result-figure", n, re.compile(r"(?<![\d.])" + re.escape(n) + r"(?![\d])"))
              for n in key.get("forbidden_numbers", [])]
    leaks = []
    for p in files:
        try:
            lines = p.read_text(errors="ignore").splitlines()
        except OSError:
            continue
        for i, line in enumerate(lines, 1):
            for fid, term, pat in terms:
                if pat.search(line):
                    leaks.append((fid, term, p.relative_to(root), i))

    print(f"  scanned {len(files)} files under {root}")
    print(f"  answer key: {key['policy']}, {len(key['findings'])} findings, "
          f"{sum(len(f['terms']) for f in key['findings'])} terms")
    if not leaks:
        print("\n  CLEAN -- no identifying term found. The sandbox does not contain "
              "the answers.")
        return 0
    print(f"\n  {len(leaks)} LEAK(S) -- this sandbox would invalidate a run:\n")
    for fid, term, rel, line in leaks[:40]:
        print(f"    {fid:32s} {term!r} at {rel}:{line}")
    if len(leaks) > 40:
        print(f"    ... and {len(leaks) - 40} more")
    print("\n  Remove these before starting an evaluation. A general description of "
          "the defect class is the harness working; naming the instance is the answer.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
