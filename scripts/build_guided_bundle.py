"""Build a redaction-checked guided-harness bundle for a blind-evaluation sandbox.

The blind-evaluation procedure in EVAL.md needs a copy of the generic, policy-agnostic
parts of WAMJET. Assembling that by hand is the step most likely to miss a file
that quietly names a policy, a config flag, or a result figure. This builds the
copy from an explicit allowlist (never an exclude-list, which silently admits anything added later)
and then runs check_redaction.py against every answer key this repo knows
about. A bundle that fails any check must not be handed to an agent under
evaluation.

Usage: python scripts/build_guided_bundle.py <dest-dir>
"""
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Generic and policy-agnostic: the skill, the instrument package, the test
# scripts, one benchmark adapter with no prior candidate or result baked in.
# Deliberately excludes README.md, USAGE.md, PROMPTS.md, EVAL.md, assets/,
# .claude/, eval/, ledger/, .refs/, .cache/, .git, and
# check_redaction.py / check_skill.py: these are sandbox-building tools for
# whoever runs this script, not harness code the agent under test has any
# use for, and giving them out is needless surface area for no benefit.
INCLUDE = [
    "skills",
    "wamjet",
    "scripts/test_capture.py",
    "scripts/test_equivalence.py",
    "scripts/test_modules.py",
    "scripts/test_preflight.py",
    "examples/benchmark.py",
    "pyproject.toml",
]

# NOTE ON WHO RUNS THIS SCRIPT: it globs eval/findings_*.json and
# ledger/**/findings-*.json from the full WAMJET checkout and prints the
# result of check_redaction.py, which -- even on a CLEAN run -- names the
# policy, its commit, and a finding/term count, and -- on a LEAK -- prints
# the leaked term itself. All of that is answer-key content. Run this only
# from an orchestrator context that already has full WAMJET repo access and
# is not itself the agent under test; hand the agent only the resulting
# `dest` directory, never this script, never eval/ or ledger/, and never
# this script's output.


def copy_tree(dest: pathlib.Path) -> None:
    dest.mkdir(parents=True)
    for rel in INCLUDE:
        src = ROOT / rel
        dst = dest / rel
        if src.is_dir():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def answer_keys() -> list[pathlib.Path]:
    return sorted(ROOT.glob("eval/findings_*.json")) + sorted(ROOT.glob("ledger/**/findings-*.json"))


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    dest = pathlib.Path(sys.argv[1]).resolve()
    copy_tree(dest)
    keys = answer_keys()
    if not keys:
        print("no answer keys found under eval/ or ledger/ -- nothing to check against")
        return 0
    ok = True
    for key in keys:
        print(f"--- checking against {key.relative_to(ROOT)} ---")
        rc = subprocess.call([sys.executable, str(ROOT / "scripts" / "check_redaction.py"),
                               str(dest), str(key)])
        ok = ok and rc == 0
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
