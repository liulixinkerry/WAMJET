"""Validate this harness's skill files against published skill-authoring guidance.

Run after ANY change to skills/ or the documents an agent reads. Guidance sources:
platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices and the
overview page's frontmatter rules.

The checker self-tests its own patterns first. A validator whose regexes silently
match nothing is worse than no validator: it converts an unchecked file into a
falsely-checked one. This module's first audit did exactly that -- escaped patterns
inside a shell heredoc matched nothing and reported a clean file that was not.

Usage:  python scripts/check_skill.py [skills_dir]
Exit 0 if every check passes, 1 otherwise.
"""
from __future__ import annotations

import pathlib
import re
import sys

RESERVED = ("anthropic", "claude")
NAME_RE = re.compile(r"^[a-z0-9-]{1,64}$")
# A bare capital I is unreliable to detect by word boundary in technical prose: it
# appears in I/O, in math such as N(0,I), and inside identifiers. Require it to look
# like a pronoun -- not adjacent to a slash, comma, parenthesis or word character --
# and rely on the unambiguous pronouns for the rest.
FIRST_PERSON = re.compile(
    r"(?:(?<![\w,(/])I(?![\w/)])|\b(?:[Mm]y|[Mm]e|[Ww]e|[Oo]ur|[Uu]s)\b)")
MEASUREMENT = re.compile(r"\d+\.\d+\s*(?:ms|x|%|GB|MB|s)\b")
DATE = re.compile(r"\b20\d\d-\d\d-\d\d\b")
WIN_PATH = re.compile(r"[A-Za-z0-9_]\\[A-Za-z0-9_]+\.(?:py|md|json)")


def self_test() -> None:
    """Prove each pattern matches what it claims. Fail loudly if not."""
    cases = [
        (FIRST_PERSON, "we measured this", True),
        (FIRST_PERSON, "the campaign measured this", False),
        (FIRST_PERSON, "I observed this", True),
        (FIRST_PERSON, "compute rather than I/O is constant", False),
        (FIRST_PERSON, "bus I/O and my notes", True),
        (FIRST_PERSON, "We took the measurement", True),      # sentence-initial
        (FIRST_PERSON, "Our approach", True),
        (FIRST_PERSON, "The approach", False),
        (FIRST_PERSON, "same N(0,I) distribution", False),   # identity matrix
        (FIRST_PERSON, "an H100 and H10 label", False),
        (FIRST_PERSON, "I measured it", True),
        (MEASUREMENT, "took 8.3 ms", True),
        (MEASUREMENT, "took several ms", False),
        (DATE, "on 2026-08-28", True),
        (DATE, "on Tuesday", False),
        (WIN_PATH, r"scripts\helper.py", True),
        (WIN_PATH, "scripts/helper.py", False),
        (NAME_RE, "optimizing-policy-inference", True),
        (NAME_RE, "Optimizing_Policy", False),
    ]
    for pat, text, want in cases:
        got = bool(pat.search(text) if pat is not NAME_RE else pat.match(text))
        if got != want:
            raise SystemExit(f"SELF-TEST FAILED: {pat.pattern!r} on {text!r} "
                             f"gave {got}, expected {want}. Fix the checker before "
                             f"trusting any result it prints.")


def check_skill(d: pathlib.Path) -> list[str]:
    errs, warns = [], []
    md = d / "SKILL.md"
    if not md.exists():
        return [f"{d}: no SKILL.md"]
    text = md.read_text()
    lines = text.splitlines()

    fm = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if not fm:
        errs.append(f"{md}: missing YAML frontmatter")
    else:
        body = fm.group(1)
        name = re.search(r"^name:\s*(.+)$", body, re.M)
        desc = re.search(r"^description:\s*(.+)$", body, re.M)
        if not name:
            errs.append(f"{md}: frontmatter has no `name`")
        else:
            n = name.group(1).strip()
            if not NAME_RE.match(n):
                errs.append(f"{md}: name {n!r} must be 1-64 chars of [a-z0-9-]")
            if any(r in n.lower() for r in RESERVED):
                errs.append(f"{md}: name {n!r} contains a reserved word")
            if n != d.name:
                errs.append(f"{md}: name {n!r} must match directory {d.name!r}")
        if not desc:
            errs.append(f"{md}: frontmatter has no `description`")
        else:
            v = desc.group(1).strip()
            if len(v) > 1024:
                errs.append(f"{md}: description is {len(v)} chars (max 1024)")
            if "<" in v and ">" in v:
                errs.append(f"{md}: description must not contain XML tags")
            if not re.search(r"\bUse when\b", v):
                errs.append(f"{md}: description must state WHEN to use the skill "
                            f"(include a 'Use when ...' clause)")
            if not re.search(r"\bNot for\b|\bnot for\b", v):
                warns.append(f"{md}: description has no negative trigger "
                             f"('Not for ...'), which raises false activation")
            if FIRST_PERSON.search(v):
                errs.append(f"{md}: description must be third person")

    if len(lines) > 500:
        errs.append(f"{md}: body is {len(lines)} lines (keep under 500; split into "
                    f"references/)")

    for i, l in enumerate(lines, 1):
        if FIRST_PERSON.search(l):
            errs.append(f"{md}:{i}: first person — use third-person imperative")
        if MEASUREMENT.search(l):
            errs.append(f"{md}:{i}: session-specific measurement — evidence belongs "
                        f"in campaign artifacts; state the mechanism here")
        if DATE.search(l):
            errs.append(f"{md}:{i}: time-sensitive date")
        if WIN_PATH.search(l):
            errs.append(f"{md}:{i}: Windows-style path — use forward slashes")

    for ref in sorted((d / "references").glob("*.md")) if (d / "references").is_dir() else []:
        rl = ref.read_text().splitlines()
        if len(rl) > 100 and not any(l.strip().lower().startswith("## contents") for l in rl):
            errs.append(f"{ref}: {len(rl)} lines without a '## Contents' table of "
                        f"contents (required above 100 lines)")
        for i, l in enumerate(rl, 1):
            if FIRST_PERSON.search(l):
                errs.append(f"{ref}:{i}: first person")
            if MEASUREMENT.search(l):
                errs.append(f"{ref}:{i}: session-specific measurement")

    # references must be one level deep: a referenced file may not reference another
    for ref in sorted((d / "references").glob("*.md")) if (d / "references").is_dir() else []:
        for i, l in enumerate(ref.read_text().splitlines(), 1):
            if re.search(r"\]\((?!http)[^)]*\.md\)", l):
                errs.append(f"{ref}:{i}: nested reference — keep all references one "
                            f"level deep from SKILL.md")

    for w in warns:
        print(f"  WARN  {w}")
    return errs


def check_prose(path: pathlib.Path, allow_measurements: bool) -> list[str]:
    """Voice check for documents an agent reads outside skills/.

    Measurements are permitted only when allow_measurements is set. Keep
    session-specific evidence in campaign artifacts, outside reusable guidance.
    """
    errs = []
    if not path.exists():
        return errs
    for i, l in enumerate(path.read_text().splitlines(), 1):
        if FIRST_PERSON.search(l):
            errs.append(f"{path}:{i}: first person — use third-person imperative")
        if not allow_measurements and MEASUREMENT.search(l):
            errs.append(f"{path}:{i}: session-specific measurement in guidance")
    return errs


def main() -> int:
    self_test()
    # Default relative to this file, not to the caller's working directory: a
    # checker that passes or fails depending on where it was invoked from is a
    # checker people route around, and a gate that gets routed around protects
    # nothing.
    default = pathlib.Path(__file__).resolve().parent.parent / "skills"
    root = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else default
    dirs = [p for p in root.iterdir() if p.is_dir()] if root.is_dir() else []
    if not dirs:
        print(f"no skill directories under {root}")
        return 1
    all_errs = []
    for d in sorted(dirs):
        errs = check_skill(d)
        all_errs += errs
        print(f"  {'FAIL' if errs else 'PASS'}  {d}")
        for e in errs:
            print(f"          {e}")
    # Documents outside skills/ that an agent still reads.
    root_dir = root.parent if root.name == "skills" else pathlib.Path(".")
    # EVAL.md is guidance, so it is held to the no-measurements rule that skills
    # are: a protocol that needs one run's numbers to state itself is not a protocol.
    for name, allow in (("README.md", True), ("EVAL.md", False)):
        if not (root_dir / name).is_file():
            continue
        errs = check_prose(root_dir / name, allow_measurements=allow)
        all_errs += errs
        print(f"  {'FAIL' if errs else 'PASS'}  {root_dir / name}")
        for e in errs[:6]:
            print(f"          {e}")
        if len(errs) > 6:
            print(f"          ... and {len(errs) - 6} more")

    print(f"\n  {len(all_errs)} error(s)")
    return 1 if all_errs else 0


if __name__ == "__main__":
    sys.exit(main())
