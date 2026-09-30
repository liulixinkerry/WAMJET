"""Host-only regression tests for first-output smoke validation.

Usage:  python scripts/test_preflight.py
"""
import inspect
import json
import pathlib
import subprocess
import sys
import tempfile
import types
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from wamjet import preflight  # noqa: E402


fails = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}{'  -- ' + detail if detail and not cond else ''}")
    if not cond:
        fails.append(name)


preflight.fingerprint.cache_clear()
with patch.dict(sys.modules, {"torch": types.SimpleNamespace()}), \
        patch("wamjet.preflight.metadata.version", return_value="test") as versions, \
        patch("wamjet.preflight.subprocess.run", return_value=types.SimpleNamespace(stdout="")), \
        patch("wamjet.preflight.importlib.import_module", side_effect=AssertionError("optional import")):
    first = preflight.fingerprint()
    calls = versions.call_count
    second = preflight.fingerprint()
    check("fingerprint uses metadata without importing optional libraries", first["triton"] == "test")
    check("fingerprint is collected once per process", calls > 0 and versions.call_count == calls)
preflight.fingerprint.cache_clear()


supports_validation = "validate" in inspect.signature(preflight.smoke).parameters
check("smoke accepts a first-output validator", supports_validation)

if supports_validation:
    old_torch = sys.modules.get("torch")
    syncs = []
    sys.modules["torch"] = types.SimpleNamespace(
        cuda=types.SimpleNamespace(synchronize=lambda: syncs.append(True))
    )
    try:
        calls = []

        def infer():
            calls.append(len(calls) + 1)
            return {"action": calls[-1]}

        seen = []
        result = preflight.smoke(
            infer,
            validate=lambda output: seen.append(output["action"]) or {"ok": True},
        )
        check("validator observes the first output", result["ok"] and seen == [1], str(result))
        check("smoke invokes the endpoint exactly once", calls == [1], str(calls))
        check("smoke retains serializable validation", result.get("validation") == {"ok": True})
        check("smoke synchronizes before validation", len(syncs) == 1, str(syncs))

        bad = preflight.smoke(
            lambda: {"action": 0},
            validate=lambda _: {"ok": False, "reason": "wrong first output"},
        )
        check("failed output validation fails smoke", not bad["ok"], str(bad))
        check("failed output validation is preserved", bad.get("validation", {}).get("reason") ==
              "wrong first output", str(bad))

        blocked = preflight.smoke(
            lambda: {"action": 0},
            validate=lambda _: {"blocked": "input mismatch"},
        )
        check("blocked equivalence fails smoke", not blocked["ok"], str(blocked))

        owing = preflight.smoke(
            lambda: {"action": 0},
            validate=lambda _: {"tier": "C", "owes": "G3 paired task evaluation"},
        )
        check("validation that owes a gate fails smoke", not owing["ok"], str(owing))

        tier_c = {"tier": "C", "owes": "G3 paired task evaluation"}
        unproved = preflight.smoke(lambda: 1, validate=lambda _: tier_c, maximum_tier="C")
        check("tier C policy alone cannot pass smoke", not unproved["ok"], str(unproved))
        default_policy = preflight.smoke(
            lambda: 1, validate=lambda _: tier_c, task_evaluation={"passed": True}
        )
        check("task evidence does not override the default numerical policy",
              not default_policy["ok"], str(default_policy))
        passed = preflight.smoke(
            lambda: 1, validate=lambda _: tier_c,
            maximum_tier="C", task_evaluation={"passed": True},
        )
        check("tier C smoke accepts explicit policy and passing task evidence",
              passed["ok"] and passed["validation"] == tier_c, str(passed))
        failed = preflight.smoke(
            lambda: 1, validate=lambda _: tier_c,
            maximum_tier="C", task_evaluation={"passed": False},
        )
        check("failed task evaluation blocks tier C smoke", not failed["ok"], str(failed))
        bypass = preflight.smoke(lambda: 1, validate=lambda _: {"ok": True, **tier_c})
        check("inline ok cannot bypass the numerical policy", not bypass["ok"], str(bypass))

        legacy = preflight.smoke(lambda: 1)
        check("legacy smoke call remains valid", legacy["ok"] and "validation" not in legacy,
              str(legacy))
    finally:
        if old_torch is None:
            del sys.modules["torch"]
        else:
            sys.modules["torch"] = old_torch


imports = preflight.probe_imports(["json:loads", "json:no_such_attribute",
                                  "wamjet_nonexistent_dependency"])
check("selected module and symbol can import", imports["imports"][0]["ok"])
check("API mismatch retains its actual traceback",
      not imports["imports"][1]["ok"] and "AttributeError" in imports["imports"][1]["stderr"])
check("missing dependency retains its actual traceback",
      not imports["imports"][2]["ok"] and "ModuleNotFoundError" in imports["imports"][2]["stderr"])
check("mixed import results do not claim readiness", not imports["imports_ok"])
with patch("wamjet.preflight.subprocess.run",
           side_effect=subprocess.TimeoutExpired("probe", 1, stderr=b"partial traceback")):
    timed = preflight.probe_imports(["slow.module"], timeout=1)
check("slow imports produce a bounded diagnostic",
      timed["imports"][0]["status"] == "timeout" and
      timed["imports"][0]["stderr"] == "partial traceback")
with tempfile.TemporaryDirectory() as td:
    path = pathlib.Path(td) / "imports.json"
    for target, expected in (("json:loads", 0), ("json:no_such_attribute", 1)):
        process = subprocess.run([sys.executable, "-m", "wamjet.preflight",
                                  "--import-target", target, "--json", str(path)],
                                 capture_output=True, text=True)
        report = json.loads(path.read_text())
        check(f"import CLI returns {expected} for {target}", process.returncode == expected)
        check("import-only CLI skips GPU capability scanning",
              "capabilities" not in report and report["executable"] == sys.executable)

print()
if fails:
    print(f"  {len(fails)} failure(s): {', '.join(fails)}")
    sys.exit(1)
print("  all preflight checks passed")
