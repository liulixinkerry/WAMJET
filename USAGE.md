# WAMJET usage guide

WAMJET is a skill and a small set of tools for accelerating world-action-model inference.
The [optimization skill](skills/optimizing-policy-inference/SKILL.md) owns the workflow:
establish a runnable baseline, find a bottleneck, edit, measure, keep or revert, repeat,
then validate the best result within budget.

Use the optimization skill directly for a policy, or invoke the
[launcher](.claude/skills/wamjet-launch/SKILL.md) and its
[template](.claude/skills/wamjet-launch/template.md) to prepare a campaign. Supply model
sources, checkpoints, workload, interpreter, resources and time budget. The launcher
reads external memory only when the user explicitly supplies its path.

The worker follows the optimization skill and reads its references as needed. This
guide describes the repository and tool APIs; it is not a required worker instruction.
Keep model-specific findings and experiment artifacts in the campaign workspace. After
a run, make small edits or deletions to reusable guidance that helped or wasted time.

For a blind comparison, explicitly request the launcher's blind-evaluation procedure
in [EVAL.md](EVAL.md). Its guided bundle contains the optimization skill, generic tools,
selected tests and the shared runner. It excludes this guide, the README and the launcher.
Keep prior findings, external memory and Git history outside worker access.

Keep checkpoints, inputs, seeds and workloads fixed when comparing candidates. Reuse
one allocation and physically staged local checkpoints. Optimize loading when it limits
iteration, and report startup savings separately from endpoint speedup. A short record
of each experiment is enough. Missing broad tests or task evaluation does not stop
exploration; label pending quality checks and validate promising finalists.

## Tools

Use tools when they answer a question; there is no required sequence of gates.

| Question | Tool |
|---|---|
| Is the endpoint runnable on this stack? | `wamjet.preflight` |
| What is its warmed latency? | `wamjet.capture.measure`, `wamjet.instrument` |
| Where does it spend time? | `wamjet.capture.capture`, `pipeline`, `attribute`, `fusion` |
| Did outputs change? | `wamjet.equivalence` |
| Did task success change? | `wamjet.paired` |
| Does latency fit the action horizon? | `wamjet.deadline` |
| Does loading limit iteration? | `wamjet.startup` |

The [detection recipes](skills/optimizing-policy-inference/references/detection.md),
[setup notes](skills/optimizing-policy-inference/references/setup.md) and
[approximation checks](skills/optimizing-policy-inference/references/approximate.md)
are optional references.

## Setup and timing

Install into an environment where package changes are permitted; otherwise use the
assigned interpreter and existing libraries. Numerical comparisons need NumPy, and GPU
measurement needs the policy's PyTorch environment. The commands below use `python` to
stand for the assigned interpreter.

```bash
python -m pip install -e '.[numerics]'
python -m wamjet.preflight --import-target package.module:RequiredClass --json imports.json
```

When installation is not permitted, expose the checkout to the existing interpreter:

```bash
export PYTHONPATH="/path/to/WAMJET${PYTHONPATH:+:$PYTHONPATH}"
```

Selected imports run in fresh processes with a timeout and retained tracebacks. They do
not load checkpoints unless the imported module does so itself. Broad capability and
linker probes are available through `python -m wamjet.preflight --json env.json` when
needed. Ordinary fingerprinting reads package metadata and caches one snapshot per
process; call `fingerprint.cache_clear()` if deliberately changing that process's stack.

```python
from wamjet.capture import measure, capture

result = measure(lambda: policy.infer(obs), warmup=40)
# Collect a trace when diagnosing a bottleneck:
capture(lambda: policy.infer(obs), "trace.json", warmup=40, calls=2)
```

`measure` returns synchronized timings, the steady suffix and its trust status without
opening a profiler. `capture` also writes a trace and adjacent metadata. Both preserve
the caller's execution mode. Unstable timing returns `trusted=false` for diagnosis;
`strict=True` opts into raising an error. Long calls have no default cutoff; an explicit
`stall_s` rejects a completed call over that duration. Job limits bound total runtime.
Inflated traces cannot establish scheduling gaps; summed
kernel durations may overlap and do not measure elapsed GPU busy time.

```bash
python -m wamjet.pipeline trace.json --calls 2 --json pipeline.json
python -m wamjet.attribute trace.json --calls 2 --by module
python -m wamjet.fusion trace.json --calls 2
```

## Output checks

```python
from wamjet.equivalence import noise_floor, save_reference, check_reference

floor = noise_floor(policy.infer, obs, repeats=3)
# Illustrative limits; justify them for the model's precision and output scale.
tolerance = {"per_key": {"action": {"max_abs": 1e-3, "rel_l2": 1e-3}}}
save_reference("baseline", policy.infer(obs), floor=floor, inputs=obs,
               tier_a_tolerance=tolerance)
result = check_reference("baseline", candidate.infer(obs), inputs=obs)
```

The skill's Tier A preserves mathematical computation, workload, precision and state
semantics. Equivalent fusion and GEMM/reduction reordering may introduce roundoff;
bit identity is optional. Review the transformation and investigate unexpected drift.
Retain the original reference and declare per-output limits for automated numerical
checks. Numerical closeness alone cannot establish mathematical equivalence. The
[skill](skills/optimizing-policy-inference/SKILL.md#correctness) defines method eligibility
separately from the checker's numerical labels.

The numerical checker reports A for exact or within-tolerance outputs, B for drift within the
baseline noise floor, and C otherwise. Without a tolerance, A requires exact outputs.
B/C acceptance requires passing `task_evaluation` within `maximum_tier`. Checks cover
numeric output keys, shapes, dtypes and finiteness, including nested outputs; re-record
legacy references that omitted these. For state/cache/graph changes, use changed-input
and reset checks; `check_sequence` compares A→B→A replays using the saved Tier-A limits.

## Shared benchmark runner

The generic [runner](examples/benchmark.py) accepts a `build()` function returning
`Adapter(infer, inputs, metadata, reset=None)`. It loads once, checks the first output
and measures repeated endpoint latency. Stateful adapters supply `reset` so baseline
samples start from the same state. Model-specific adapters are supplied with each
experiment and are not included in this repository.

In the experiment's `policy_adapter.py`, define `build()` for the supplied policy and
invoke the shared entry point:

```python
from benchmark import Adapter, main

# Define build() here: load the policy and return an Adapter.

if __name__ == "__main__":
    raise SystemExit(main(build, prefix="POLICY", root="/path/to/campaign"))
```

Put the checkout and its `examples/` directory on the import path. Run the adapter
against the original source to record the reference, then against the candidate source
with the same checkpoint and workload:

```bash
export PYTHONPATH="/path/to/WAMJET:/path/to/WAMJET/examples${PYTHONPATH:+:$PYTHONPATH}"
TAG=baseline REF=/path/to/campaign/reference REF_MODE=record \
  OUT_DIR=/path/to/campaign/results python /path/to/policy_adapter.py
TAG=candidate REF=/path/to/campaign/reference \
  OUT_DIR=/path/to/campaign/results python /path/to/policy_adapter.py
```

Set `PROFILE=1` to collect a trace and `POWER=1` for power sampling. Both default off.
`ACTIVITIES=cuda` selects CUDA-only tracing. `WARMUP` controls the timed call series;
`PROF_CALLS` controls the optional profiler window. Build/load time, first-inference time
and total runner time are reported separately from endpoint latency.

Use `MAXIMUM_TIER=A` to require numerical tier A (default: B). Set `TIER_A_TOLERANCE=limits.json`
when recording the baseline; candidate checks reuse its saved limits. During approximate
search, `SCREENING=true MAXIMUM_TIER=C` records provisional timing and drift. Final B/C
validation omits screening and supplies `TASK_EVALUATION=/path/to/task-evaluation.json`.
Check mode requires an existing `REF`; without a reference, success measures timing only.

For task-quality screening, use the policy's own evaluator with identical tasks, initial
states and seeds in each arm. Save one JSON per variant mapping the same nonempty set
of episode keys to success booleans, then compare them:

```bash
python -m wamjet.paired baseline=baseline.json candidate=candidate.json
```

Use the endpoint benchmark for latency; a small task screen reports only the episodes
exercised. Keep search/calibration cases separate from final evaluation, and report the
sample size and raw outcomes. No detected regression does not establish equivalence.

## Development

Run focused checks for changed tools:

```bash
python scripts/test_modules.py
python scripts/test_capture.py
python scripts/test_examples.py
python scripts/test_equivalence.py
python scripts/test_preflight.py
python scripts/check_skill.py
python scripts/check_skill.py .claude/skills
```

No GPU campaign is required for routine guidance edits. Use subsequent runs to improve
or delete instructions, and controlled comparisons when claiming better skill performance.
