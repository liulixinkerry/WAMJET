---
name: optimizing-policy-inference
description: Reduces GPU inference latency for robot policies, VLAs, diffusion policies, and world-action models through iterative measurement and code changes. Use when accelerating policy inference, compilation, CUDA graphs, memory traffic, or precision. Not for training throughput or multi-GPU scaling.
---

# Optimizing policy inference

Make the public inference endpoint faster within the user's budget. Explore freely
within the permitted workload and quality limits; keep checks proportional to the change.

## The loop

1. **Get one real inference call working.** Reuse the working environment and prior
   setup notes. Trace the selected endpoint's imports; repair required APIs and defer
   unused dependencies. Use `references/setup.md` if needed. Keep compatibility repairs
   common to the baseline and candidate. Reuse one allocation, physically staged local
   checkpoints and compatible compiler caches. Optimize loading when it limits iteration.
2. **Save a baseline.** Record the revision, command, checkpoint and full input/config
   identity, original outputs, and warmed repeated endpoint latency. Keep inputs, seeds,
   horizon, schedule and output contract fixed. Measure per-output run-to-run noise when
   needed to judge numerical drift. Collect environment details once per process.
3. **Find a bottleneck and edit.** Inspect the hot path; profile when it will answer a
   specific question. Try a promising mechanism, then measure again on the same GPU.
   Compilation, unnecessary synchronization, repeated work, memory traffic and permitted
   approximation are hypotheses to prioritize from the actual workload.
4. **Keep or revert.** Check output keys, shapes, dtypes, finite values and drift against
   the original reference. For state/cache/graph changes, also exercise changed inputs
   and reset. Retain a working fallback. A small, correct edit may stay with neutral
   latency if it reduces measured sync/copy work, simplifies the pipeline or enables a
   concrete next experiment. Record "no measured speedup"; defer speculative complexity
   and revert regressions. Keep one short record: revision/command, change, latency,
   output drift and decision. Missing task evaluation does not stop exploration.
   With the example runner, use `SCREENING=true`. Numerical drift alone does not change
   the method's tier. Allow screening to continue and classify the method separately
   from numerical comparison results. Final acceptance checks belong to the finalist run.
5. **Repeat, then validate the best result.** Reserve time to remeasure the original
   baseline and best candidate under comparable conditions. Run focused checks for the
   changed behavior. Report the reproducible speedup, quality status, command/revision
   and remaining opportunities. Separate startup savings from endpoint speedup.

Use the optimization budget to keep investigating and trying changes. A speedup, reached
latency target, failed trial or short plateau is a progress update. When ideas run out,
inspect the remaining hot path and refresh the diagnosis to find the next experiment.
Finish on user request, budget exhaustion or an external blocker that prevents further
work, such as unavailable required access or hardware. A failed check applies to that
candidate or measurement; continue other approaches.
In-scope experiments need no further approval or completion certificate.

## Correctness

Classify the method as well as its measured outputs:

- **A — lossless:** preserve the mathematical computation, workload, precision and
  state semantics. Ordinary floating-point drift from equivalent compilation, fusion,
  GEMM or reduction ordering is acceptable; bit identity is optional. Record per-output
  `max_abs` and `rel_l2` as diagnostics, not automatic rejection thresholds. Do not reject
  a mathematically equivalent candidate solely because drift exceeds an earlier limit.
  Judge eligibility through mathematical/code review and focused checks of output keys,
  shapes, dtypes, finiteness, changed inputs, reset and replay. Investigate unexpected
  drift for implementation errors. Quantization, pruning and approximate caching are
  not lossless.
- **B — quality-preserving approximation:** approximate internal computation while
  keeping steps, schedule, horizon and public contract fixed. Require paired task
  evaluation against a declared acceptable quality-loss bound.
- **C — changed budget or unvalidated quality:** fewer steps, early exit, dropped
  frames/tokens, changed horizon/schedule, or approximations lacking passing task evidence.

Output agreement alone cannot establish mathematical equivalence. Review the change;
quantization, pruning and approximate caches still need task evidence. Keep the output
and state checks in the loop above. Reject invalid outputs, stale state and known quality
failures; use focused tests, with broad suites and ablations optional.

Rebaseline timing after changing allocation or stack. A changed stack needs an original
build's reference on that stack. Keep workloads, permitted access and budgets equal
when comparing methods. Profiler traces diagnose work; ordinary endpoint timing judges
speedups. Inflated traces cannot establish scheduling gaps or elapsed GPU busy time.

## Learn from the run

Read the supplied external memory during ordinary optimization. After a run, save useful
environment fixes and reusable findings there, separating confirmed results from ideas.
For a general lesson, edit or delete the relevant skill instruction; consolidate repeats.
Prefer a small correction over another rule. Try the revised guidance during subsequent
work. Formal skill comparisons are optional and described in the repository's `EVAL.md`.
During a blind evaluation, use only supplied material and let the coordinator update
memory and guidance after both arms finish.

Read references only when useful: `references/detection.md` for diagnostic recipes,
`references/setup.md` for setup failures, and `references/approximate.md` for approximate
methods. Tools are optional: `wamjet.capture.measure` for timing, `capture` for traces,
`pipeline`/`attribute`/`fusion` for diagnosis, `equivalence` for outputs, `paired` for tasks.
After editing the skill in the full repository, run `python scripts/check_skill.py`.
