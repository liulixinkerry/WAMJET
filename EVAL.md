# Optional skill evaluation

Ordinary WAM optimization reuses prior work and external memory. This protocol is for
testing whether guidance helps an agent, or comparing two versions of that guidance.
It is not a prerequisite for every skill edit or optimization experiment.

## A fair comparison

1. Choose the same policy revision, checkpoint, workload, model, resources and budget
   for both arms. Establish runnable setup and original output references before search;
   share necessary compatibility repairs equally.
2. Give fresh agents either the old/new guidance or guidance/no guidance. Hold guidance
   fixed during the run. Keep the arms independent and avoid mid-run optimization hints.
3. For blind runs, the coordinator builds guided bundles with
   `python scripts/build_guided_bundle.py <new-destination>` and checks exit code zero.
   Keep answer keys, build logs, policy-specific findings, prior runs and commit history
   outside worker access. Provide fresh policy snapshots and identical permitted generic
   tools, method references and task-evaluation assets. Report any missing answer keys;
   a successful build without a key does not establish that policy's redaction coverage.
4. Compare reproducible endpoint speedup, output/task quality and time spent. A short
   record of useful changes, failed attempts and the stopping reason is enough. Do not
   claim that a bounded search exhausted all possible optimizations.
5. Report run counts and variability. One comparison is preliminary; repeat when the
   size or importance of a claimed improvement warrants it. Validate final results
   against the original workload, independently of the search agent's acceptance label.

Keep calibration/search cases separate from final task-evaluation cases. A lower latency
with failed task quality is not an accepted acceleration under a preserved-quality goal.

## Improving the skill

After ordinary work, identify a useful lesson and make a small edit or deletion in the
relevant instruction. Store deployment facts and model-specific findings in external
memory, distinguishing confirmed fixes from untested hypotheses. Reuse them on the next
run. Consolidate repeated instructions and remove advice that no longer helps.

Use a controlled comparison when assessing a general claim about skill effectiveness.
Routine cleanup needs focused checks and feedback from subsequent use, not another
full campaign. During a comparison, collect lessons and apply them after both arms end.
