# Inference environment preparation

Prepare the selected inference entry point before loading large checkpoints or
starting optimization trials. A package used by another server, training path or
legacy adapter is not automatically a requirement of this endpoint.

## Short setup pass

1. Record the interpreter, source revision and selected load/infer entry point.
   Trace that entry point's imports and configuration-selected modules. Distinguish
   unused imports, annotation-only imports, wrapper containers and actual model kernels.
2. Probe the exact modules/symbols with the assigned interpreter. Use CPU-safe targets
   before allocation; do GPU-dependent imports/checks inside the existing allocation.

   ```bash
   python -m wamjet.preflight --import-target package.module:RequiredClass \
     --import-timeout 30 --json import-check.json
   ```

   Replace the target with the selected entry point or required dependency; repeat
   `--import-target` for additional targets. This path probes only the listed imports,
   with a fresh process and timeout per target. It skips broad GPU/library discovery
   and saves actual tracebacks. Success establishes importability, not runnable inference.
3. Fix the first real cause. An installed package with a removed symbol needs an API
   repair or removal of the unnecessary dependency, not repeated installation attempts.
   Defer training/simulator imports to their callers. A wrapper used only as a container
   may be replaceable: preserve nested values, attribute access, shape/dtype handling
   and input/output transforms, then exercise those operations with a small check.
   Do not fabricate no-op implementations of behavior the endpoint actually needs.
4. Check checkpoint/config/tokenizer assets, resolved paths and scratch/RAM/VRAM needs.
   Use the authorized interpreter and dependency policy. If installs are forbidden,
   make source repairs or use already available alternatives; do not mutate the shared
   environment or change model workload to escape an error.
5. Inside one allocation, verify the selected backend on the actual GPU and inspect
   CUDA/library search paths before loading weights. Stage once, load and run one real
   endpoint call. Imports passing does not prove the kernel or checkpoint is usable.
6. Save the successful command, relevant environment variables, dependency versions,
   source repairs and unresolved items. Reuse those settings across trials; repeat only
   affected checks when code or stack changes. Keep preparation time separate from
   checkpoint-loading and steady-state timing, and use common repairs in both arms.

## When setup stalls

Preserve the first traceback and distinguish an absent package from an API mismatch,
unsupported GPU backend, linker failure, missing asset or import timeout. Test one
proposed repair at a time; do not repeat the same failed launch without a changed cause.
An import timeout is unresolved, not evidence that the package is absent.

After a failed approach, try an in-scope alternative: defer the unused dependency,
simplify the wrapper, repair the needed interface or choose an equivalent installed
backend. There is no separate setup retry quota; use the campaign budget and test a
changed cause on each retry. Setup failure does not reject an optimization hypothesis.
If access or resources block this path, continue independent work. In blind comparisons,
share permitted environment facts equally; keep prior optimization findings isolated.
