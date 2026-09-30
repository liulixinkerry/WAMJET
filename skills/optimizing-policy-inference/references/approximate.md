# Approximate methods

Consult this reference when approximation is permitted and a measured bottleneck makes
it promising. Preserve the best working revision and compare outputs to the original
baseline. Exploration can continue before task evaluation is available.

## Focused checks

| Method | Check the changed behavior |
|---|---|
| Approximate caching | Actual reuse, refresh behavior, changed inputs and reset |
| Pruning / token merging | Token reduction, position/mask alignment, restoration and output shapes |
| Sparsity | Executed sparse kernels, selection overhead and comparison to the intended masked reference |
| Quantization | Executed low-precision path, scales/calibration and applicable quantizer or GEMM reference |
| KV-cache compression | Read-back error, position/slot alignment and unchanged eviction policy |

Use relevant tests from the implementation actually used, or a focused check against
its reference. An import, skipped test or speed microbenchmark does not establish
correctness. Broad author suites and component ablations are optional. A passing
implementation test does not establish preserved WAM task quality.

## Quantization and latency

Profile the full model before and after quantization. Report a timing breakdown by
major stage/module and operation: GEMMs, attention, activation scaling/casts, layout
conversions, transfers and CPU/launch overhead. Identify the remaining bottleneck;
confirm actual low-precision kernels. Use warmed, unprofiled endpoint timing for
speedups; account for overlap instead of summing kernel durations as wall time.
Report compilation and one-time weight preparation separately.

Try `torch.compile` on the model or supported hot regions, plus lossless fusion,
CUDA graphs and removal of redundant copies/synchronization where profiling motivates
them. Check numerical/state behavior. Give the unquantized comparison the same
applicable optimizations to isolate quantization gains; retain the original reference.

Choose recipes supported by the installed backend and GPU:

- **FP8:** E4M3, static per-output-channel weight scales, dynamic per-token activation
  scales recomputed each invocation.
- **Blackwell MXFP8:** try torchao's MXFP8 inference path: 32-element blocks with
  E8M0 scales and dynamic activation scaling.
- **Blackwell NVFP4:** try Transformer Engine: E2M1 values, E4M3 block scales and a
  separate global FP32 scale per tensor. Prefer `1x16` blocks along the reduction
  dimension for weights/activations when supported. TE's `16x16` weight / `1x16`
  activation training recipe is another candidate, not an assumed inference optimum.

Avoid static activation scales for WAM inference: fixed activation ranges can cause
large quantization errors as inputs and denoising steps change. Use dynamic activation
scales recomputed from the current activation tensor on each invocation; static weight
scales are appropriate. For NVFP4 activation quantization, a fixed global scale can
coexist with dynamic block scales.
Record layers, backend, scale formats/update policy and calibration.

Sources: [FP8 PTQ](https://developer.nvidia.com/blog/?p=104049),
[torchao on Blackwell](https://pytorch.org/blog/faster-diffusion-on-blackwell-mxfp8-and-nvfp4-with-diffusers-and-torchao/),
[Transformer Engine NVFP4](https://nvidia.github.io/TransformerEngine/examples/fp8_primer.html#nvfp4-format).

## Final task quality

For a promising finalist, use the policy's own task evaluator with baseline and candidate
on identical nonempty task/episode keys, initial states and seeds. Keep workload and
success criteria fixed. Choose the tolerated quality change and sample budget before
examining candidate scores. Keep calibration/search cases separate from final evaluation.

`wamjet.paired.pair` summarizes paired outcomes and detectable regression. Report sample
size and raw scores; no detected significance does not establish equivalence. If using
the numerical acceptance API for tier C, pass the actual candidate's task decision as
`task_evaluation`, including `passed`, the criterion, revisions and result paths.

Without applicable task evaluation, report latency and output drift with task quality
unvalidated. Continue useful experiments within budget; reject known quality failures.
Give comparison arms equal method/source access and evaluation budgets. Blind runs use
only the coordinator's supplied references and assets.
