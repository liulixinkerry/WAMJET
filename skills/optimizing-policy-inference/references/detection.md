# Detection recipes

## Contents

- Locating synchronizations by source line
- Synchronization pattern catalogue
- Finding graph breaks and counting recompiles
- Finding ops the compiler declined
- Persisting the compile cache
- Writing to a persistent buffer without a functionalization copy
- Overlapping observation upload
- Capturing CUDA graphs safely
- Bucketing shapes instead of enabling dynamic shapes
- Finding host-resident constants
- Profiling startup by phase
- Telling a real cast from a no-op `.to()`
- Data-dependent control flow that must stay compiled

## Locating synchronizations by source line

Counting synchronizations is straightforward; locating them is what permits a fix.
PyTorch raises the warning from C++, so the message names an internal source file and
identifies nothing actionable. The `warnings` record carries the Python frame that
triggered it, and that frame is the line to change.

```python
import warnings, torch
torch.cuda.set_sync_debug_mode("warn")
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    infer()
    torch.cuda.synchronize()
torch.cuda.set_sync_debug_mode("default")
for c in caught:
    if "synchron" in str(c.message).lower():
        print(f"{c.filename}:{c.lineno}")
```

Use `"error"` instead of `"warn"` to fail on the first synchronization once the hot path
is expected to be clean.

## Synchronization pattern catalogue

| pattern | why it synchronizes | fix |
|---|---|---|
| `tensor.item()`, `.tolist()`, `bool(t)`, `if t > 0` | reads a device value into Python | keep the comparison on device |
| `for i, v in enumerate(cuda_tensor)` | iterating a device tensor reads it to the host | compare positions against a reduced value computed on device |
| `x.to(device)` on tokenizer or preprocessing output | host-resident tensor crossing over | cache the result when the input is constant across a rollout |
| `torch.randn(device="cpu").to(device)` | host-sampled values staged through pageable memory | sample on device, or into a pinned buffer |
| `result.to(device="cpu")` | the result readback | preallocated pinned destination with `non_blocking=True`; synchronize only when the host reads the values |
| compiler-generated `buf.copy_(arg, False)` | a *compiled graph input* is host-resident | construct that tensor on the device; this one appears in generated code, not source |
| data-dependent shapes: `nonzero`, `masked_select`, `unique` | output shape depends on device data | restructure to a fixed shape where possible |

Two rules decide most cases:

**A transfer is not a synchronization; pageable host memory is.** CUDA cannot DMA to or
from unpinned memory, so the driver stages through a bounce buffer and blocks. The same
transfer through pinned memory is an asynchronous DMA.

**Transfer size does not predict cost.** A scalar read stalls the pipeline as thoroughly
as a large tensor. Never triage synchronizations by byte count.

## Finding graph breaks and counting recompiles

```python
model = torch.compile(model, fullgraph=True)     # a break raises instead of falling back
```

```bash
TORCH_LOGS=graph_breaks,recompiles python run.py
```

```python
from torch._dynamo.utils import counters, compile_times
print(sum(counters["graph_break"].values()))     # zero if fullgraph compiled
print(counters["stats"])                          # unique_graphs, calls_captured
print(compile_times())
```

A graph count that stays bounded across steady-state calls indicates shape specialization
is working. A count that keeps climbing indicates an unbounded shape source.

## Persisting the compile cache

```python
artifacts = torch.compiler.save_cache_artifacts()   # after warmup
torch.compiler.load_cache_artifacts(artifacts)      # at startup, on every rank
```

One shared cache file accumulates artifacts across precisions and then saturates; a
second file gains nothing. Autotuners that benchmark at launch are not captured by this
cache, so their cost repeats on every process start.

## Writing to a persistent buffer without a functionalization copy

```python
self.register_buffer("cache", torch.empty(...))
torch._dynamo.mark_static_address(self.cache)
self.cache.index_copy_(1, idx, new)      # not self.cache[:, a:b] = new
```

Slice assignment on a graph input causes functionalization to clone the entire buffer
regardless of how much is written. Measure before adopting a persistent buffer: it saves
only the write of the invariant prefix, while the read remains, because attention
consumes the whole window however it was built.

## Overlapping observation upload

Before hiding the copy, check whether it needs to exist at all: an unchanged frame
across cycles, a source that could write directly into a pinned or device buffer, or a
value already resident on the device from an earlier step can remove the transfer
outright. Overlap only the copy that survives that check.

```python
staging = torch.empty(shape, pin_memory=True)     # allocate once
staging.copy_(frame)                              # host side
gpu = staging.to("cuda", non_blocking=True)       # overlaps prior compute
```

Double-buffer the staging tensor so one control tick uploads while the previous computes.

Verify the overlap happened rather than inferring it from the code shape: check that
the H2D copy's trace span sits inside the previous call's compute span, not after it.
A wall-time improvement with no matching trace change is not evidence of real overlap.

## Capturing CUDA graphs safely

```python
torch.compiler.cudagraph_mark_step_begin()   # at inference entry, retires prior outputs
```

Call this at every entry point that replays a captured graph. When a subclass overrides
the entry point and delegates elsewhere, the guard is inherited around rather than
inherited, and the second call raises on overwritten graph outputs.

Graph capture pays only when marshalling inputs into static buffers is cheaper than the
launch gap it removes. Compare device-event counts before and after: the copies added
are visible as device-to-device memcpys.

## Finding ops the compiler declined

A graph break is loud: `fullgraph=True` raises. A backend refusal can be silent: the
region remains while an unsupported operation runs through another kernel path.

`wamjet.pipeline` reports possible fallbacks: kernels launched from inside a compiled
region whose names are not recognized as compiler-generated or vendor kernels. Naming is
a screening heuristic. Confirm a refusal from generated code or the compiler log before
changing the model.

To get the reason rather than the symptom, make the backend say why:

```bash
TORCH_LOGS="+inductor,graph_breaks,recompiles" python bench.py 2>&1 | grep -iE \
  "fallback|unsupported|does not support|cannot|skipping"
```

Common refusals, in rough order of frequency:

| refusal | usual cause | fix |
|---|---|---|
| complex dtypes | rotary embeddings built through `polar`/`view_as_complex` | re-express as real-valued sin/cos pairs |
| custom operators | an op registered without a lowering or a `Fake` kernel | register one, or decompose the op |
| data-dependent shapes | `nonzero`, `masked_select` inside the region | restructure to a fixed shape |
| unsupported reductions | an uncommon dtype/axis combination | cast to a supported dtype around the reduction |

The dtype refusals are the ones worth hunting: they usually sit in code written once, for
numerical comfort rather than necessity, and the value is real work running at eager speed
in the middle of a region everyone believes is compiled.

## Bucketing shapes instead of enabling dynamic shapes

Pad variable-length inputs to the next of a small fixed set of lengths, pre-warm every
bucket at startup, and keep `dynamic=False`. This trades a little padded compute for
static kernels and no runtime recompilation.

## Finding host-resident constants

A constant that lives on the host is copied into every compiled region that reads it,
once per call, and each copy is a synchronization if the memory is pageable. The cost is
independent of the constant's size.

Search for the three patterns:

```bash
# 1. tensors constructed with no device
grep -rnE "torch\.(tensor|as_tensor|arange|linspace|zeros|ones)\(" src/ | grep -v "device="

# 2. modules holding constants but registering none
grep -rc "register_buffer" src/model/*.py     # zero is a red flag if constants exist

# 3. derived constants inside Python containers -- the hardest to spot
grep -rnE "self\.[a-z_]+ *= *[\[\{].*self\." src/
```

The third pattern is the subtle one, and registering the elements does not fix it. A list
or dict is not a tensor, so `nn.Module.to()` never touches it, and `nn.Module._apply`
replaces entries in `_buffers` — leaving a container built at construction pointing at the
original host tensors permanently.

Replace the container with a tensor rather than keeping it in sync:

```python
self.register_buffer(
    "scale", torch.stack([torch.as_tensor(mean), 1.0 / torch.as_tensor(std)]),
    persistent=False,
)
```

Indexing gives the same values, so `scale[0]` / `scale[1]` and any
`isinstance(..., torch.Tensor)` branch keep working. Confirm the distinction:

```python
m = m.cuda()
[t.device.type for t in m.scale]      # stacked buffer -> ['cuda', 'cuda']
[t.device.type for t in m._stored]    # stored list    -> ['cpu', 'cpu']
```

Confirm from the compiler's own output: a `buf.copy_(arg, False)` appearing in generated
code means a graph input is host-resident. The buffer's width and dtype identify which
constant it is — match them against the model's channel counts, head dimensions, or
schedule lengths.

Verify the fix by re-running the synchronization probe; these copies should disappear
from the count entirely, not merely shrink.

## Telling a real cast from a no-op `.to()`

A per-step `.to(device=..., dtype=...)` inside a hot loop looks like waste and often is
not. `Tensor.to()` returns `self` when the device and dtype already match, so the call
launches no kernel and allocates nothing. Defensive conversions in a scheduler or a step
function are usually free, and removing them makes the function fragile for callers that
pass a different dtype.

Decide by pointer identity rather than by reading the call:

```python
out = t.unsqueeze(0).to(dtype=sample.dtype, device=sample.device)
out.untyped_storage().data_ptr() == t.untyped_storage().data_ptr()   # True -> no-op
```

Or profile a loop of the calls and count kernels; a no-op contributes none.

**The condition to check is upstream.** These calls are free only while the producer emits
the consumer's dtype. If a schedule, a cache, or a preprocessing step ever changes
precision, the same lines silently become one cast kernel per step and nothing reports it.
When a conversion sits in a loop, verify where its input dtype is set and treat that as the
invariant, rather than deleting the conversion.

## Data-dependent control flow that must stay compiled

Every pattern in the synchronization catalogue above assumes the branch can be
restructured away. Some cannot: an early exit on a convergence test, an adaptive step
count, a branch on a predicate the device computes. The usual escape hatch is to read the
predicate to the host with `.item()`, which synchronizes.

Under `fullgraph=True` a Python branch on a device value is a hard error:

```
Unsupported: Data-dependent branching
```

leaving only two bad options — break the graph, or synchronize. `torch.cond` and
`torch.while_loop` are the third: they express the branch as a traceable operator, so the
region compiles as one graph.

```python
# fails under fullgraph=True, and synchronizes in eager
if x.sum() > 0:
    y = f(x)
else:
    y = g(x)

# traces into the graph; both branches must return matching shapes and dtypes
y = torch.cond(x.sum() > 0, f, g, (x,))
```

**Stronger option: device-side branching inside a captured graph.** CUDA graph
*conditional nodes* evaluate the predicate on the device during replay, so no host round
trip occurs at all. PyTorch exposes them on `torch.cuda.CUDAGraph`:

```python
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    pred = (x.sum() > 0)                                    # scalar CUDA bool tensor
    cur = torch.cuda.CUDAGraph.get_currently_capturing_graph()
    out.copy_(x)                                            # always runs
    cur.begin_capture_to_if_node(pred)
    out.mul_(10)                                            # runs only when pred is true
    cur.end_capture_to_conditional_node()

g.replay()      # branch decided on the device
```

Measured against the equivalent Python branch: **0 synchronizations per replay versus 1**,
with the correct path taken in both directions. Requires a driver exposing
`cuGraphConditionalHandleCreate`, static input and output buffers as with any captured
graph, and the predicate as a scalar CUDA tensor.

`torch.cond` remains the right tool when the region is compiled through `torch.compile`
rather than captured by hand; it keeps the region whole where a Python branch is a hard
error. Measure its synchronization effect separately rather than assuming one.

**Order of preference. Restructure first; reach for machinery only when the data
dependency is genuine.**

1. **Remove the branch.** Most data-dependent control flow in a policy is not genuinely
   data-dependent. Check first:
   - Does the predicate come from configuration rather than from data? Then it is static —
     resolve it once at setup, outside the hot path.
   - Is the branch loop-invariant? Hoist it above the loop.
   - Does each side write the same shape? Compute both and select with `torch.where`; two
     cheap paths usually cost less than any branching mechanism.
   - Can the condition be folded into arithmetic? A mask multiply replaces a branch, which
     is how a padding loop becomes a positional comparison.
2. **`torch.where`** for an elementwise choice — no control flow at all.
3. **`torch.cond` / `torch.while_loop`** when the region is compiled by `torch.compile`
   and the dependency is real. Keeps the region whole where a Python branch is a hard
   error; measure any synchronization effect separately.
4. **CUDA graph conditional nodes**, last. They remove the host round trip outright, but
   they require hand-captured graphs, static input and output buffers, driver support, and
   a scalar device predicate. That is a large amount of fragile machinery to carry for one
   branch.

The ordering matters because each step down adds complexity that must be maintained
forever, while the step above deletes the problem. A branch that can be restructured away
costs nothing to keep removed; a conditional node costs capture discipline on every future
change to that region.

## Profiling startup by phase

Startup does not change inference latency, but it sets how many experiments are possible
per day. Measure it the same way as the hot path: by phase, before changing anything.

```python
from wamjet.startup import Phase, residency, warm

warm(*weight_files)                     # or measure cold -- state which
p = Phase()
cfg   = p("config",               build_config)
model = p("construct + base wts", construct)
_     = p("task checkpoint",      lambda: model.load_checkpoint(path))
p.report(cache_state=residency(*weight_files)["state"])
```

`Phase` synchronizes before starting each timer, so CUDA context creation lands outside
the first phase rather than inside it. `report` refuses to print a comparable-looking
total when cache state was not recorded.

Then split the dominant phase again. To time a component loader without editing the
library, wrap it — **with `functools.wraps`**, or a signature-introspecting validator will
see the wrapper instead of the original and fail in a way that looks unrelated:

```python
import functools
orig = lib._load_component
@functools.wraps(orig)
def timed(*a, **kw):
    t0 = time.perf_counter(); r = orig(*a, **kw)
    print(f"  {a[1]:20s} {time.perf_counter()-t0:6.1f}s"); return r
lib._load_component = timed
```

What this typically finds, in rough order of size:

| phase | usual cause | fix |
|---|---|---|
| model construction | parameters initialized on the host, then moved, then overwritten | `with torch.device(device):` — see below |
| per-component weight loads | each component read to host, then moved | load with `device=`, or mmap plus `assign=True` |
| task checkpoint | one large read | often already small relative to the rest — measure before optimizing |
| compilation | cache miss | shared-storage cache keyed by stack, not host |

**Constructing on the device.** Wrap construction so parameters are created where they
will live:

```python
with torch.device(device):
    model = BigModel(**config)
```

A meta device is stronger still — it skips the initialization instead of relocating it,
so build time stops scaling with parameter count — but it is only safe when the
checkpoint covers every tensor the module declares. Anything it misses stays
unmaterialized and fails at first use, with an error that points at the forward pass
rather than at the loader.

**Verify coverage against what the module declares, not against what it serializes.**
The obvious checks — `missing_keys` from `load_state_dict`, or a diff against
`state_dict().keys()` — cannot see the most common failure. A buffer registered
`persistent=False` is excluded from `state_dict()` by definition: it is absent from the
checkpoint by construction, so it never appears in `missing_keys`, and a module full of
them reports perfect coverage. Such buffers routinely hold real constants — normalization
statistics, scale factors, index tables — computed in `__init__`. Under a meta device
those computations produce meta tensors and the constants are lost.

```python
declared = {n for n, _ in chain(model.named_parameters(), model.named_buffers())}
uncovered = declared - set(checkpoint)      # non-persistent buffers appear here,
                                            # and in `missing_keys` they never do
```

Prefer the device context when `uncovered` is non-empty or unchecked. It keeps the
throwaway initialization, so it is slower than meta, but every buffer computed in
`__init__` still holds its value. Reach for meta only with the check above passing, and
re-run it after any change to the module — registering a buffer is a common consequence
of unrelated optimizations, and it revokes meta safety silently.

**Compile cache placement.** Set `TORCHINDUCTOR_CACHE_DIR` to shared storage. A
node-local path, or one keyed by hostname, is discarded whenever a scheduler places the
job elsewhere and repays every compile. Hostname is the wrong key: compiled output is
invalidated by the framework, toolkit and architecture combination, so two identical
nodes are guaranteed a miss where a hit was available.

**Confounds to control.** Page-cache state can change load timings several fold, so
compare cold-to-cold or warm-to-warm and state which. Measure it rather than assuming it:
`residency()` reports per-file residency through `mincore`, which does not read the file,
because reading it to find out would populate the cache being measured. Dropping the cache
needs privilege a scheduled job usually lacks, so warming both arms is the practical
control.

Cache state is not the only confound, and controlling it is not sufficient. A build
competing with other tenants for host memory bandwidth can take **several times** its
usual wall clock with the cache fully warm and correctly reported, and nothing in the
phase breakdown distinguishes that from a real regression — the slowdown lands on the
component loads, exactly where a real change would land. Repeat the build before
attributing any difference to a change, and discard an arm whose components move together
by a large factor, which is contention rather than anything in the diff. A saving that is compute rather
than I/O is constant in absolute terms, so quote it as a duration, not only as a share of
a build whose total depends on cache warmth.

**Separating contention from a real regression, at steady state as well as at startup.**
The same confound shows up per-call, not only during a build: fresh processes of
identical code can split into two latency populations with no code difference between
them. Sample power draw alongside the timing before hypothesizing a code cause —
`nvidia-smi --query-gpu=power.draw,clocks.sm --format=csv -lms 100`, or `pynvml`
in-process. A run sharing the node's memory bandwidth with a co-tenant holds the same SM
clock but pulls less power while running slower: it is doing less work per unit time,
not entering a different branch. A thermal or power-limit cause looks different — the
clock itself drops. Power separates the two populations more reliably than latency
alone, because the latency gap can sit inside ordinary run-to-run spread while the power
gap does not. Confirm a suspected co-tenant against the scheduler or `nvidia-smi`'s
process list before accepting the correlation as the mechanism.
