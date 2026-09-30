---
name: wamjet-launch
description: Prepares WAMJET optimization prompts with reusable environments and GPU allocations. Use when asked to set up or launch a policy optimization campaign or fill its template. Blind skill evaluation is optional. Not for training jobs.
---

# WAMJET launch

Fill `template.md` using the current request and established session settings.
The optimization skill owns the experiment loop; this launcher supplies deployment facts.

1. Resolve `{{WAMJET_ROOT}}` from this file's repository. For ordinary optimization,
   read external memory only when the user explicitly supplies its path.
   Keep memory outside WAMJET. Reuse relevant setup fixes and prior findings;
   verify facts affected by a changed environment. Use this run's selected WAMJET
   guidance even when reusing a policy workspace; old campaign harness copies are
   historical snapshots. Refresh worker instructions when resuming with revised guidance.
2. Fill resource settings from the conversation; ask only for missing required values:
   `{{PYTHON_INTERPRETER}}`, `{{CHECKPOINT_ROOT}}`, `{{SCHEDULER}}`,
   `{{GPU_CONSTRAINT}}` (including requested RAM, disk and CPUs), `{{TIME_LIMIT}}`
   and `{{SCRATCH_DIR}}`. State both the total campaign budget and allocation limit;
   use the supplied time limit for both unless the user specifies otherwise. Honor
   established workspace and source-access limits. Scheduler-provided scratch variables
   resolve inside the job. Stage physical copies there, including supporting model assets.
3. Identify model sources and starting revisions. Reuse a requested existing checkout
   in a separate branch/worktree as appropriate. Honor explicit commit hashes. Resolve
   a branch live only when no commit is pinned; record the actual starting revision.
   For an unknown model with no supplied checkout or remote, ask for its source.

   | policy | upstream remote |
   |---|---|
   | FastWAM | `https://github.com/yuantianyuan01/FastWAM` |
   | DreamZero | `https://github.com/dreamzero0/dreamzero` |
   | Cosmos-Policy | `https://github.com/NVlabs/cosmos-policy.git` |
   | LingBot-VA | `https://github.com/Robbyant/lingbot-va` |

4. Fill `{{MODEL_LIST}}` with each variant, source/workspace and pinned revision.
   Fill `{{RUN_CONTEXT}}` with workspace paths, external memory path, relevant launch
   fixes and any source-access restrictions. Ordinary optimization may reuse prior
   work and documentation. Keep all model trials under the allocation's job payload.
   If the user explicitly requests a blind evaluation, follow the optional procedure
   below instead of giving workers access to prior findings.
5. Return the filled prompt when asked to prepare one. If launching is already
   authorized, execute it without requesting confirmation again. Review early worker
   returns and assign the next experiment while budget remains and work can proceed.
   Update external memory with useful findings after the run, and simplify general skill instructions when a
   lesson warrants it. No GPU jobs are needed merely to edit or review this skill.

## Optional blind evaluation

Use `EVAL.md` in the repository for comparisons of the skill itself. The coordinator
builds a redacted bundle with `scripts/build_guided_bundle.py <new-destination>` and
requires exit code zero before handing it to a worker. Keep answer keys, builder output,
prior optimization results and full WAMJET history outside worker access. Report missing
answer-key coverage to the user. Give workers fresh policy snapshots at the same pinned
revision in their campaign workspaces, and set `{{WAMJET_ROOT}}` in each worker's
prompt to its bundle. Supply only curated environment facts equally to both arms;
workers must not read external memory, prior runs or target-policy optimization recipes.
Hold guidance fixed during the comparison; apply lessons after both arms finish.

After editing, run `python scripts/check_skill.py .claude/skills` from the repository.
