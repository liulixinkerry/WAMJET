Optimize these models using `{{WAMJET_ROOT}}/skills/optimizing-policy-inference/SKILL.md`:

{{MODEL_LIST}}

Run one worker per model when agent capacity permits. Each model gets one reusable,
exclusive allocation: {{SCHEDULER}}, {{GPU_CONSTRAINT}}, {{TIME_LIMIT}}.
Use `{{PYTHON_INTERPRETER}}` and its installed libraries; no installs or other interpreters.
Checkpoints: `{{CHECKPOINT_ROOT}}`. Physically copy the required assets to
`{{SCRATCH_DIR}}` once per allocation and reuse them and compatible compiler caches.

{{RUN_CONTEXT}}

Follow the optimization skill for experiments, final checks and reporting. Keep useful
changes as local commits; never push. Return the baseline/final latency, quality status,
reproducible command and revision, and remaining opportunities.
