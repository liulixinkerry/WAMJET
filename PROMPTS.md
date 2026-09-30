# How to start

## Example WAMJET campaign prompts

1. Start your coding agent (e.g. Codex)

2. Replace the `<PLACEHOLDERS>` with your environment's paths and campaign ID.

3. Input the following prompt

### Lossless

```text
Launch optimization campaigns using: .claude/skills/wamjet-launch/SKILL.md and template.md beside it.

Models:
- OpenWAM-α: https://github.com/OpenWAM-Official/OpenWAM/commit/d8dd33d8576b475f5a5cdc6fb8ca902778a199b3

Resolve and record each upstream starting commit at launch.
Interpreter: <PYTHON_INTERPRETER>
Checkpoint root: <CHECKPOINT_ROOT>
Per-model allocation: exactly one GPU, 16 CPUs
Budget: 6h total campaign; 6h limit per allocation
Campaign workspace: <CAMPAIGN_WORKSPACE>
Workers: one GPT 6 Astra with xhigh reasoning per model; run concurrently when subagent capacity permits.
LIBERO root: <LIBERO_ROOT>
Campaign ID: <CAMPAIGN_ID>

Create the specified campaign workspace, using a directory name ending in "-openwam". Physically copy required checkpoints and supporting assets to local NVMe. Use the assigned interpreter and installed libraries; no installs.

Follow WAMJET's optimization skill. Repair required environment incompatibilities and defer unused optional dependencies. Preserve fair baseline comparisons, use focused checks, and keep exploring within budget after wins or failed trials. No Superpowers.

Keep useful policy changes as local commits; never push. Launch now.

Focus on lossless optimization first. Preserve mathematical correctness; reduction-order changes and their numerical drift are acceptable as lossless. Explore further inference latency improvements within budget. Report any environment issues.
```

### Approximate

```text
Use:
- Campaign ID: <NEW_CAMPAIGN_ID>
- Campaign workspace: <NEW_CAMPAIGN_WORKSPACE>
- Previous campaign workspace: <PREVIOUS_CAMPAIGN_WORKSPACE>

Launch the next WAMJET optimization round for OpenWAM-α based on the previous best validated lossless results.

Investigate further lossless optimization on GPU; reduction-order changes are lossless. After those trials, explore lossy methods such as FP8 or NVFP4 quantization. Do not reduce the DiT steps.

Read .claude/skills/wamjet-launch/SKILL.md, template.md beside it, and the previous campaign's reports and artifacts. Create the specified new campaign workspace, using a directory name ending in "-approx".

Assign one GPT 6 Astra worker with xhigh reasoning per model, concurrently as capacity permits. Use the same interpreter, checkpoint and LIBERO roots. 

Use exactly one GPU per allocation and 16 CPUs. Physically stage checkpoints and supporting assets on local NVMe. Budget: 3h limit. No installs, no pushes, and no Superpowers. Record each upstream.

Start from the latest validated local lossless commit and record that commit too. Profile it, review previous experiments, and independently search for further inference latency improvements. Keep exploring potential optimization opportunities and do not give up too easily.

Keep optimization results in your own campaign workspace! Commit useful policy changes locally; never push. Continue exploring within the campaign budget after wins or failed trials.
```
