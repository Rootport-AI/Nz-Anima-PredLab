# Agent Handoff

This repository is a Forge Neo extension named `Nz-Anima-PredLab`.

Use this file as the short orientation. The detailed source of truth is `docs/nz-anima-predlab-spec.md`.

## Current State

- Package: `nz_anima_predlab`
- Forge entrypoint: `scripts/nz_anima_predlab.py`
- Version: `0.1.1`
- UI prefix / setting keys: `nzap-*` / `nzap_*`
- Console prefix: `[Nz-Anima-PredLab]`

Implemented runtime patches:

- `block_structure_trace`
- `block_forward_identity`
- `attention_kernel`
- `sparse_attention`
- `teacache`

Implemented but not automatically used by the normal UI flow:

- `cond_batch_trace`

Scaffolded / logged but not yet real optimization patches:

- Cond/uncond optimization
- Low-bit experiment
- Torch compile experiment

## TeaCache Status

TeaCache is working in Forge Neo by patching `backend.nn.anima.Anima.forward` when `_forward` is absent. A successful run shows:

```text
teacache_summary=model_calls=32 ... skips=N ... fallbacks=0 errors=0 active=True
```

The first model call must always be full calculation. Do not allow cache use when `previous_residual` is missing.

Forge Neo passes unused kwargs such as `control` into `Anima.forward`; TeaCache should ignore unused kwargs and consume only the values it needs, especially `transformer_options`.

## Important Rules

- Before adding a feature, decide whether it is truly mutually exclusive with existing features.
- If a combination can cause server crashes, CUDA errors, tensor shape corruption, or unrecoverable patch conflicts, prevent simultaneous Enable.
- If a combination is merely not recommended because images may degrade or speed may decrease, it may remain simultaneously enableable with clear logs or UI messaging.
- Preserve baseline behavior when the parent Enable is off.
- Restore monkey patches on disable, unsupported model, unload, and degraded paths.
- Do not silently fail. Log degraded or fallback reasons with the `[Nz-Anima-PredLab]` prefix.

## Useful Files

- `nz_anima_predlab/script.py`: Gradio UI and generation-time patch selection.
- `nz_anima_predlab/state.py`: settings snapshot and runtime counters.
- `nz_anima_predlab/patcher.py`: monkey patch implementation and restore logic.
- `nz_anima_predlab/diagnostics.py`: console snapshots and summaries.
- `docs/nz-anima-predlab-spec.md`: detailed behavior and acceptance criteria.
- `docs/forge-neo-research.md`: Forge Neo investigation notes.

## Forge Neo Gotcha

Forge Neo can preserve old Gradio component ranges/defaults in `ui-config.json`. If a UI change does not appear after reinstalling the extension, check that file and restart Forge Neo.
