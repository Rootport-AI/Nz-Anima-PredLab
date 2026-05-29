# Nz-Anima-PredLab

Nz-Anima-PredLab is a Forge Neo extension for observing and experimenting with Anima / Cosmos-Predict2 T2I inference.

The extension started as a diagnostic tool and now includes opt-in experimental patches. It should leave Forge Neo baseline behavior unchanged when `Enable Nz-Anima-PredLab` is off, or when all experimental controls are disabled and the attention backend is `Forge current/default`.

## Current Features

- Forge Neo AlwaysVisible panel: `Nz-Anima-PredLab`
- Anima / Cosmos-Predict2 model detection
- Sampling timing logs
- Attention backend trace and optional attention backend override
- TeaCache / residual cache experiment for Anima block skipping
- 2D sparse attention experiment scaffold
- Opt-in tensor dump probes for forecasting research data collection
- Cond/uncond, low-bit, dtype, and Forge operations diagnostics
- Runtime patch restore on disable, unsupported model, or unload

Logs are printed to the StabilityMatrix / Forge Neo console with the `[Nz-Anima-PredLab]` prefix.

## TeaCache Notes

TeaCache patches `backend.nn.anima.Anima._forward` when available, otherwise `backend.nn.anima.Anima.forward` as used by current Forge Neo. It skips the Anima block stack on selected sampling steps and reuses the previous full-calculation residual.

Successful TeaCache runs should show nonzero `model_calls` and usually nonzero `skips` in `teacache_summary`, with `fallbacks=0 errors=0` for a clean run.

## Compatibility

- Target: StabilityMatrix Forge Neo / SD WebUI Forge Neo
- Primary workflow: txt2img with Anima / Cosmos-Predict2 T2I models
- Verified environment during development: Windows, NVIDIA GPU, PyTorch CUDA build
- Not guaranteed: A1111 mainline, Forge classic, ComfyUI, multi-GPU, heavily modified pipelines

## Documentation

- [Specification](docs/nz-anima-predlab-spec.md): source of truth for UI, runtime behavior, safety rules, and patch policy.
- [Forge Neo research](docs/forge-neo-research.md): implementation notes gathered while investigating Forge Neo internals.
- [Agent handoff](AGENTS.md): short orientation for future coding agents.

## Development Reminder

Forge Neo may cache Gradio UI component settings in `ui-config.json`. If a slider range or default appears stale after reinstalling the extension, clear or update Forge Neo's UI config and restart the WebUI.
