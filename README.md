# Nz-fast-anima

Nz-fast-anima is a Forge Neo extension for investigating and eventually speeding up Anima / Cosmos-Predict2 T2I inference.

The current build is diagnostic only. It does not patch Forge Neo's inference pipeline yet.

## Current Features

- Forge Neo settings under `Settings > Nz-fast-anima`
- Anima / Cosmos-Predict2 model detection
- Sampling timing logs
- Attention backend trace
- CFG cond/uncond trace
- Low-bit / dtype / Forge operations trace

Logs are printed to the StabilityMatrix / Forge Neo console with the `[Nz-fast-anima]` prefix.

## Compatibility

- Target: StabilityMatrix版 Forge Neo / SD WebUI Forge Neo
- Initial target workflow: txt2img with Anima / Cosmos-Predict2 T2I models
- Not guaranteed: A1111 mainline, Forge classic, ComfyUI

## Diagnostic Modes

- `Off`
- `Diagnose only`
- `Trace attention`
- `Trace cond/uncond`
- `Trace low-bit / compile`

Enable the extension in settings, select a mode, then run a generation and inspect the console output.
