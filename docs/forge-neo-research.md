# Forge Neo implementation research for Nz-fast-anima

This document summarizes implementation notes from `Haoming02/sd-webui-forge-classic`
branch `neo`, with the goal of designing a Forge Neo extension that can diagnose and
experiment with Anima / Cosmos-Predict2 T2I acceleration.

Source repository:

- https://github.com/Haoming02/sd-webui-forge-classic/tree/neo

## Project interpretation

`Nz-fast-anima` should eventually change parts of the inference pipeline to make
Anima image generation faster. The quality target is not bit-identical output; it is
that generated images still appear visually close to the Forge Neo baseline and do
not suffer major style, face, hand, linework, hair, or background failures.

The first extension milestone should still include a conservative diagnostic mode
that does not intentionally alter model output. That mode exists to discover the
current Forge Neo execution path before enabling experimental acceleration.

## Extension loading and UI hooks

Forge Neo loads extension scripts from:

```text
extensions/<extension-name>/scripts/*.py
```

Relevant source:

- `modules/scripts.py`
- `modules/script_callbacks.py`

Important extension surfaces:

- `scripts.Script`
  - `show()` can return `scripts.AlwaysVisible` for always-on controls.
  - `ui()` can create Gradio controls whose values are passed to processing hooks.
  - `process_before_every_sampling()` is called before each sampling pass and receives
    useful objects such as `p`, `x`, `noise`, `c`, and `uc`.
  - `postprocess()` can summarize results after generation.
- `script_callbacks.on_ui_settings()`
  - Adds persistent settings with `shared.opts.add_option(shared.OptionInfo(...))`.
- `script_callbacks.on_model_loaded()`
  - Runs after `modules.sd_models.forge_model_reload()` loads a model.
- `script_callbacks.on_cfg_denoiser()`
  - Runs inside `CFGDenoiser.forward()` before model inference for a denoising step.
- `script_callbacks.on_cfg_after_cfg()`
  - Runs after CFG calculation and before returning the denoised latent.
- `script_callbacks.on_script_unloaded()`
  - Cleanup hook for reverting monkey patches.

Implication for MVP:

- Use `on_ui_settings()` for global settings such as `Enable Nz-fast-anima`,
  `Mode`, `Print timing log`, and `Verbose diagnose log`.
- Use an `AlwaysVisible` script only if per-generation UI controls are needed.
- Use `on_model_loaded()` and `process_before_every_sampling()` to establish
  model and generation context.
- Use `on_cfg_denoiser()` / `on_cfg_after_cfg()` to count steps and measure per-step
  timing without patching the model.

## Model loading and Anima detection

Relevant source:

- `backend/loader.py`
- `backend/diffusion_engine/base.py`
- `backend/nn/anima.py`
- `modules/sd_models.py`

Forge Neo loads checkpoints through `modules.sd_models.forge_model_reload()`, which
calls `backend.loader.forge_loader()`. The loaded model is stored as `shared.sd_model`
and receives:

- `sd_model.sd_checkpoint_info`
- `sd_model.filename`
- `sd_model.sd_model_hash`
- `sd_model.model_config`
- `sd_model.forge_objects`

The loader maps Hugging Face component class `CosmosTransformer3DModel` to
`backend.nn.anima.Anima`. It also checks whether `"Anima"` appears in
`guess.huggingface_repo`; if true, `process_anima()` moves `llm_adapter` weights from
the transformer state dict to the text encoder state dict.

Likely detection signals for `Nz-fast-anima`:

- `type(sd_model).__name__`
- `sd_model.model_config.__class__.__name__`
- `getattr(sd_model.model_config, "huggingface_repo", "")`
- `sd_model.filename`
- `sd_model.sd_checkpoint_info.name`
- `sd_model.forge_objects.unet.model.diffusion_model.__class__.__name__`
- Presence of `backend.nn.anima.Anima` or class name `"Anima"`

Recommended MVP behavior:

- Treat a model as supported when the loaded diffusion model class or model config
  clearly indicates Anima / Cosmos-Predict2.
- Treat filename or checkpoint title matches as weaker fallback signals.
- Log all detection signals in verbose mode.
- For unsupported models, do not patch or modify generation; log only.

## Sampling and timing path

Relevant source:

- `modules/sd_samplers_kdiffusion.py`
- `modules/sd_samplers_cfg_denoiser.py`
- `backend/sampling/sampling_function.py`
- `modules/processing.py`

For k-diffusion samplers, `KDiffusionSampler.sample()`:

- calls `sampling_prepare()`
- obtains sigmas with `get_sigmas()`
- sets `sampler_extra_args` with `cond`, `uncond`, `cond_scale`, and `image_cond`
- stores `sampling_sigmas` in
  `p.sd_model.forge_objects.unet.model_options["transformer_options"]`
- launches the sampler through `launch_sampling()`
- calls `sampling_cleanup()`

`CFGDenoiser.forward()`:

- reconstructs current cond/uncond batch for the step
- creates `CFGDenoiserParams`
- calls `cfg_denoiser_callback()`
- calls `backend.sampling.sampling_function.sampling_function()`
- receives `denoised`, `cond_pred`, and `uncond_pred`
- calls `cfg_after_cfg_callback()`

`backend.sampling.sampling_function.calc_cond_uncond_batch()`:

- builds a `to_run` list for cond and uncond branches
- batches compatible cond/uncond items together based on shape and condition
  compatibility
- passes `transformer_options` into model conditioning
- calls `model.apply_model(...)`
- splits batched outputs back into cond and uncond predictions

Useful timing points:

- Sampling start: `process_before_every_sampling()`
- Step start: `on_cfg_denoiser()`
- Step end: `on_cfg_after_cfg()`
- Sampling end: `postprocess()`

Useful diagnostics:

- `p.sampler_name`
- `p.scheduler`
- `p.steps`
- `p.cfg_scale`
- `p.width`, `p.height`
- `state.sampling_step`, `state.sampling_steps`
- whether `params.text_uncond is None`
- `params.denoiser.step` and `params.denoiser.total_steps`
- `cond_or_uncond`, `cond_indices`, and `uncond_indices` from `transformer_options`
  when available

## Attention backend path

Relevant source:

- `backend/attention.py`
- `backend/nn/anima.py`
- `backend/operations.py`
- `backend/args.py`

`backend.attention` defines:

- `attention_basic`
- `attention_pytorch`
- `attention_xformers`
- `attention_sage`
- `attention_flash`

At import time, it selects global `attention_function` in priority order based on
runtime availability and command-line flags:

1. SageAttention
2. FlashAttention
3. xFormers
4. PyTorch SDPA
5. basic fallback

Anima attention path:

- `backend.nn.anima.SelfCrossAttention.forward()`
- `compute_qkv()`
- `compute_attention()`
- `torch_attention_op()`
- `backend.attention.attention_function(...)`

Anima self-attention shape:

- `Block.forward()` receives `x_B_T_H_W_D`
- self-attention flattens it with `b t h w d -> b (t h w) d`
- for T2I, T is expected to be 1, so the effective attention grid is 2D flattened
  as `(H * W)` tokens

Implication for Phase 2:

- First diagnostic target is to log `backend.attention.attention_function.__name__`.
- Also log availability flags from `backend.memory_management` if importable:
  `sage_enabled()`, `flash_enabled()`, `xformers_enabled()`,
  `pytorch_attention_enabled()`.
- A later "Fast attention kernel" mode may temporarily replace
  `backend.attention.attention_function`, but must restore it on disable/unload.

Implication for Phase 5:

- 2D sparse attention should target Anima self-attention, not generic cross-attention.
- `SelfCrossAttention.compute_attention()` or `torch_attention_op()` is a natural
  patch point because q/k/v are already separated and the original `H, W` can be
  recovered only if the patch also captures block-level shape from `Block.forward()`.
- The existing flattening loses explicit 2D layout at the attention function boundary,
  so a robust sparse experiment likely needs to patch inside `backend.nn.anima.Block`
  or `SelfCrossAttention`, not only `backend.attention.attention_function`.

### Resolved: self-attention vs cross-attention separation

This item is resolved by reading `backend/nn/anima.py`.

`Block.__init__()` creates two separate attention modules:

- `self.self_attn = SelfCrossAttention(x_dim, None, ...)`
- `self.cross_attn = SelfCrossAttention(x_dim, context_dim, ...)`

`SelfCrossAttention.__init__()` sets:

```text
self.is_SelfAttn = context_dim is None
```

`Block.forward()` then calls self-attention and cross-attention in distinct sections:

- self-attention consumes the normalized latent tokens and passes `context=None`.
- cross-attention consumes the normalized latent tokens and passes `crossattn_emb`.

Both calls flatten latent layout with `b t h w d -> b (t h w) d` before entering
`SelfCrossAttention.forward()`, then reshape back to `b t h w d` after attention.

Conclusion:

- The code-level distinction between self-attention and cross-attention is explicit.
- 2D sparse attention should target self-attention first.
- Cross-attention should remain unchanged in the first sparse experiment because its
  key/value sequence comes from text/context embeddings rather than the 2D latent grid.
- `Block.forward()` is the safest place to observe `B/T/H/W/D` and block order before
  layout is flattened.

## Remaining information classification

Items that source reading can resolve or largely resolve:

- self-attention / cross-attention separation: resolved as above.
- Whether `Block.forward()` has H/W/T information before attention flattening:
  resolved. It has `B, T, H, W, D = x_B_T_H_W_D.shape`.
- Whether `compute_attention()` receives H/W/T directly: source says no. It receives
  q/k/v after flattening unless `Block.forward()` or an equivalent wrapper passes
  shape metadata through `transformer_options`.
- NATTEN nominal compatibility: NATTEN docs list PyTorch `2.11.0+cu130` wheels, but
  Windows builds are described as experimental and not regularly tested.

Items that lightweight diagnostic logging can resolve:

- runtime `len(blocks)`.
- block index/order during the first denoiser call.
- per-block `x_B_T_H_W_D` shape and rough block time.
- q/k/v shape for self-attention and cross-attention.
- `crossattn_emb` shape.
- model `patch_spatial`, `patch_temporal`, `num_heads`, and `head_dim`.

Items that require algorithmic experiments:

- Which block range can use 2D sparse attention without visible degradation.
- Which window size / dilation / stride is acceptable.
- Whether NATTEN is faster than a simpler dependency-free prototype in this workload.
- Whether low-bit / compile preserves image quality and improves repeated generation time.

## Cond/uncond behavior

Relevant source:

- `modules/processing.py`
- `backend/sampling/sampling_function.py`

`StableDiffusionProcessing.setup_conds()` already sets `uc = None` when
`cfg_scale == 1`, and logs that negative prompts are ignored.

`sampling_function_inner()` also skips unconditional processing when
`cond_scale` is close to `1.0` and `model_options["disable_cfg1_optimization"]` is
not set.

`calc_cond_uncond_batch()` attempts to batch cond and uncond together when compatible.

Implication for Phase 3:

- There may be limited obvious optimization left for `CFG=1.0`.
- More useful diagnostics are:
  - whether `uncond is None`
  - how many cond/uncond chunks are in each model call
  - whether cond/uncond are batched in one `apply_model` call
  - whether ControlNet, masks, area prompts, or differing condition shapes prevent
    batching
- If Forge already batches the normal Anima txt2img case, avoid deep changes here.

## Low-bit and compile-related implementation

Relevant source:

- `backend/args.py`
- `backend/loader.py`
- `backend/operations.py`
- `backend/operations_int8.py`
- `backend/operations_mixed_precision.py`

Command-line options include:

- `--fp16-unet`, `--bf16-unet`, `--fp32-unet`
- `--fp8_e4m3fn-unet`, `--fp8_e5m2-unet`, `--fp8_e8m0fnu-unet`
- `--fast-fp8`
- `--fast-fp16`
- `--sage`, `--flash`, `--xformers`
- `--sage-function`

The loader chooses storage and computation dtype during model construction. It uses
`backend.operations.using_forge_operations(...)` to temporarily replace PyTorch
modules such as `Linear`, `Conv*`, `LayerNorm`, `RMSNorm`, and `Embedding` while
building the model.

Existing operation families include:

- `ForgeOperations`
- `ForgeOperationsInt8`
- `ForgeOperationsBNB4bits`
- `ForgeOperationsGGUF`
- `ForgeOperationsFP8`
- mixed precision ops from `operations_mixed_precision.py`

Notable Anima-specific INT8 hook:

- `using_forge_operations()` selects `ForgeOperationsInt8` for `bnb_dtype == "Anima"`.
- It excludes module names containing `"embed"` and `"adaln"`.

Implication for Phase 4:

- Low-bit experiments should first observe `backend.args.dynamic_args.ops`,
  model `storage_dtype`, and `computation_dtype`.
- Runtime toggling of low-bit after model load is risky because Forge chooses operation
  classes while constructing the model.
- The safer extension experiment is initially diagnostic: report current dtype and ops.
- If the extension later changes low-bit behavior, it may need to trigger model reload
  through Forge loading parameters rather than patching an already-loaded model.

## Proposed investigation-first extension design

The first substantial extension should expose these modes:

- `Off`
- `Diagnose only`
- `Trace attention`
- `Trace cond/uncond`
- `Trace low-bit / compile`

Before implementing acceleration, the extension should be able to print:

- model detection result and all evidence
- selected attention backend
- sampler and scheduler
- resolution, steps, CFG
- total sampling time
- average denoiser step time
- per-step timing min / max / p50 / p95 if possible
- cond/uncond presence and batching shape hints
- model storage dtype and computation dtype
- selected Forge operation family
- peak VRAM if available from Torch CUDA APIs

Suggested implementation modules:

```text
scripts/nz_fast_anima.py
nz_fast_anima/__init__.py
nz_fast_anima/settings.py
nz_fast_anima/model_detect.py
nz_fast_anima/diagnostics.py
nz_fast_anima/timing.py
nz_fast_anima/forge_introspection.py
```

## Open questions

- Where is Forge Neo's current `torch.compile` toggle implemented? The README
  advertises it, but the exact runtime hook was not identified in this pass.
- Does Anima txt2img always use `backend.nn.anima.SelfCrossAttention`, or can a
  wrapped/compiled model hide that class behind another module?
- Is Anima always loaded from `CosmosTransformer3DModel`, or do some checkpoints use
  converted names that require filename-based detection?
- Does `on_cfg_denoiser()` fire exactly once per logical step for all target samplers
  (`ER SDE`, `Euler a`), or do some samplers call the denoiser multiple times per UI
  step?
- Is `state.sampling_step` reliable for average step timing with second-order samplers?
