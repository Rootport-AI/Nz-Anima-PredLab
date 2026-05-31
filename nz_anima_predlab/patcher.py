from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

from .logging import exception, info, warning
from .spectrum import FastChebyshevForecaster
from .state import (
    ATTENTION_BACKEND_CURRENT,
    ATTENTION_TARGET_BOTH,
    ATTENTION_TARGET_CROSS,
    ATTENTION_TARGET_SELF,
    MODE_IDENTITY_PATCH,
    SPARSE_BACKEND_NATTEN,
    SPARSE_BACKEND_TORCH,
    STATE,
    TEACACHE_CACHE_DEVICE_CPU,
    TEACACHE_COEFFICIENTS_ANIMA_2B_30STEP_FIRST_BLOCK_SHIFT,
    TEACACHE_PROFILE_ANIMA_2B_30STEP_FIRST_BLOCK_SHIFT,
    TEACACHE_PROFILE_IDENTITY,
    TEACACHE_SOURCE_FIRST_BLOCK_SHIFT,
    UJICACHE_FORMULA_LINEAR,
    UJICACHE_FORMULA_TAYLOR2,
    UJICACHE_FORMULA_TEACACHE,
)


@dataclass
class PatchResult:
    ok: bool
    kind: str
    message: str = ""


UJICACHE_MAX_NORM_RATIO = 3.0


def apply_patch(kind: str, context: Any = None) -> PatchResult:
    if kind == "cond_batch_trace":
        return _apply_cond_batch_trace_patch()
    if kind == "block_structure_trace":
        return _apply_block_structure_trace_patch()
    if kind == "block_forward_identity":
        return _apply_block_forward_identity_patch()
    if kind == "attention_kernel":
        return _apply_attention_kernel_patch()
    if kind == "sparse_attention":
        return _apply_sparse_attention_patch()
    if kind == "teacache":
        return _apply_teacache_patch()
    if kind == "ujicache":
        return _apply_ujicache_patch()
    if kind == "spectrum":
        return _apply_spectrum_patch()
    if kind == "tensor_dump":
        return _apply_tensor_dump_patch()
    if kind == "tensor_dump_output":
        return _apply_tensor_dump_output_patch()
    warning(f"patch '{kind}' is not implemented in the diagnostic build")
    return PatchResult(False, kind, "not implemented")


def remove_patch(kind: str) -> PatchResult:
    patch = STATE.patches.pop(kind, None)
    if patch is None:
        return PatchResult(True, kind, "not patched")
    try:
        restore = patch["restore"]
        restore()
        info(f"removed patch kind={kind}")
        return PatchResult(True, kind, "removed")
    except Exception as exc:
        STATE.set_error(f"failed to remove patch {kind}: {exc}")
        return PatchResult(False, kind, str(exc))


def remove_all_patches() -> PatchResult:
    ok = True
    messages: list[str] = []
    for kind in list(STATE.patches.keys()):
        result = remove_patch(kind)
        ok = ok and result.ok
        messages.append(f"{kind}:{result.message}")
    return PatchResult(ok, "all", ", ".join(messages))


def is_patched(kind: str) -> bool:
    return kind in STATE.patches


def _apply_cond_batch_trace_patch() -> PatchResult:
    kind = "cond_batch_trace"
    if is_patched(kind):
        return PatchResult(True, kind, "already patched")

    try:
        from backend.sampling import sampling_function
    except Exception as exc:
        return PatchResult(False, kind, f"import failed: {exc}")

    original = getattr(sampling_function, "calc_cond_uncond_batch", None)
    if original is None:
        return PatchResult(False, kind, "calc_cond_uncond_batch not found")

    def traced_calc_cond_uncond_batch(model, cond, uncond, x_in, timestep, model_options):
        if not STATE.active() or STATE.cond_batch_trace_logged:
            return original(model, cond, uncond, x_in, timestep, model_options)

        patched_model_options = dict(model_options or {})
        existing_wrapper = patched_model_options.get("model_function_wrapper")

        def model_function_wrapper(apply_model, args):
            _log_model_apply_args(args, cond, uncond)
            STATE.cond_batch_trace_logged = True
            if existing_wrapper is not None:
                return existing_wrapper(apply_model, args)
            return apply_model(args["input"], args["timestep"], **args["c"])

        patched_model_options["model_function_wrapper"] = model_function_wrapper
        return original(model, cond, uncond, x_in, timestep, patched_model_options)

    sampling_function.calc_cond_uncond_batch = traced_calc_cond_uncond_batch

    def restore() -> None:
        sampling_function.calc_cond_uncond_batch = original

    STATE.patches[kind] = {"restore": restore}
    info("applied diagnostic patch kind=cond_batch_trace")
    return PatchResult(True, kind, "applied")


def _log_model_apply_args(args: dict[str, Any], cond: Any, uncond: Any) -> None:
    c = args.get("c") or {}
    transformer_options = c.get("transformer_options", {})
    input_x = args.get("input")
    timestep = args.get("timestep")
    cond_or_uncond = args.get("cond_or_uncond")
    cond_indices = transformer_options.get("cond_indices")
    uncond_indices = transformer_options.get("uncond_indices")

    info(
        "cond_batch="
        f"cond_or_uncond={cond_or_uncond} cond_indices={cond_indices} "
        f"uncond_indices={uncond_indices} input_shape={_shape(input_x)} "
        f"timestep_shape={_shape(timestep)} cond_len={_len(cond)} "
        f"uncond_len={_len(uncond)}"
    )


def _apply_spectrum_patch() -> PatchResult:
    kind = "spectrum"
    if is_patched(kind):
        return PatchResult(True, kind, "already patched")

    try:
        from backend.sampling import sampling_function
    except Exception as exc:
        return PatchResult(False, kind, f"import failed: {exc}")

    original = getattr(sampling_function, "calc_cond_uncond_batch", None)
    if original is None or not callable(original):
        return PatchResult(False, kind, "calc_cond_uncond_batch not found")

    runtime = _spectrum_new_runtime()

    def spectrum_calc_cond_uncond_batch(model, cond, uncond, x_in, timestep, model_options):
        if not _should_spectrum_patch():
            return original(model, cond, uncond, x_in, timestep, model_options)

        patched_model_options = dict(model_options or {})
        existing_wrapper = patched_model_options.get("model_function_wrapper")
        if existing_wrapper is not None:
            _spectrum_mark_unavailable("existing_model_function_wrapper")
            return original(model, cond, uncond, x_in, timestep, model_options)

        def model_function_wrapper(model_function, args):
            return _spectrum_model_function_wrapper(model_function, args, runtime)

        patched_model_options["model_function_wrapper"] = model_function_wrapper
        return original(model, cond, uncond, x_in, timestep, patched_model_options)

    sampling_function.calc_cond_uncond_batch = spectrum_calc_cond_uncond_batch

    def restore() -> None:
        sampling_function.calc_cond_uncond_batch = original

    STATE.patches[kind] = {"restore": restore}
    info(
        "applied experimental patch kind=spectrum "
        "target=backend.sampling.sampling_function.calc_cond_uncond_batch "
        f"preset={STATE.spectrum_preset} w={STATE.spectrum_w:.2f} "
        f"m={STATE.spectrum_m} lambda={STATE.spectrum_lambda:.2f}"
    )
    return PatchResult(True, kind, "applied")


def _should_spectrum_patch() -> bool:
    return STATE.active() and STATE.spectrum_enabled and not STATE.teacache_enabled


def _spectrum_new_runtime() -> dict[str, Any]:
    return {
        "generation_index": None,
        "forecaster": None,
        "cnt": 0,
        "num_cached": 0,
        "curr_ws": 1.0,
        "last_t": None,
        "last_input_signature": None,
    }


def _spectrum_reset_runtime(runtime: dict[str, Any]) -> None:
    runtime["generation_index"] = STATE.generation_index
    runtime["forecaster"] = None
    runtime["cnt"] = 0
    runtime["num_cached"] = 0
    runtime["curr_ws"] = float(max(1, STATE.spectrum_window_size))
    runtime["last_t"] = None
    runtime["last_input_signature"] = None


def _spectrum_ensure_runtime(runtime: dict[str, Any]) -> None:
    if runtime.get("generation_index") != STATE.generation_index:
        _spectrum_reset_runtime(runtime)


def _spectrum_mark_unavailable(reason: str) -> None:
    if STATE.spectrum_unavailable_reason != reason:
        STATE.spectrum_unavailable_reason = reason
        warning(f"spectrum_unavailable reason={reason}")


def _spectrum_model_function_wrapper(model_function: Any, args: dict[str, Any], runtime: dict[str, Any]):
    try:
        return _spectrum_model_function_wrapper_body(model_function, args, runtime)
    except Exception as exc:
        STATE.spectrum_errors += 1
        STATE.spectrum_fallbacks += 1
        STATE.spectrum_unavailable_reason = _short_error(exc)
        if STATE.spectrum_logged_calls < 12:
            STATE.spectrum_logged_calls += 1
            warning(f"spectrum_fallback=reason={_short_error(exc)} route=actual_forward")
        return _spectrum_actual_forward(model_function, args)


def _spectrum_model_function_wrapper_body(model_function: Any, args: dict[str, Any], runtime: dict[str, Any]):
    _spectrum_ensure_runtime(runtime)

    x = args.get("input")
    timestep = args.get("timestep")
    if x is None or timestep is None or args.get("c") is None:
        raise RuntimeError("Spectrum wrapper args are incomplete")

    reset_reason = _spectrum_runtime_reset_reason(runtime, x, timestep)
    if reset_reason:
        _spectrum_reset_runtime(runtime)

    STATE.spectrum_model_calls += 1
    cnt = int(runtime.get("cnt", 0))
    step_index = max(0, STATE.denoiser_calls - 1)
    progress = _spectrum_progress(cnt)
    reason = reset_reason or _spectrum_actual_reason(runtime, cnt, progress)

    if reason is not None:
        out = _spectrum_actual_forward(model_function, args)
        _dump_spectrum_final_output(out, cnt, step_index, timestep, reason)
        STATE.spectrum_actual_forwards += 1
        _spectrum_update_after_actual(runtime, cnt, out, reason)
        _spectrum_record_runtime_input(runtime, x, timestep)
        _spectrum_log_call("actual", cnt, step_index, progress, reason, runtime)
        return out

    if STATE.spectrum_dry_run:
        out = _spectrum_actual_forward(model_function, args)
        _dump_spectrum_final_output(out, cnt, step_index, timestep, "dry_run")
        STATE.spectrum_actual_forwards += 1
        STATE.spectrum_dry_run_forecasts += 1
        runtime["num_cached"] = int(runtime.get("num_cached", 0)) + 1
        runtime["cnt"] = cnt + 1
        _spectrum_record_runtime_input(runtime, x, timestep)
        _spectrum_log_call("actual", cnt, step_index, progress, "dry_run", runtime)
        return out

    try:
        forecaster = runtime.get("forecaster")
        if forecaster is None or not forecaster.ready():
            raise RuntimeError("forecaster is not ready")
        out = _spectrum_cast_forecast(forecaster.predict(cnt, STATE.spectrum_w), x)
        _spectrum_validate_forecast(out, x)
        STATE.spectrum_forecasts += 1
        runtime["num_cached"] = int(runtime.get("num_cached", 0)) + 1
        runtime["cnt"] = cnt + 1
        _spectrum_record_runtime_input(runtime, x, timestep)
        _spectrum_log_call("forecast", cnt, step_index, progress, "window", runtime)
        return out
    except Exception as exc:
        STATE.spectrum_fallbacks += 1
        out = _spectrum_actual_forward(model_function, args)
        _dump_spectrum_final_output(out, cnt, step_index, timestep, "fallback")
        STATE.spectrum_actual_forwards += 1
        _spectrum_update_after_actual(runtime, cnt, out, f"fallback:{_short_error(exc)}")
        _spectrum_record_runtime_input(runtime, x, timestep)
        _spectrum_log_call("actual", cnt, step_index, progress, "fallback", runtime)
        return out


def _spectrum_actual_forward(model_function: Any, args: dict[str, Any]):
    return model_function(args["input"], args["timestep"], **args["c"])


def _dump_spectrum_final_output(
    out: Any,
    cnt: int,
    step_index: int,
    timestep: Any,
    reason: str,
) -> None:
    if not (STATE.tensor_dump_active() and STATE.dump_spectrum_final_output):
        return
    local_call_index = STATE.tensor_dump_spectrum_local_call_index
    STATE.tensor_dump_spectrum_local_call_index += 1
    from .tensor_dump import dump_tensor

    dump_tensor(
        "spectrum_final_output",
        out,
        logical_step_index=step_index,
        local_call_index=local_call_index,
        call_index=cnt,
        decision="actual",
        timestep_value=timestep,
        spectrum_cnt=cnt,
        extra={"reason": reason, "source": "spectrum"},
    )


def _apply_tensor_dump_output_patch() -> PatchResult:
    kind = "tensor_dump_output"
    if is_patched(kind):
        return PatchResult(True, kind, "already patched")

    try:
        from backend.sampling import sampling_function
    except Exception as exc:
        return PatchResult(False, kind, f"import failed: {exc}")

    original = getattr(sampling_function, "calc_cond_uncond_batch", None)
    if original is None or not callable(original):
        return PatchResult(False, kind, "calc_cond_uncond_batch not found")

    def dumped_calc_cond_uncond_batch(model, cond, uncond, x_in, timestep, model_options):
        if not _should_tensor_dump_output_patch():
            return original(model, cond, uncond, x_in, timestep, model_options)

        patched_model_options = dict(model_options or {})
        existing_wrapper = patched_model_options.get("model_function_wrapper")
        if existing_wrapper is not None:
            _tensor_dump_warn_once(
                "existing_model_function_wrapper",
                "tensor_dump_output_unavailable reason=existing_model_function_wrapper",
            )
            return original(model, cond, uncond, x_in, timestep, model_options)

        def model_function_wrapper(model_function, args):
            out = model_function(args["input"], args["timestep"], **args["c"])
            local_call_index = STATE.tensor_dump_baseline_local_call_index
            STATE.tensor_dump_baseline_local_call_index += 1
            from .tensor_dump import dump_tensor

            dump_tensor(
                "baseline_final_output",
                out,
                logical_step_index=max(0, STATE.denoiser_calls - 1),
                local_call_index=local_call_index,
                call_index=local_call_index,
                decision="actual",
                timestep_value=args.get("timestep"),
                extra={"source": "baseline"},
            )
            return out

        patched_model_options["model_function_wrapper"] = model_function_wrapper
        return original(model, cond, uncond, x_in, timestep, patched_model_options)

    sampling_function.calc_cond_uncond_batch = dumped_calc_cond_uncond_batch

    def restore() -> None:
        sampling_function.calc_cond_uncond_batch = original

    STATE.patches[kind] = {"restore": restore}
    info("applied diagnostic patch kind=tensor_dump_output target=model_function_output")
    return PatchResult(True, kind, "applied")


def _should_tensor_dump_output_patch() -> bool:
    return (
        STATE.tensor_dump_active()
        and STATE.dump_baseline_final_output
        and not STATE.spectrum_enabled
    )


def _spectrum_cast_forecast(out: Any, x: Any) -> Any:
    if not hasattr(out, "to"):
        return out
    kwargs: dict[str, Any] = {}
    dtype = getattr(x, "dtype", None)
    device = getattr(x, "device", None)
    if dtype is not None:
        kwargs["dtype"] = dtype
    if device is not None:
        kwargs["device"] = device
    if not kwargs:
        return out
    return out.to(**kwargs)


def _spectrum_update_after_actual(runtime: dict[str, Any], cnt: int, out: Any, reason: str) -> None:
    import torch

    if not torch.is_tensor(out):
        runtime["cnt"] = cnt + 1
        runtime["num_cached"] = 0
        return

    forecaster = runtime.get("forecaster")
    if forecaster is None:
        forecaster = FastChebyshevForecaster(
            m=STATE.spectrum_m,
            lam=STATE.spectrum_lambda,
            steps=_spectrum_steps(),
        )
        runtime["forecaster"] = forecaster

    if not forecaster.compatible(out):
        forecaster.reset()
    forecaster.update(cnt, out)
    if cnt >= STATE.spectrum_warmup_steps and not reason.startswith("first_call"):
        runtime["curr_ws"] = float(runtime.get("curr_ws", 1.0)) + STATE.spectrum_flex_window
    runtime["num_cached"] = 0
    runtime["cnt"] = cnt + 1


def _spectrum_actual_reason(runtime: dict[str, Any], cnt: int, progress: float) -> str | None:
    import math

    if cnt == 0:
        return "first_call"
    if cnt < STATE.spectrum_warmup_steps:
        return "warmup"
    if progress >= STATE.spectrum_stop_progress:
        return "tail_guard"
    forecaster = runtime.get("forecaster")
    if forecaster is None or not forecaster.ready():
        return "no_history"
    current_ws = max(1, int(math.floor(float(runtime.get("curr_ws", 1.0)))))
    if (int(runtime.get("num_cached", 0)) + 1) % current_ws == 0:
        return "window"
    return None


def _spectrum_runtime_reset_reason(runtime: dict[str, Any], x: Any, timestep: Any) -> str | None:
    signature = _spectrum_input_signature(x)
    previous_signature = runtime.get("last_input_signature")
    if previous_signature is not None and signature != previous_signature:
        return "shape_mismatch"

    current_t = _safe_float_timestep(timestep)
    previous_t = runtime.get("last_t")
    if previous_t is not None and current_t > float(previous_t) + 1e-7:
        return "timestep_reset"
    return None


def _spectrum_input_signature(x: Any) -> tuple[Any, Any, Any]:
    return (getattr(x, "shape", None), getattr(x, "dtype", None), getattr(x, "device", None))


def _spectrum_record_runtime_input(runtime: dict[str, Any], x: Any, timestep: Any) -> None:
    runtime["last_t"] = _safe_float_timestep(timestep)
    runtime["last_input_signature"] = _spectrum_input_signature(x)


def _spectrum_progress(cnt: int) -> float:
    steps = _spectrum_steps()
    if steps <= 1:
        return 0.0
    return max(0.0, min(1.0, float(cnt) / float(steps - 1)))


def _spectrum_steps() -> int:
    try:
        steps = int(STATE.generation_steps or 0)
    except Exception:
        steps = 0
    return max(1, steps or 50)


def _safe_float_timestep(timestep: Any) -> float:
    try:
        if hasattr(timestep, "flatten"):
            return float(timestep.flatten()[0].item())
        return float(timestep)
    except Exception:
        return 0.0


def _spectrum_validate_forecast(out: Any, x: Any) -> None:
    import torch

    if getattr(out, "shape", None) != getattr(x, "shape", None):
        raise RuntimeError(f"forecast shape mismatch output={_shape(out)} input={_shape(x)}")
    if getattr(out, "dtype", None) != getattr(x, "dtype", None):
        raise RuntimeError(f"forecast dtype mismatch output={_dtype(out)} input={_dtype(x)}")
    if getattr(out, "device", None) != getattr(x, "device", None):
        raise RuntimeError(f"forecast device mismatch output={_device(out)} input={_device(x)}")
    if not torch.isfinite(out).all():
        raise RuntimeError("forecast contains NaN or Inf")


def _spectrum_log_call(
    decision: str,
    cnt: int,
    step_index: int,
    progress: float,
    reason: str,
    runtime: dict[str, Any],
) -> None:
    if not STATE.spectrum_verbose_trace and STATE.spectrum_logged_calls >= 12:
        return
    STATE.spectrum_logged_calls += 1
    forecaster = runtime.get("forecaster")
    history = len(getattr(forecaster, "h_buf", []) or [])
    info(
        "spectrum_call="
        f"call={STATE.spectrum_model_calls} step={step_index} cnt={cnt} "
        f"progress={progress:.3f} decision={decision} reason={reason} "
        f"history={history} window={float(runtime.get('curr_ws', 1.0)):.2f} "
        f"dry_run={STATE.spectrum_dry_run}"
    )


def _shape(value: Any) -> str:
    shape = getattr(value, "shape", None)
    if shape is None:
        return ""
    try:
        return "x".join(str(part) for part in shape)
    except Exception:
        return str(shape)


def _len(value: Any) -> str:
    if value is None:
        return "None"
    try:
        return str(len(value))
    except Exception:
        return "unknown"


def _apply_block_structure_trace_patch() -> PatchResult:
    kind = "block_structure_trace"
    if is_patched(kind):
        return PatchResult(True, kind, "already patched")

    try:
        from backend.nn import anima
    except Exception as exc:
        return PatchResult(False, kind, f"import failed: {exc}")

    block_cls = getattr(anima, "Block", None)
    attention_cls = getattr(anima, "SelfCrossAttention", None)
    if block_cls is None or attention_cls is None:
        return PatchResult(False, kind, "Anima Block/SelfCrossAttention not found")

    original_block_forward = block_cls.forward
    original_compute_qkv = attention_cls.compute_qkv

    def traced_block_forward(self, x_B_T_H_W_D, *args, **kwargs):
        if not _should_trace_block():
            return original_block_forward(self, x_B_T_H_W_D, *args, **kwargs)

        block_index = STATE.block_trace_call_count
        STATE.block_trace_call_count += 1
        previous_block_index = STATE.current_block_index
        STATE.current_block_index = block_index

        start = perf_counter()
        try:
            return original_block_forward(self, x_B_T_H_W_D, *args, **kwargs)
        finally:
            elapsed = perf_counter() - start
            STATE.current_block_index = previous_block_index
            info(
                "block_trace="
                f"index={block_index} x_shape={_shape(x_B_T_H_W_D)} "
                f"elapsed={elapsed:.4f}s self_attn={hasattr(self, 'self_attn')} "
                f"cross_attn={hasattr(self, 'cross_attn')}"
            )

    def traced_compute_qkv(self, x, context=None, rope_emb=None):
        q, k, v = original_compute_qkv(self, x, context=context, rope_emb=rope_emb)
        block_index = STATE.current_block_index
        attn_type = "self" if getattr(self, "is_SelfAttn", False) else "cross"
        key = (block_index if block_index is not None else -1, attn_type)
        if _should_trace_qkv(key):
            STATE.block_trace_qkv_logged.add(key)
            info(
                "qkv_trace="
                f"block={block_index} type={attn_type} x_shape={_shape(x)} "
                f"context_shape={_shape(context)} q_shape={_shape(q)} "
                f"k_shape={_shape(k)} v_shape={_shape(v)} "
                f"heads={getattr(self, 'n_heads', None)} "
                f"head_dim={getattr(self, 'head_dim', None)}"
            )
        return q, k, v

    block_cls.forward = traced_block_forward
    attention_cls.compute_qkv = traced_compute_qkv

    def restore() -> None:
        block_cls.forward = original_block_forward
        attention_cls.compute_qkv = original_compute_qkv

    STATE.patches[kind] = {"restore": restore}
    info("applied diagnostic patch kind=block_structure_trace")
    return PatchResult(True, kind, "applied")


def _should_trace_block() -> bool:
    return STATE.active() and STATE.denoiser_calls <= 1 and STATE.block_trace_call_count < 64


def _should_trace_qkv(key: tuple[int, str]) -> bool:
    return (
        STATE.active()
        and STATE.denoiser_calls <= 1
        and key not in STATE.block_trace_qkv_logged
        and len(STATE.block_trace_qkv_logged) < 128
    )


def _apply_block_forward_identity_patch() -> PatchResult:
    kind = "block_forward_identity"
    if is_patched(kind):
        return PatchResult(True, kind, "already patched")

    try:
        from backend.nn import anima
    except Exception as exc:
        return PatchResult(False, kind, f"import failed: {exc}")

    block_cls = getattr(anima, "Block", None)
    if block_cls is None:
        return PatchResult(False, kind, "Anima Block not found")

    original_block_forward = block_cls.forward

    def identity_block_forward(self, x_B_T_H_W_D, *args, **kwargs):
        if not _should_identity_patch():
            return original_block_forward(self, x_B_T_H_W_D, *args, **kwargs)

        _ensure_identity_num_blocks()
        call_index = STATE.identity_patch_calls
        STATE.identity_patch_calls += 1
        block_index = _identity_block_index(call_index)

        try:
            output = original_block_forward(self, x_B_T_H_W_D, *args, **kwargs)
        except Exception:
            STATE.identity_patch_errors += 1
            STATE.set_error("identity patch failed inside Anima Block.forward")
            exception("identity_patch_exception target=backend.nn.anima.Block.forward")
            raise

        same_shape = _shape(output) == _shape(x_B_T_H_W_D)
        if not same_shape:
            STATE.identity_patch_shape_mismatches += 1

        if _should_log_identity_call(call_index, block_index):
            STATE.identity_patch_logged_calls += 1
            info(
                "identity_patch_call="
                f"call={call_index} block_index={block_index} "
                f"input_shape={_shape(x_B_T_H_W_D)} output_shape={_shape(output)} "
                f"same_shape={same_shape} input_dtype={_dtype(x_B_T_H_W_D)} "
                f"output_dtype={_dtype(output)} device={_device(output)} "
                "route=Nz-Anima-PredLab->original_Block.forward"
            )

        return output

    block_cls.forward = identity_block_forward

    def restore() -> None:
        block_cls.forward = original_block_forward

    STATE.patches[kind] = {"restore": restore}
    info(
        "applied identity patch kind=block_forward_identity "
        "target=backend.nn.anima.Block.forward behavior=call_original"
    )
    return PatchResult(True, kind, "applied")


def _should_identity_patch() -> bool:
    return STATE.active() and STATE.mode == MODE_IDENTITY_PATCH


def _ensure_identity_num_blocks() -> None:
    if STATE.identity_patch_num_blocks is not None:
        return
    STATE.identity_patch_num_blocks = _runtime_num_blocks()


def _runtime_num_blocks() -> int | None:
    blocks = _runtime_blocks()
    if blocks is None:
        return None
    return len(blocks)


def _runtime_blocks() -> Any | None:
    try:
        from modules import shared

        sd_model = getattr(shared, "sd_model", None)
        forge_objects = getattr(sd_model, "forge_objects", None)
        if isinstance(forge_objects, dict):
            unet = forge_objects.get("unet")
        else:
            unet = getattr(forge_objects, "unet", None)
        model = getattr(unet, "model", None)
        diffusion_model = getattr(model, "diffusion_model", model)
        blocks = getattr(diffusion_model, "blocks", None)
        if blocks is None:
            return None
        return blocks
    except Exception:
        return None


def _identity_block_index(call_index: int) -> int | str:
    num_blocks = STATE.identity_patch_num_blocks
    if not num_blocks:
        return "unknown"
    return call_index % num_blocks


def _should_log_identity_call(call_index: int, block_index: int | str) -> bool:
    if STATE.identity_patch_logged_calls < 8:
        return True
    if block_index == 0 and call_index < 256:
        return True
    return False


def _dtype(value: Any) -> str:
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        return ""
    return str(dtype)


def _device(value: Any) -> str:
    device = getattr(value, "device", None)
    if device is None:
        return ""
    return str(device)


def _apply_attention_kernel_patch() -> PatchResult:
    kind = "attention_kernel"
    if is_patched(kind):
        return PatchResult(True, kind, "already patched")

    try:
        from backend.nn import anima
    except Exception as exc:
        return PatchResult(False, kind, f"import failed: {exc}")

    block_cls = getattr(anima, "Block", None)
    attention_cls = getattr(anima, "SelfCrossAttention", None)
    if block_cls is None or attention_cls is None:
        return PatchResult(False, kind, "Anima Block/SelfCrossAttention not found")

    original_block_forward = block_cls.forward
    original_compute_attention = attention_cls.compute_attention

    def attention_block_forward(self, x_B_T_H_W_D, *args, **kwargs):
        if not _should_attention_kernel_patch():
            return original_block_forward(self, x_B_T_H_W_D, *args, **kwargs)

        _ensure_attention_num_blocks()
        call_index = STATE.attention_kernel_block_calls
        STATE.attention_kernel_block_calls += 1
        block_index = _attention_block_index(call_index)
        previous_context = STATE.attention_kernel_current_context
        STATE.attention_kernel_current_context = {"block_index": block_index}
        try:
            return original_block_forward(self, x_B_T_H_W_D, *args, **kwargs)
        finally:
            STATE.attention_kernel_current_context = previous_context

    def attention_compute_attention(self, q, k, v, transformer_options=None):
        if not _should_replace_attention_kernel(self):
            return original_compute_attention(
                self,
                q,
                k,
                v,
                transformer_options=transformer_options or {},
            )
        context = STATE.attention_kernel_current_context or {}
        try:
            unavailable_reason = _attention_backend_unavailable_reason(STATE.attention_backend)
            if unavailable_reason:
                raise RuntimeError(unavailable_reason)
            backend_fn = _attention_backend_function(STATE.attention_backend)
            if backend_fn is None:
                raise RuntimeError(f"attention backend not found: {STATE.attention_backend}")
            result = _compute_attention_with_backend(
                backend_fn,
                q,
                k,
                v,
                transformer_options=transformer_options or {},
            )
            STATE.attention_kernel_calls += 1
            if STATE.attention_kernel_logged_calls < 12:
                STATE.attention_kernel_logged_calls += 1
                attn_type = "self" if getattr(self, "is_SelfAttn", False) else "cross"
                trace = STATE.attention_kernel_last_trace or {}
                info(
                    "attention_kernel_call="
                    f"call={STATE.attention_kernel_calls} block={context.get('block_index')} "
                    f"type={attn_type} requested_backend={STATE.attention_backend} "
                    f"actual_backend={trace.get('actual_backend', 'unknown')} "
                    f"internal_fallback={trace.get('internal_fallback', 'unknown')} "
                    f"backend_trace={_format_attention_backend_trace(trace)} "
                    f"q_shape={_shape(q)} result_shape={_shape(result)}"
                )
            return self.output_dropout(self.output_proj(result))
        except Exception as exc:
            STATE.attention_kernel_errors += 1
            STATE.attention_kernel_fallbacks += 1
            if STATE.attention_kernel_logged_calls < 12:
                STATE.attention_kernel_logged_calls += 1
                warning(
                    "attention_kernel_fallback="
                    f"reason={exc} block={context.get('block_index')} "
                    f"backend={STATE.attention_backend}"
                )
            return original_compute_attention(
                self,
                q,
                k,
                v,
                transformer_options=transformer_options or {},
            )

    block_cls.forward = attention_block_forward
    attention_cls.compute_attention = attention_compute_attention

    def restore() -> None:
        block_cls.forward = original_block_forward
        attention_cls.compute_attention = original_compute_attention

    STATE.patches[kind] = {"restore": restore}
    info(
        "applied experimental patch kind=attention_kernel "
        f"backend={STATE.attention_backend} target={STATE.attention_target} "
        f"blocks={STATE.attention_block_start}..{STATE.attention_block_end}"
    )
    return PatchResult(True, kind, "applied")


def _should_attention_kernel_patch() -> bool:
    return STATE.active() and STATE.attention_override_active()


def _ensure_attention_num_blocks() -> None:
    if STATE.attention_kernel_num_blocks is not None:
        return
    STATE.attention_kernel_num_blocks = _runtime_num_blocks()


def _attention_block_index(call_index: int) -> int | str:
    num_blocks = STATE.attention_kernel_num_blocks
    if not num_blocks:
        return "unknown"
    return call_index % num_blocks


def _should_replace_attention_kernel(attn_module: Any) -> bool:
    if not _should_attention_kernel_patch():
        return False
    context = STATE.attention_kernel_current_context
    if not context:
        return False
    block_index = context.get("block_index")
    if not isinstance(block_index, int):
        return False
    if not (STATE.attention_block_start <= block_index <= STATE.attention_block_end):
        return False
    is_self = bool(getattr(attn_module, "is_SelfAttn", False))
    if STATE.attention_target == ATTENTION_TARGET_SELF and not is_self:
        return False
    if STATE.attention_target == ATTENTION_TARGET_CROSS and is_self:
        return False
    return STATE.attention_target in (ATTENTION_TARGET_BOTH, ATTENTION_TARGET_SELF, ATTENTION_TARGET_CROSS)


def _attention_backend_function(name: str) -> Any | None:
    if name == ATTENTION_BACKEND_CURRENT:
        return None
    if _attention_backend_unavailable_reason(name):
        return None
    try:
        from backend import attention
    except Exception:
        return None
    return getattr(attention, name, None)


def _attention_backend_unavailable_reason(name: str) -> str:
    try:
        from backend import attention
    except Exception as exc:
        return f"backend.attention import failed: {exc}"
    if name == "attention_xformers":
        xformers_module = getattr(attention, "xformers", None)
        xformers_ops = getattr(xformers_module, "ops", None)
        if not callable(getattr(xformers_ops, "memory_efficient_attention", None)):
            return "xformers is not available in backend.attention"
    if not callable(getattr(attention, name, None)):
        return f"attention backend function is not callable: {name}"
    return ""


def _compute_attention_with_backend(backend_fn: Any, q: Any, k: Any, v: Any, transformer_options: dict[str, Any]):
    from einops import rearrange

    in_q_shape = q.shape
    in_k_shape = k.shape
    q_bhsd = rearrange(q, "b ... h d -> b h ... d").view(
        in_q_shape[0],
        in_q_shape[-2],
        -1,
        in_q_shape[-1],
    )
    k_bhsd = rearrange(k, "b ... h d -> b h ... d").view(
        in_k_shape[0],
        in_k_shape[-2],
        -1,
        in_k_shape[-1],
    )
    v_bhsd = rearrange(v, "b ... h d -> b h ... d").view(
        in_k_shape[0],
        in_k_shape[-2],
        -1,
        in_k_shape[-1],
    )
    return _call_attention_backend_with_trace(
        backend_fn,
        q_bhsd,
        k_bhsd,
        v_bhsd,
        in_q_shape[-2],
        transformer_options,
    )


def _call_attention_backend_with_trace(
    backend_fn: Any,
    q: Any,
    k: Any,
    v: Any,
    heads: int,
    transformer_options: dict[str, Any],
):
    trace = {
        "counts": {
            "sage": 0,
            "flash": 0,
            "xformers": 0,
            "pytorch_sdpa": 0,
        },
        "errors": [],
        "trace_errors": [],
        "actual_backend": "unknown",
        "internal_fallback": False,
    }
    restores: list[tuple[Any, str, Any]] = []

    try:
        _install_attention_backend_trace_wrappers(trace, restores)
        result = backend_fn(
            q,
            k,
            v,
            heads,
            skip_reshape=True,
            transformer_options=transformer_options,
        )
        _record_attention_backend_trace(trace)
        return result
    except Exception:
        _record_attention_backend_trace(trace)
        raise
    finally:
        for owner, name, original in reversed(restores):
            try:
                setattr(owner, name, original)
            except Exception:
                pass


def _install_attention_backend_trace_wrappers(trace: dict[str, Any], restores: list[tuple[Any, str, Any]]) -> None:
    try:
        from backend import attention, operations
    except Exception as exc:
        trace["trace_errors"].append(f"import:{_short_error(exc)}")
        return

    _wrap_attention_callable(attention, "sageattn", "sage", trace, restores)
    _wrap_attention_callable(attention, "flash_attn_wrapper", "flash", trace, restores)
    xformers_ops = getattr(getattr(attention, "xformers", None), "ops", None)
    _wrap_attention_callable(xformers_ops, "memory_efficient_attention", "xformers", trace, restores)
    _wrap_attention_callable(operations, "scaled_dot_product_attention", "pytorch_sdpa", trace, restores)


def _wrap_attention_callable(
    owner: Any,
    name: str,
    label: str,
    trace: dict[str, Any],
    restores: list[tuple[Any, str, Any]],
) -> None:
    if owner is None or not hasattr(owner, name):
        return
    original = getattr(owner, name)
    if not callable(original):
        return

    def traced_callable(*args, **kwargs):
        trace["counts"][label] += 1
        try:
            return original(*args, **kwargs)
        except Exception as exc:
            trace["errors"].append(f"{label}:{_short_error(exc)}")
            raise

    try:
        setattr(owner, name, traced_callable)
        restores.append((owner, name, original))
    except Exception as exc:
        trace["trace_errors"].append(f"{label}:{_short_error(exc)}")


def _record_attention_backend_trace(trace: dict[str, Any]) -> None:
    actual_backend = _infer_actual_attention_backend(STATE.attention_backend, trace)
    internal_fallback = _is_attention_internal_fallback(STATE.attention_backend, actual_backend, trace)
    trace["actual_backend"] = actual_backend
    trace["internal_fallback"] = internal_fallback
    STATE.attention_kernel_last_trace = trace
    STATE.attention_kernel_actual_counts[actual_backend] = (
        STATE.attention_kernel_actual_counts.get(actual_backend, 0) + 1
    )
    if internal_fallback:
        STATE.attention_kernel_internal_fallbacks += 1
    STATE.attention_kernel_internal_errors += len(trace.get("errors", []))


def _infer_actual_attention_backend(requested_backend: str, trace: dict[str, Any]) -> str:
    counts = trace.get("counts", {})
    if counts.get("pytorch_sdpa", 0):
        if requested_backend == "attention_pytorch":
            return "pytorch_sdpa"
        return "pytorch_sdpa_fallback"
    if counts.get("sage", 0):
        return "sage"
    if counts.get("flash", 0):
        return "flash"
    if counts.get("xformers", 0):
        return "xformers"
    if trace.get("trace_errors"):
        return "trace_unavailable"
    return "unobserved"


def _is_attention_internal_fallback(requested_backend: str, actual_backend: str, trace: dict[str, Any]) -> bool:
    if requested_backend == "attention_pytorch":
        return False
    if actual_backend.endswith("_fallback"):
        return True
    counts = trace.get("counts", {})
    return bool(counts.get("pytorch_sdpa", 0))


def _format_attention_backend_trace(trace: dict[str, Any] | None) -> str:
    if not trace:
        return ""
    counts = trace.get("counts", {})
    parts = [
        f"sage:{counts.get('sage', 0)}",
        f"flash:{counts.get('flash', 0)}",
        f"xformers:{counts.get('xformers', 0)}",
        f"pytorch_sdpa:{counts.get('pytorch_sdpa', 0)}",
    ]
    errors = trace.get("errors") or []
    trace_errors = trace.get("trace_errors") or []
    if errors:
        parts.append(f"errors:{len(errors)}")
    if trace_errors:
        parts.append(f"trace_errors:{len(trace_errors)}")
    return ",".join(parts)


def _short_error(exc: Exception) -> str:
    text = str(exc).replace("\n", " ")
    if len(text) > 160:
        return text[:157] + "..."
    return text


def _tensor_dump_warn_once(key: str, message: str) -> None:
    if key in STATE.tensor_dump_warned_reasons:
        return
    STATE.tensor_dump_warned_reasons.add(key)
    warning(message)


def _apply_teacache_patch() -> PatchResult:
    kind = "teacache"
    if is_patched(kind):
        return PatchResult(True, kind, "already patched")

    try:
        from backend.nn import anima
    except Exception as exc:
        return PatchResult(False, kind, f"import failed: {exc}")

    anima_cls = getattr(anima, "Anima", None)
    if anima_cls is None:
        return PatchResult(False, kind, "Anima class not found")

    target_name = "_forward"
    original_forward = getattr(anima_cls, target_name, None)
    if original_forward is None or not callable(original_forward):
        target_name = "forward"
        original_forward = getattr(anima_cls, target_name, None)
    if original_forward is None or not callable(original_forward):
        return PatchResult(False, kind, "Anima._forward/forward not found")

    def teacache_forward(self, x, timesteps, context, *args, **kwargs):
        if not _should_teacache_patch():
            return original_forward(self, x, timesteps, context, *args, **kwargs)
        try:
            fps, padding_mask, body_kwargs = _teacache_parse_forward_args(
                target_name,
                args,
                kwargs,
            )
            return _teacache_forward_body(
                self,
                original_forward,
                x,
                timesteps,
                context,
                fps=fps,
                padding_mask=padding_mask,
                **body_kwargs,
            )
        except Exception as exc:
            STATE.teacache_errors += 1
            STATE.teacache_fallbacks += 1
            STATE.teacache_unavailable_reason = _short_error(exc)
            if STATE.teacache_logged_calls < 12:
                STATE.teacache_logged_calls += 1
                warning(f"teacache_fallback=reason={_short_error(exc)} route=original_Anima.{target_name}")
            return original_forward(self, x, timesteps, context, *args, **kwargs)

    setattr(anima_cls, target_name, teacache_forward)

    def restore() -> None:
        setattr(anima_cls, target_name, original_forward)

    STATE.patches[kind] = {"restore": restore}
    info(
        "applied experimental patch kind=teacache "
        f"target=backend.nn.anima.Anima.{target_name} "
        f"threshold={STATE.teacache_threshold:.4f} "
        f"progress={STATE.teacache_start_percent:.2f}..{STATE.teacache_end_percent:.2f} "
        f"cache_device={STATE.teacache_cache_device} source={STATE.teacache_modulated_source}"
    )
    return PatchResult(True, kind, "applied")


def _apply_ujicache_patch() -> PatchResult:
    kind = "ujicache"
    if is_patched(kind):
        return PatchResult(True, kind, "already patched")

    try:
        from backend.nn import anima
    except Exception as exc:
        return PatchResult(False, kind, f"import failed: {exc}")

    anima_cls = getattr(anima, "Anima", None)
    if anima_cls is None:
        return PatchResult(False, kind, "Anima class not found")

    target_name = "_forward"
    original_forward = getattr(anima_cls, target_name, None)
    if original_forward is None or not callable(original_forward):
        target_name = "forward"
        original_forward = getattr(anima_cls, target_name, None)
    if original_forward is None or not callable(original_forward):
        return PatchResult(False, kind, "Anima._forward/forward not found")

    def ujicache_forward(self, x, timesteps, context, *args, **kwargs):
        if not _should_ujicache_patch():
            return original_forward(self, x, timesteps, context, *args, **kwargs)
        try:
            fps, padding_mask, body_kwargs = _teacache_parse_forward_args(
                target_name,
                args,
                kwargs,
            )
            return _ujicache_forward_body(
                self,
                original_forward,
                x,
                timesteps,
                context,
                fps=fps,
                padding_mask=padding_mask,
                **body_kwargs,
            )
        except Exception as exc:
            STATE.ujicache_errors += 1
            STATE.ujicache_fallbacks += 1
            STATE.ujicache_unavailable_reason = _short_error(exc)
            if STATE.ujicache_logged_calls < 12:
                STATE.ujicache_logged_calls += 1
                warning(f"ujicache_fallback=reason={_short_error(exc)} route=original_Anima.{target_name}")
            return original_forward(self, x, timesteps, context, *args, **kwargs)

    setattr(anima_cls, target_name, ujicache_forward)

    def restore() -> None:
        setattr(anima_cls, target_name, original_forward)

    STATE.patches[kind] = {"restore": restore}
    info(
        "applied experimental patch kind=ujicache "
        f"target=backend.nn.anima.Anima.{target_name} "
        f"formula={STATE.ujicache_formula} "
        f"threshold={STATE.ujicache_threshold:.4f} "
        f"progress={STATE.ujicache_start_percent:.2f}..{STATE.ujicache_end_percent:.2f} "
        f"use_prediction_after={STATE.ujicache_use_prediction_after_progress:.2f} "
        f"apply_from_skip={STATE.ujicache_apply_prediction_from_skip} "
        f"prediction_strength={STATE.ujicache_prediction_strength:.2f} "
        f"taylor2_curve_strength={STATE.ujicache_taylor2_curve_strength:.2f} "
        f"cache_device={STATE.ujicache_cache_device} source={STATE.ujicache_modulated_source}"
    )
    return PatchResult(True, kind, "applied")


def _teacache_parse_forward_args(
    target_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[Any, Any, dict[str, Any]]:
    body_kwargs = dict(kwargs)
    fps = body_kwargs.pop("fps", None)
    padding_mask = body_kwargs.pop("padding_mask", None)
    if target_name == "_forward":
        if len(args) > 0:
            fps = args[0]
        if len(args) > 1:
            padding_mask = args[1]
        if len(args) > 2:
            raise RuntimeError("Anima._forward extra positional arguments are unsupported by TeaCache")
    else:
        if len(args) > 0:
            padding_mask = args[0]
        if len(args) > 1:
            raise RuntimeError("Anima.forward extra positional arguments are unsupported by TeaCache")
    return fps, padding_mask, body_kwargs


def _should_teacache_patch() -> bool:
    return STATE.active() and STATE.teacache_enabled


def _should_ujicache_patch() -> bool:
    return STATE.active() and STATE.ujicache_enabled


def _teacache_forward_body(
    model: Any,
    original_forward: Any,
    x: Any,
    timesteps: Any,
    context: Any,
    fps: Any = None,
    padding_mask: Any = None,
    **kwargs,
):
    import torch

    transformer_options = kwargs.get("transformer_options", {}) or {}
    cond_or_uncond = _teacache_cond_or_uncond(transformer_options.get("cond_or_uncond"))
    if not cond_or_uncond:
        raise RuntimeError("transformer_options.cond_or_uncond is missing")

    STATE.teacache_model_calls += 1
    _ensure_teacache_num_blocks(model)

    orig_shape = list(x.shape)
    x = _pad_to_patch_size_5d(
        x,
        (
            int(getattr(model, "patch_temporal", 1)),
            int(getattr(model, "patch_spatial", 1)),
            int(getattr(model, "patch_spatial", 1)),
        ),
    )
    x_B_C_T_H_W = x
    timesteps_B_T = timesteps
    crossattn_emb = context

    x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb = _teacache_prepare_embedded_sequence(
        model,
        x_B_C_T_H_W,
        fps,
        padding_mask,
    )

    if timesteps_B_T.ndim == 1:
        timesteps_B_T = timesteps_B_T.unsqueeze(1)

    t_embedding_B_T_D, adaln_lora_B_T_3D = model.t_embedder[1](
        model.t_embedder[0](timesteps_B_T).to(x_B_T_H_W_D.dtype)
    )
    t_embedding_B_T_D = model.t_embedding_norm(t_embedding_B_T_D)

    cache_device = _teacache_cache_device(x_B_T_H_W_D)
    modulated_inp = _teacache_modulated_input(
        model,
        t_embedding_B_T_D,
        adaln_lora_B_T_3D,
        cache_device,
    )
    cache = _teacache_state_for_model(model)
    batch_per_slot = _teacache_batch_per_slot(x_B_T_H_W_D, cond_or_uncond)
    step_index = max(0, STATE.denoiser_calls - 1)
    progress = _teacache_progress(step_index)

    rels: dict[Any, float | None] = {}
    slot_should_calc: dict[Any, bool] = {}
    for slot_index, key in enumerate(cond_or_uncond):
        key = int(key)
        item = _teacache_slot(cache, key)
        modulated_slice = modulated_inp[slot_index * batch_per_slot : (slot_index + 1) * batch_per_slot]
        rels[key] = _teacache_update_slot(item, modulated_slice)
        slot_should_calc[key] = bool(item["should_calc"])

    force_full_reason = _teacache_force_full_reason(
        cache,
        step_index,
        progress,
        cond_or_uncond,
    )
    should_calc = force_full_reason is not None or any(slot_should_calc.values())
    if STATE.teacache_dry_run and not should_calc:
        STATE.teacache_dry_run_skips += 1
        should_calc = True
        force_full_reason = "dry_run"

    block_kwargs = {
        "rope_emb_L_1_1_D": rope_emb_L_1_1_D.unsqueeze(1).unsqueeze(0),
        "adaln_lora_B_T_3D": adaln_lora_B_T_3D,
        "extra_per_block_pos_emb": extra_pos_emb,
        "transformer_options": transformer_options,
    }

    if x_B_T_H_W_D.dtype == torch.float16:
        x_B_T_H_W_D = x_B_T_H_W_D.float()

    if should_calc:
        ori_x = x_B_T_H_W_D.to(cache_device)
        for block in model.blocks:
            x_B_T_H_W_D = block(
                x_B_T_H_W_D,
                t_embedding_B_T_D,
                crossattn_emb,
                **block_kwargs,
            )
        residual = x_B_T_H_W_D.to(cache_device) - ori_x
        _dump_teacache_residual(
            residual,
            cond_or_uncond,
            batch_per_slot,
            step_index,
            timesteps_B_T,
            cache_device,
        )
        for slot_index, key in enumerate(cond_or_uncond):
            item = _teacache_slot(cache, int(key))
            item["previous_residual"] = residual[
                slot_index * batch_per_slot : (slot_index + 1) * batch_per_slot
            ]
            item["accumulated_rel_l1_distance"] = 0.0
            item["should_calc"] = True
        cache["skip_streak"] = 0
        STATE.teacache_full_calcs += 1
        if STATE.teacache_model_calls == 1:
            STATE.teacache_first_full_calcs += 1
        if force_full_reason:
            STATE.teacache_forced_full_calcs += 1
        _teacache_log_call("full", step_index, progress, rels, force_full_reason)
    else:
        _teacache_apply_residual(
            x_B_T_H_W_D,
            cache,
            cond_or_uncond,
            batch_per_slot,
        )
        cache["skip_streak"] = int(cache.get("skip_streak", 0)) + 1
        STATE.teacache_skips += 1
        _teacache_log_call("skip", step_index, progress, rels, None)

    x_B_T_H_W_O = model.final_layer(
        x_B_T_H_W_D.to(crossattn_emb.dtype),
        t_embedding_B_T_D,
        adaln_lora_B_T_3D=adaln_lora_B_T_3D,
    )
    return model.unpatchify(x_B_T_H_W_O)[
        :, :, : orig_shape[-3], : orig_shape[-2], : orig_shape[-1]
    ]


def _ujicache_forward_body(
    model: Any,
    original_forward: Any,
    x: Any,
    timesteps: Any,
    context: Any,
    fps: Any = None,
    padding_mask: Any = None,
    **kwargs,
):
    import torch

    transformer_options = kwargs.get("transformer_options", {}) or {}
    cond_or_uncond = _teacache_cond_or_uncond(transformer_options.get("cond_or_uncond"))
    if not cond_or_uncond:
        raise RuntimeError("transformer_options.cond_or_uncond is missing")

    STATE.ujicache_model_calls += 1
    _ensure_ujicache_num_blocks(model)

    orig_shape = list(x.shape)
    x = _pad_to_patch_size_5d(
        x,
        (
            int(getattr(model, "patch_temporal", 1)),
            int(getattr(model, "patch_spatial", 1)),
            int(getattr(model, "patch_spatial", 1)),
        ),
    )
    x_B_C_T_H_W = x
    timesteps_B_T = timesteps
    crossattn_emb = context

    x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb = _teacache_prepare_embedded_sequence(
        model,
        x_B_C_T_H_W,
        fps,
        padding_mask,
    )

    if timesteps_B_T.ndim == 1:
        timesteps_B_T = timesteps_B_T.unsqueeze(1)

    t_embedding_B_T_D, adaln_lora_B_T_3D = model.t_embedder[1](
        model.t_embedder[0](timesteps_B_T).to(x_B_T_H_W_D.dtype)
    )
    t_embedding_B_T_D = model.t_embedding_norm(t_embedding_B_T_D)

    cache_device = _ujicache_cache_device(x_B_T_H_W_D)
    modulated_inp = _ujicache_modulated_input(
        model,
        t_embedding_B_T_D,
        adaln_lora_B_T_3D,
        cache_device,
    )
    cache = _ujicache_state_for_model(model)
    batch_per_slot = _teacache_batch_per_slot(x_B_T_H_W_D, cond_or_uncond)
    step_index = max(0, STATE.denoiser_calls - 1)
    progress = _teacache_progress(step_index)

    rels: dict[Any, float | None] = {}
    slot_should_calc: dict[Any, bool] = {}
    for slot_index, key in enumerate(cond_or_uncond):
        key = int(key)
        item = _ujicache_slot(cache, key)
        modulated_slice = modulated_inp[slot_index * batch_per_slot : (slot_index + 1) * batch_per_slot]
        rels[key] = _ujicache_update_slot(item, modulated_slice)
        slot_should_calc[key] = bool(item["should_calc"])

    force_full_reason = _ujicache_force_full_reason(
        cache,
        step_index,
        progress,
        cond_or_uncond,
    )
    should_calc = force_full_reason is not None or any(slot_should_calc.values())
    if STATE.ujicache_dry_run and not should_calc:
        STATE.ujicache_dry_run_predictions += 1
        should_calc = True
        force_full_reason = "dry_run"

    block_kwargs = {
        "rope_emb_L_1_1_D": rope_emb_L_1_1_D.unsqueeze(1).unsqueeze(0),
        "adaln_lora_B_T_3D": adaln_lora_B_T_3D,
        "extra_per_block_pos_emb": extra_pos_emb,
        "transformer_options": transformer_options,
    }

    if x_B_T_H_W_D.dtype == torch.float16:
        x_B_T_H_W_D = x_B_T_H_W_D.float()

    if should_calc:
        ori_x = x_B_T_H_W_D.to(cache_device)
        for block in model.blocks:
            x_B_T_H_W_D = block(
                x_B_T_H_W_D,
                t_embedding_B_T_D,
                crossattn_emb,
                **block_kwargs,
            )
        residual = x_B_T_H_W_D.to(cache_device) - ori_x
        _dump_ujicache_residual(
            residual,
            cond_or_uncond,
            batch_per_slot,
            step_index,
            timesteps_B_T,
            cache_device,
        )
        for slot_index, key in enumerate(cond_or_uncond):
            item = _ujicache_slot(cache, int(key))
            start = slot_index * batch_per_slot
            end = (slot_index + 1) * batch_per_slot
            residual_slice = residual[start:end]
            item["previous_residual"] = residual_slice
            _ujicache_record_residual(item, step_index, residual_slice)
            item["accumulated_rel_l1_distance"] = 0.0
            item["should_calc"] = True
        cache["skip_streak"] = 0
        STATE.ujicache_full_calcs += 1
        if STATE.ujicache_model_calls == 1:
            STATE.ujicache_first_full_calcs += 1
        if force_full_reason:
            STATE.ujicache_forced_full_calcs += 1
        _ujicache_log_call(
            "full",
            step_index,
            progress,
            rels,
            reason=force_full_reason,
            late_phase=False,
            skip_streak=0,
            slot_actions={},
            slot_reasons={},
        )
    else:
        slot_actions, slot_reasons, late_phase, skip_streak = _ujicache_apply_residual(
            x_B_T_H_W_D,
            cache,
            cond_or_uncond,
            batch_per_slot,
            step_index,
            progress,
        )
        cache["skip_streak"] = skip_streak
        STATE.ujicache_skips += 1
        if any(action == "prediction" for action in slot_actions.values()):
            STATE.ujicache_prediction_used += 1
            decision = "prediction"
        else:
            STATE.ujicache_fallback_used += 1
            decision = "fallback"
        for reason in slot_reasons.values():
            if reason:
                STATE.ujicache_fallback_reasons[reason] = (
                    STATE.ujicache_fallback_reasons.get(reason, 0) + 1
                )
        _ujicache_log_call(
            decision,
            step_index,
            progress,
            rels,
            reason=None,
            late_phase=late_phase,
            skip_streak=skip_streak,
            slot_actions=slot_actions,
            slot_reasons=slot_reasons,
        )

    x_B_T_H_W_O = model.final_layer(
        x_B_T_H_W_D.to(crossattn_emb.dtype),
        t_embedding_B_T_D,
        adaln_lora_B_T_3D=adaln_lora_B_T_3D,
    )
    return model.unpatchify(x_B_T_H_W_O)[
        :, :, : orig_shape[-3], : orig_shape[-2], : orig_shape[-1]
    ]


def _ensure_teacache_num_blocks(model: Any) -> None:
    if STATE.teacache_num_blocks is not None:
        return
    blocks = getattr(model, "blocks", None)
    try:
        STATE.teacache_num_blocks = len(blocks)
    except Exception:
        STATE.teacache_num_blocks = _runtime_num_blocks()


def _ensure_ujicache_num_blocks(model: Any) -> None:
    if STATE.ujicache_num_blocks is not None:
        return
    blocks = getattr(model, "blocks", None)
    try:
        STATE.ujicache_num_blocks = len(blocks)
    except Exception:
        STATE.ujicache_num_blocks = _runtime_num_blocks()


def _teacache_prepare_embedded_sequence(model: Any, x: Any, fps: Any, padding_mask: Any):
    if fps is not None:
        try:
            return model.prepare_embedded_sequence(
                x,
                fps=fps,
                padding_mask=padding_mask,
            )
        except TypeError:
            pass
    return model.prepare_embedded_sequence(x, padding_mask=padding_mask)


def _pad_to_patch_size_5d(x: Any, patch_size: tuple[int, int, int]):
    if len(getattr(x, "shape", ())) != 5:
        raise RuntimeError(f"expected 5D latent tensor, got shape={_shape(x)}")
    try:
        from backend.utils import pad_to_patch_size
    except Exception:
        pad_to_patch_size = None
    if pad_to_patch_size is not None:
        return pad_to_patch_size(x, patch_size)

    import torch
    import torch.nn.functional as functional

    padding_mode = "reflect" if (torch.jit.is_tracing() or torch.jit.is_scripting()) else "circular"
    pad = ()
    for i in range(x.ndim - 2):
        size = max(1, int(patch_size[i]))
        pad = (0, (size - int(x.shape[i + 2]) % size) % size) + pad
    return functional.pad(x, pad, mode=padding_mode)


def _teacache_cache_device(x: Any):
    import torch

    if STATE.teacache_cache_device == TEACACHE_CACHE_DEVICE_CPU:
        return torch.device("cpu")
    return getattr(x, "device", torch.device("cpu"))


def _ujicache_cache_device(x: Any):
    import torch

    if STATE.ujicache_cache_device == TEACACHE_CACHE_DEVICE_CPU:
        return torch.device("cpu")
    return getattr(x, "device", torch.device("cpu"))


def _teacache_modulated_input(model: Any, t_embedding: Any, adaln_lora: Any, cache_device: Any):
    return _cache_modulated_input(
        model,
        t_embedding,
        adaln_lora,
        cache_device,
        STATE.teacache_modulated_source,
    )


def _ujicache_modulated_input(model: Any, t_embedding: Any, adaln_lora: Any, cache_device: Any):
    return _cache_modulated_input(
        model,
        t_embedding,
        adaln_lora,
        cache_device,
        STATE.ujicache_modulated_source,
    )


def _cache_modulated_input(
    model: Any,
    t_embedding: Any,
    adaln_lora: Any,
    cache_device: Any,
    source: str,
):
    if source != TEACACHE_SOURCE_FIRST_BLOCK_SHIFT:
        return t_embedding.to(cache_device)
    blocks = getattr(model, "blocks", None)
    if not blocks:
        raise RuntimeError("Anima blocks are unavailable")
    first_block = blocks[0]
    adaln = getattr(first_block, "adaln_modulation_self_attn", None)
    if not callable(adaln):
        raise RuntimeError("first block adaln_modulation_self_attn is unavailable")
    modulated = adaln(t_embedding)
    if adaln_lora is not None and bool(getattr(model, "use_adaln_lora", False)):
        modulated = modulated + adaln_lora
    return modulated.chunk(3, dim=-1)[0].to(cache_device)


def _teacache_state_for_model(model: Any) -> dict[str, Any]:
    cache = getattr(model, "_nzap_teacache_state", None)
    if not isinstance(cache, dict) or cache.get("generation_index") != STATE.generation_index:
        cache = {
            "generation_index": STATE.generation_index,
            "skip_streak": 0,
            "slots": {},
        }
        setattr(model, "_nzap_teacache_state", cache)
    return cache


def _ujicache_state_for_model(model: Any) -> dict[str, Any]:
    cache = getattr(model, "_nzap_ujicache_state", None)
    if not isinstance(cache, dict) or cache.get("generation_index") != STATE.generation_index:
        cache = {
            "generation_index": STATE.generation_index,
            "skip_streak": 0,
            "slots": {},
        }
        setattr(model, "_nzap_ujicache_state", cache)
    return cache


def _teacache_slot(cache: dict[str, Any], key: int) -> dict[str, Any]:
    slots = cache.setdefault("slots", {})
    if key not in slots:
        slots[key] = {
            "should_calc": True,
            "accumulated_rel_l1_distance": 0.0,
            "previous_modulated_input": None,
            "previous_residual": None,
        }
    return slots[key]


def _ujicache_slot(cache: dict[str, Any], key: int) -> dict[str, Any]:
    slots = cache.setdefault("slots", {})
    if key not in slots:
        slots[key] = {
            "should_calc": True,
            "accumulated_rel_l1_distance": 0.0,
            "previous_modulated_input": None,
            "previous_residual": None,
            "residual_history": [],
        }
    return slots[key]


def _teacache_cond_or_uncond(value: Any) -> list[int]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    try:
        return [int(item) for item in value]
    except Exception:
        return []


def _teacache_batch_per_slot(x: Any, cond_or_uncond: Any) -> int:
    total = int(x.shape[0])
    slots = len(cond_or_uncond)
    if slots <= 0 or total % slots != 0:
        raise RuntimeError(f"invalid cond_or_uncond={cond_or_uncond} for batch={total}")
    return total // slots


def _teacache_update_slot(slot: dict[str, Any], modulated_slice: Any) -> float | None:
    return _cache_update_slot(
        slot,
        modulated_slice,
        STATE.teacache_threshold,
        _teacache_coefficients(),
        "TeaCache",
    )


def _ujicache_update_slot(slot: dict[str, Any], modulated_slice: Any) -> float | None:
    return _cache_update_slot(
        slot,
        modulated_slice,
        STATE.ujicache_threshold,
        _ujicache_coefficients(),
        "UjiCache",
    )


def _cache_update_slot(
    slot: dict[str, Any],
    modulated_slice: Any,
    threshold: float,
    coefficients: list[float],
    label: str,
) -> float | None:
    import math

    previous = slot.get("previous_modulated_input")
    rel: float | None = None
    if previous is None:
        slot["should_calc"] = True
    else:
        try:
            denom = previous.abs().mean()
            if float(denom.item()) <= 0.0:
                raise RuntimeError("previous_modulated_input mean is zero")
            rel_tensor = (modulated_slice - previous).abs().mean() / denom
            rel = float(rel_tensor.item())
            if not math.isfinite(rel):
                raise RuntimeError(f"non-finite {label} rel_l1: {rel}")
            estimate = _cache_poly1d(rel, coefficients)
            if not math.isfinite(estimate) or estimate < 0.0:
                raise RuntimeError(f"invalid {label} estimate: {estimate}")
            accumulated = float(slot.get("accumulated_rel_l1_distance", 0.0)) + estimate
            if accumulated < threshold:
                slot["should_calc"] = False
                slot["accumulated_rel_l1_distance"] = accumulated
            else:
                slot["should_calc"] = True
                slot["accumulated_rel_l1_distance"] = 0.0
        except Exception:
            slot["should_calc"] = True
            slot["accumulated_rel_l1_distance"] = 0.0
    slot["previous_modulated_input"] = modulated_slice.detach()
    return rel


def _teacache_poly1d(value: float) -> float:
    return _cache_poly1d(value, _teacache_coefficients())


def _cache_poly1d(value: float, coefficients: list[float]) -> float:
    result = 0.0
    for coefficient in coefficients:
        result = result * value + float(coefficient)
    return result


def _teacache_coefficients() -> list[float]:
    if STATE.teacache_coefficient_profile == TEACACHE_PROFILE_IDENTITY:
        return [1.0, 0.0]
    if STATE.teacache_coefficient_profile == TEACACHE_PROFILE_ANIMA_2B_30STEP_FIRST_BLOCK_SHIFT:
        return TEACACHE_COEFFICIENTS_ANIMA_2B_30STEP_FIRST_BLOCK_SHIFT
    return TEACACHE_COEFFICIENTS_ANIMA_2B_30STEP_FIRST_BLOCK_SHIFT


def _ujicache_coefficients() -> list[float]:
    if STATE.ujicache_coefficient_profile == TEACACHE_PROFILE_IDENTITY:
        return [1.0, 0.0]
    if STATE.ujicache_coefficient_profile == TEACACHE_PROFILE_ANIMA_2B_30STEP_FIRST_BLOCK_SHIFT:
        return TEACACHE_COEFFICIENTS_ANIMA_2B_30STEP_FIRST_BLOCK_SHIFT
    return TEACACHE_COEFFICIENTS_ANIMA_2B_30STEP_FIRST_BLOCK_SHIFT


def _teacache_progress(step_index: int) -> float:
    steps = STATE.generation_steps
    if not steps or steps <= 1:
        return 0.0
    return max(0.0, min(1.0, step_index / float(steps - 1)))


def _teacache_force_full_reason(
    cache: dict[str, Any],
    step_index: int,
    progress: float,
    cond_or_uncond: Any,
) -> str | None:
    if STATE.teacache_model_calls == 1:
        return "first_call"
    for key in cond_or_uncond:
        if _teacache_slot(cache, int(key)).get("previous_residual") is None:
            return "missing_residual"
    if progress < STATE.teacache_start_percent or progress > STATE.teacache_end_percent:
        return "outside_progress"
    interval = STATE.teacache_force_full_interval
    if interval > 0 and step_index > 0 and step_index % interval == 0:
        return "force_full_interval"
    max_skip_streak = STATE.teacache_max_skip_streak
    if max_skip_streak > 0 and int(cache.get("skip_streak", 0)) >= max_skip_streak:
        return "max_skip_streak"
    return None


def _ujicache_force_full_reason(
    cache: dict[str, Any],
    step_index: int,
    progress: float,
    cond_or_uncond: Any,
) -> str | None:
    if STATE.ujicache_model_calls == 1:
        return "first_call"
    for key in cond_or_uncond:
        if _ujicache_slot(cache, int(key)).get("previous_residual") is None:
            return "missing_residual"
    if progress < STATE.ujicache_start_percent or progress > STATE.ujicache_end_percent:
        return "outside_progress"
    interval = STATE.ujicache_force_full_interval
    if interval > 0 and step_index > 0 and step_index % interval == 0:
        return "force_full_interval"
    max_skip_streak = STATE.ujicache_max_skip_streak
    if max_skip_streak > 0 and int(cache.get("skip_streak", 0)) >= max_skip_streak:
        return "max_skip_streak"
    return None


def _teacache_apply_residual(
    x: Any,
    cache: dict[str, Any],
    cond_or_uncond: Any,
    batch_per_slot: int,
) -> None:
    for slot_index, key in enumerate(cond_or_uncond):
        residual = _teacache_slot(cache, int(key)).get("previous_residual")
        if residual is None:
            raise RuntimeError(f"missing previous_residual for slot={key}")
        start = slot_index * batch_per_slot
        end = (slot_index + 1) * batch_per_slot
        if getattr(residual, "shape", None) != getattr(x[start:end], "shape", None):
            raise RuntimeError(
                f"residual shape mismatch slot={key} residual={_shape(residual)} target={_shape(x[start:end])}"
            )
        x[start:end] = x[start:end] + residual.to(x.device)


def _ujicache_apply_residual(
    x: Any,
    cache: dict[str, Any],
    cond_or_uncond: Any,
    batch_per_slot: int,
    step_index: int,
    progress: float,
) -> tuple[dict[int, str], dict[int, str | None], bool, int]:
    skip_streak = int(cache.get("skip_streak", 0)) + 1
    late_phase = progress > STATE.ujicache_use_prediction_after_progress
    slot_actions: dict[int, str] = {}
    slot_reasons: dict[int, str | None] = {}
    for slot_index, key in enumerate(cond_or_uncond):
        key = int(key)
        start = slot_index * batch_per_slot
        end = (slot_index + 1) * batch_per_slot
        target_slice = x[start:end]
        residual, action, reason = _ujicache_residual_for_slot(
            _ujicache_slot(cache, key),
            target_slice,
            step_index,
            skip_streak,
            late_phase,
        )
        x[start:end] = target_slice + residual.to(x.device)
        slot_actions[key] = action
        slot_reasons[key] = reason
    return slot_actions, slot_reasons, late_phase, skip_streak


def _ujicache_residual_for_slot(
    slot: dict[str, Any],
    target_slice: Any,
    step_index: int,
    skip_streak: int,
    late_phase: bool,
) -> tuple[Any, str, str | None]:
    previous = slot.get("previous_residual")
    if previous is None:
        raise RuntimeError("missing previous_residual")
    if getattr(previous, "shape", None) != getattr(target_slice, "shape", None):
        raise RuntimeError(
            f"residual shape mismatch residual={_shape(previous)} target={_shape(target_slice)}"
        )

    formula = STATE.ujicache_formula
    if formula == UJICACHE_FORMULA_TEACACHE:
        return previous.to(target_slice.device), "fallback", "formula"

    prediction_allowed = late_phase or skip_streak >= STATE.ujicache_apply_prediction_from_skip
    if not prediction_allowed:
        return previous.to(target_slice.device), "fallback", "streak"

    try:
        if formula == UJICACHE_FORMULA_LINEAR:
            prediction = _ujicache_predict_linear(slot, step_index, previous)
        elif formula == UJICACHE_FORMULA_TAYLOR2:
            prediction = _ujicache_predict_taylor2(slot, step_index, previous)
        else:
            return previous.to(target_slice.device), "fallback", "formula"
        prediction = _ujicache_validate_prediction(prediction, previous, target_slice)
        return prediction.to(target_slice.device), "prediction", None
    except _UjiCachePredictionFallback as exc:
        return previous.to(target_slice.device), "fallback", exc.reason
    except Exception:
        return previous.to(target_slice.device), "fallback", "prediction_error"


class _UjiCachePredictionFallback(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _ujicache_record_residual(slot: dict[str, Any], step_index: int, residual: Any) -> None:
    history = slot.setdefault("residual_history", [])
    history.append(
        {
            "step_index": int(step_index),
            "residual": residual.detach(),
        }
    )
    if len(history) > 5:
        del history[:-5]


def _ujicache_predict_linear(slot: dict[str, Any], step_index: int, previous: Any):
    history = _ujicache_residual_history(slot, 2)
    raw_prediction = _ujicache_lagrange_prediction(history[-2:], step_index)
    previous_f32 = previous.float()
    return previous_f32 + STATE.ujicache_prediction_strength * (raw_prediction - previous_f32)


def _ujicache_predict_taylor2(slot: dict[str, Any], step_index: int, previous: Any):
    history = _ujicache_residual_history(slot, 3)
    linear_prediction = _ujicache_lagrange_prediction(history[-2:], step_index)
    quadratic_prediction = _ujicache_lagrange_prediction(history[-3:], step_index)
    curve = STATE.ujicache_taylor2_curve_strength
    raw_prediction = (1.0 - curve) * linear_prediction + curve * quadratic_prediction
    previous_f32 = previous.float()
    return previous_f32 + STATE.ujicache_prediction_strength * (raw_prediction - previous_f32)


def _ujicache_residual_history(slot: dict[str, Any], count: int) -> list[dict[str, Any]]:
    history = slot.get("residual_history") or []
    if len(history) < count:
        raise _UjiCachePredictionFallback("insufficient_history")
    recent = history[-count:]
    reference_shape = getattr(recent[-1].get("residual"), "shape", None)
    for item in recent:
        if getattr(item.get("residual"), "shape", None) != reference_shape:
            raise _UjiCachePredictionFallback("shape_mismatch")
    return recent


def _ujicache_lagrange_prediction(history: list[dict[str, Any]], step_index: int):
    if not history:
        raise _UjiCachePredictionFallback("insufficient_history")
    times = [float(item["step_index"]) for item in history]
    target = float(step_index)
    result = None
    for i, item in enumerate(history):
        weight = 1.0
        for j, other_time in enumerate(times):
            if i == j:
                continue
            denom = times[i] - other_time
            if abs(denom) < 1e-6:
                raise _UjiCachePredictionFallback("duplicate_history_step")
            weight *= (target - other_time) / denom
        residual = item["residual"].float()
        weighted = residual * weight
        result = weighted if result is None else result + weighted
    if result is None:
        raise _UjiCachePredictionFallback("insufficient_history")
    return result


def _ujicache_validate_prediction(prediction: Any, previous: Any, target_slice: Any):
    import math
    import torch

    if getattr(prediction, "shape", None) != getattr(previous, "shape", None):
        raise _UjiCachePredictionFallback("shape_mismatch")
    if getattr(prediction, "shape", None) != getattr(target_slice, "shape", None):
        raise _UjiCachePredictionFallback("shape_mismatch")
    if not bool(torch.isfinite(prediction).all().item()):
        raise _UjiCachePredictionFallback("numeric_error")

    prediction_norm = float(torch.linalg.vector_norm(prediction.float()).item())
    previous_norm = float(torch.linalg.vector_norm(previous.float()).item())
    if not math.isfinite(prediction_norm) or not math.isfinite(previous_norm):
        raise _UjiCachePredictionFallback("numeric_error")
    if previous_norm > 0.0 and prediction_norm > previous_norm * UJICACHE_MAX_NORM_RATIO:
        raise _UjiCachePredictionFallback("norm_guard")
    try:
        return prediction.to(device=previous.device, dtype=previous.dtype)
    except Exception as exc:
        raise _UjiCachePredictionFallback("dtype_conversion") from exc


def _dump_teacache_residual(
    residual: Any,
    cond_or_uncond: list[int],
    batch_per_slot: int,
    step_index: int,
    timestep: Any,
    cache_device: Any,
) -> None:
    if not (
        STATE.tensor_dump_active()
        and STATE.dump_teacache_residual
        and STATE.teacache_enabled
    ):
        return
    from .tensor_dump import dump_tensor

    for slot_index, key in enumerate(cond_or_uncond):
        start = slot_index * batch_per_slot
        end = start + batch_per_slot
        local_call_index = STATE.tensor_dump_teacache_local_call_index
        STATE.tensor_dump_teacache_local_call_index += 1
        dump_tensor(
            "teacache_residual",
            residual[start:end],
            logical_step_index=step_index,
            local_call_index=local_call_index,
            call_index=local_call_index,
            slot=int(key),
            decision="full",
            timestep_value=timestep,
            teacache_model_call=STATE.teacache_model_calls,
            extra={"cache_device": str(cache_device)},
        )


def _dump_ujicache_residual(
    residual: Any,
    cond_or_uncond: list[int],
    batch_per_slot: int,
    step_index: int,
    timestep: Any,
    cache_device: Any,
) -> None:
    if not (
        STATE.tensor_dump_active()
        and STATE.dump_ujicache_residual
        and STATE.ujicache_enabled
    ):
        return
    from .tensor_dump import dump_tensor

    for slot_index, key in enumerate(cond_or_uncond):
        start = slot_index * batch_per_slot
        end = start + batch_per_slot
        local_call_index = STATE.tensor_dump_ujicache_local_call_index
        STATE.tensor_dump_ujicache_local_call_index += 1
        dump_tensor(
            "ujicache_residual",
            residual[start:end],
            logical_step_index=step_index,
            local_call_index=local_call_index,
            call_index=local_call_index,
            slot=int(key),
            decision="full",
            timestep_value=timestep,
            extra={
                "cache_device": str(cache_device),
                "ujicache_model_call": STATE.ujicache_model_calls,
                "formula": STATE.ujicache_formula,
            },
        )


def _teacache_log_call(
    decision: str,
    step_index: int,
    progress: float,
    rels: dict[Any, float | None],
    reason: str | None,
) -> None:
    if not STATE.teacache_verbose_trace and STATE.teacache_logged_calls >= 12:
        return
    STATE.teacache_logged_calls += 1
    rel_text = ",".join(
        f"{key}:{'None' if value is None else f'{value:.6f}'}"
        for key, value in sorted(rels.items())
    )
    info(
        "teacache_call="
        f"call={STATE.teacache_model_calls} step={step_index} "
        f"progress={progress:.3f} decision={decision} "
        f"reason={reason or 'threshold'} rel_l1={rel_text} "
        f"threshold={STATE.teacache_threshold:.4f} dry_run={STATE.teacache_dry_run}"
    )


def _ujicache_log_call(
    decision: str,
    step_index: int,
    progress: float,
    rels: dict[Any, float | None],
    reason: str | None,
    late_phase: bool,
    skip_streak: int,
    slot_actions: dict[int, str],
    slot_reasons: dict[int, str | None],
) -> None:
    if not STATE.ujicache_verbose_trace and STATE.ujicache_logged_calls >= 12:
        return
    STATE.ujicache_logged_calls += 1
    rel_text = ",".join(
        f"{key}:{'None' if value is None else f'{value:.6f}'}"
        for key, value in sorted(rels.items())
    )
    action_text = ",".join(
        f"{key}:{slot_actions[key]}" for key in sorted(slot_actions)
    ) or "None"
    reason_text = ",".join(
        f"{key}:{slot_reasons[key]}" for key in sorted(slot_reasons) if slot_reasons[key]
    ) or (reason or "threshold")
    info(
        "ujicache_call="
        f"call={STATE.ujicache_model_calls} step={step_index} "
        f"progress={progress:.3f} late={late_phase} streak={skip_streak} "
        f"decision={decision} reason={reason_text} "
        f"formula={STATE.ujicache_formula} action={action_text} rel_l1={rel_text} "
        f"threshold={STATE.ujicache_threshold:.4f} "
        f"use_prediction_after={STATE.ujicache_use_prediction_after_progress:.2f} "
        f"apply_from_skip={STATE.ujicache_apply_prediction_from_skip} "
        f"prediction_strength={STATE.ujicache_prediction_strength:.2f} "
        f"taylor2_curve_strength={STATE.ujicache_taylor2_curve_strength:.2f} "
        f"dry_run={STATE.ujicache_dry_run}"
    )


def _apply_tensor_dump_patch() -> PatchResult:
    kind = "tensor_dump"
    if is_patched(kind):
        return PatchResult(True, kind, "already patched")

    try:
        from backend.nn import anima
    except Exception as exc:
        return PatchResult(False, kind, f"import failed: {exc}")

    anima_cls = getattr(anima, "Anima", None)
    block_cls = getattr(anima, "Block", None)
    attention_cls = getattr(anima, "SelfCrossAttention", None)
    if anima_cls is None or block_cls is None or attention_cls is None:
        return PatchResult(False, kind, "Anima/Block/SelfCrossAttention not found")

    target_name = "_forward"
    original_anima_forward = getattr(anima_cls, target_name, None)
    if original_anima_forward is None or not callable(original_anima_forward):
        target_name = "forward"
        original_anima_forward = getattr(anima_cls, target_name, None)
    if original_anima_forward is None or not callable(original_anima_forward):
        return PatchResult(False, kind, "Anima._forward/forward not found")

    original_block_forward = block_cls.forward
    original_compute_attention = attention_cls.compute_attention
    mlp_restores = _install_tensor_dump_mlp_wrappers()

    def dumped_anima_forward(self, x, timesteps, context, *args, **kwargs):
        if not _should_tensor_dump_patch():
            return original_anima_forward(self, x, timesteps, context, *args, **kwargs)
        previous_context = STATE.tensor_dump_current_context
        current_context = dict(previous_context or {})
        current_context["timestep_value"] = timesteps
        STATE.tensor_dump_current_context = current_context
        try:
            return original_anima_forward(self, x, timesteps, context, *args, **kwargs)
        finally:
            STATE.tensor_dump_current_context = previous_context

    def dumped_block_forward(self, x_B_T_H_W_D, *args, **kwargs):
        if not _should_tensor_dump_patch():
            return original_block_forward(self, x_B_T_H_W_D, *args, **kwargs)

        _ensure_tensor_dump_num_blocks()
        block_call_index = STATE.tensor_dump_block_call_index
        STATE.tensor_dump_block_call_index += 1
        block_index = _tensor_dump_block_index(block_call_index)
        previous_context = STATE.tensor_dump_current_context
        current_context = dict(previous_context or {})
        current_context.update(
            {
                "block_call_index": block_call_index,
                "block_index": block_index,
                "logical_step_index": max(0, STATE.denoiser_calls - 1),
            }
        )
        STATE.tensor_dump_current_context = current_context
        try:
            output = original_block_forward(self, x_B_T_H_W_D, *args, **kwargs)
            if STATE.dump_block_output:
                local_call_index = STATE.tensor_dump_block_local_call_index
                STATE.tensor_dump_block_local_call_index += 1
                _dump_tensor_from_context(
                    "block_output",
                    output,
                    local_call_index=local_call_index,
                    call_index=local_call_index,
                    context=current_context,
                )
            return output
        finally:
            STATE.tensor_dump_current_context = previous_context

    def dumped_compute_attention(self, q, k, v, transformer_options=None):
        output = original_compute_attention(
            self,
            q,
            k,
            v,
            transformer_options=transformer_options or {},
        )
        if (
            _should_tensor_dump_patch()
            and STATE.dump_cross_attention_output
            and not getattr(self, "is_SelfAttn", False)
        ):
            local_call_index = STATE.tensor_dump_cross_attention_local_call_index
            STATE.tensor_dump_cross_attention_local_call_index += 1
            _dump_tensor_from_context(
                "cross_attention_output",
                output,
                local_call_index=local_call_index,
                call_index=local_call_index,
                context=STATE.tensor_dump_current_context or {},
                attn_type="cross",
            )
        return output

    setattr(anima_cls, target_name, dumped_anima_forward)
    block_cls.forward = dumped_block_forward
    attention_cls.compute_attention = dumped_compute_attention

    def restore() -> None:
        setattr(anima_cls, target_name, original_anima_forward)
        block_cls.forward = original_block_forward
        attention_cls.compute_attention = original_compute_attention
        for module, original_forward in reversed(mlp_restores):
            try:
                module.forward = original_forward
            except Exception:
                pass

    STATE.patches[kind] = {"restore": restore}
    info(
        "applied diagnostic patch kind=tensor_dump "
        f"target=Anima.{target_name}/Block.forward/SelfCrossAttention.compute_attention"
    )
    return PatchResult(True, kind, "applied")


def _should_tensor_dump_patch() -> bool:
    return (
        STATE.tensor_dump_block_level_active()
        and not STATE.teacache_enabled
        and not STATE.spectrum_enabled
        and not STATE.sparse_enabled
        and not STATE.attention_override_active()
    )


def _ensure_tensor_dump_num_blocks() -> None:
    if STATE.tensor_dump_num_blocks is not None:
        return
    STATE.tensor_dump_num_blocks = _runtime_num_blocks()


def _tensor_dump_block_index(call_index: int) -> int | None:
    num_blocks = STATE.tensor_dump_num_blocks
    if not num_blocks:
        return None
    return call_index % num_blocks


def _dump_tensor_from_context(
    tensor_type: str,
    tensor: Any,
    *,
    local_call_index: int,
    call_index: int,
    context: dict[str, Any],
    attn_type: str | None = None,
) -> None:
    from .tensor_dump import dump_tensor

    block_index = context.get("block_index")
    if not isinstance(block_index, int):
        block_index = None
    block_call_index = context.get("block_call_index")
    if not isinstance(block_call_index, int):
        block_call_index = None
    logical_step_index = context.get("logical_step_index")
    if not isinstance(logical_step_index, int):
        logical_step_index = max(0, STATE.denoiser_calls - 1)
    dump_tensor(
        tensor_type,
        tensor,
        logical_step_index=logical_step_index,
        local_call_index=local_call_index,
        call_index=call_index,
        block_call_index=block_call_index,
        block_index=block_index,
        timestep_value=context.get("timestep_value"),
        attn_type=attn_type,
    )


def _install_tensor_dump_mlp_wrappers() -> list[tuple[Any, Any]]:
    restores: list[tuple[Any, Any]] = []
    if not STATE.dump_mlp_output:
        return restores
    blocks = _runtime_blocks()
    if not blocks:
        _tensor_dump_warn_once("mlp_no_blocks", "mlp_dump_unavailable reason=blocks_not_found")
        return restores
    found = 0
    for block_index, block in enumerate(blocks):
        module = _find_mlp_module(block)
        if module is None:
            continue
        original_forward = getattr(module, "forward", None)
        if not callable(original_forward):
            continue

        def dumped_mlp_forward(*args, _original=original_forward, _block_index=block_index, **kwargs):
            output = _original(*args, **kwargs)
            if _should_tensor_dump_patch() and STATE.dump_mlp_output:
                local_call_index = STATE.tensor_dump_mlp_local_call_index
                STATE.tensor_dump_mlp_local_call_index += 1
                context = dict(STATE.tensor_dump_current_context or {})
                context.setdefault("block_index", _block_index)
                context.setdefault("logical_step_index", max(0, STATE.denoiser_calls - 1))
                _dump_tensor_from_context(
                    "mlp_output",
                    output,
                    local_call_index=local_call_index,
                    call_index=local_call_index,
                    context=context,
                )
            return output

        module.forward = dumped_mlp_forward
        restores.append((module, original_forward))
        found += 1
    if found == 0:
        _tensor_dump_warn_once("mlp_module_not_found", "mlp_dump_unavailable reason=mlp_module_not_found")
    return restores


def _find_mlp_module(block: Any) -> Any | None:
    candidate_names = (
        "mlp",
        "ffn",
        "ff",
        "feed_forward",
        "feedforward",
        "feed_forward_layer",
    )
    for name in candidate_names:
        module = getattr(block, name, None)
        if callable(getattr(module, "forward", None)):
            return module
    named_children = getattr(block, "named_children", None)
    if callable(named_children):
        try:
            for name, module in named_children():
                lowered = str(name).lower()
                class_name = type(module).__name__.lower()
                if any(token in lowered for token in candidate_names) or any(
                    token in class_name for token in ("mlp", "feedforward", "ffn")
                ):
                    if callable(getattr(module, "forward", None)):
                        return module
        except Exception:
            return None
    return None


def _apply_sparse_attention_patch() -> PatchResult:
    kind = "sparse_attention"
    if is_patched(kind):
        return PatchResult(True, kind, "already patched")

    try:
        from backend.nn import anima
    except Exception as exc:
        return PatchResult(False, kind, f"import failed: {exc}")

    block_cls = getattr(anima, "Block", None)
    attention_cls = getattr(anima, "SelfCrossAttention", None)
    if block_cls is None or attention_cls is None:
        return PatchResult(False, kind, "Anima Block/SelfCrossAttention not found")

    original_block_forward = block_cls.forward
    original_compute_attention = attention_cls.compute_attention

    def sparse_block_forward(self, x_B_T_H_W_D, *args, **kwargs):
        if not _should_sparse_patch():
            return original_block_forward(self, x_B_T_H_W_D, *args, **kwargs)

        _ensure_sparse_num_blocks()
        call_index = STATE.sparse_block_calls
        STATE.sparse_block_calls += 1
        block_index = _sparse_block_index(call_index)
        previous_context = STATE.sparse_current_context
        STATE.sparse_current_context = _sparse_context(x_B_T_H_W_D, block_index)
        try:
            return original_block_forward(self, x_B_T_H_W_D, *args, **kwargs)
        finally:
            STATE.sparse_current_context = previous_context

    def sparse_compute_attention(self, q, k, v, transformer_options=None):
        if not _should_replace_attention(self, q, k, v):
            return original_compute_attention(
                self,
                q,
                k,
                v,
                transformer_options=transformer_options or {},
            )

        context = STATE.sparse_current_context or {}
        try:
            result = _compute_sparse_attention(q, k, v, context)
            STATE.sparse_attention_calls += 1
            if STATE.sparse_logged_calls < 8:
                STATE.sparse_logged_calls += 1
                info(
                    "sparse_attention_call="
                    f"call={STATE.sparse_attention_calls} block={context.get('block_index')} "
                    f"step={_current_sparse_step()} backend={STATE.sparse_backend} "
                    f"local_window={STATE.sparse_local_window} dilation={STATE.sparse_dilation} "
                    f"q_shape={_shape(q)} result_shape={_shape(result)}"
                )
            return self.output_dropout(self.output_proj(result))
        except Exception as exc:
            STATE.sparse_errors += 1
            STATE.sparse_fallbacks += 1
            if STATE.sparse_logged_calls < 8:
                STATE.sparse_logged_calls += 1
                warning(
                    "sparse_attention_fallback="
                    f"reason={exc} block={context.get('block_index')} backend={STATE.sparse_backend}"
                )
            return original_compute_attention(
                self,
                q,
                k,
                v,
                transformer_options=transformer_options or {},
            )

    block_cls.forward = sparse_block_forward
    attention_cls.compute_attention = sparse_compute_attention

    def restore() -> None:
        block_cls.forward = original_block_forward
        attention_cls.compute_attention = original_compute_attention

    STATE.patches[kind] = {"restore": restore}
    info("applied experimental patch kind=sparse_attention target=Anima self-attention")
    return PatchResult(True, kind, "applied")


def _should_sparse_patch() -> bool:
    return STATE.active() and STATE.sparse_enabled


def _ensure_sparse_num_blocks() -> None:
    if STATE.sparse_num_blocks is not None:
        return
    STATE.sparse_num_blocks = _runtime_num_blocks()


def _sparse_block_index(call_index: int) -> int | str:
    num_blocks = STATE.sparse_num_blocks
    if not num_blocks:
        return "unknown"
    return call_index % num_blocks


def _sparse_context(x_B_T_H_W_D: Any, block_index: int | str) -> dict[str, Any] | None:
    shape = getattr(x_B_T_H_W_D, "shape", None)
    if shape is None or len(shape) != 5:
        return None
    _, t, height, width, _ = shape
    return {
        "block_index": block_index,
        "t": int(t),
        "height": int(height),
        "width": int(width),
    }


def _should_replace_attention(attn_module: Any, q: Any, k: Any, v: Any) -> bool:
    if not _should_sparse_patch():
        return False
    if not getattr(attn_module, "is_SelfAttn", False):
        return False
    context = STATE.sparse_current_context
    if not context:
        return False
    block_index = context.get("block_index")
    if not isinstance(block_index, int):
        return False
    if not (STATE.sparse_block_start <= block_index <= STATE.sparse_block_end):
        return False
    step = _current_sparse_step()
    if step < STATE.sparse_step_start:
        return False
    if STATE.sparse_step_end >= 0 and step > STATE.sparse_step_end:
        return False
    interval = STATE.sparse_full_attention_interval
    if interval > 0 and step % interval == 0:
        return False
    if STATE.sparse_backend == SPARSE_BACKEND_NATTEN and not _natten_ready():
        return False
    if context.get("t") != 1:
        STATE.sparse_fallbacks += 1
        return False
    if getattr(q, "shape", None) != getattr(k, "shape", None) or getattr(q, "shape", None) != getattr(v, "shape", None):
        return False
    return True


def _current_sparse_step() -> int:
    return max(0, STATE.denoiser_calls - 1)


def _compute_sparse_attention(q: Any, k: Any, v: Any, context: dict[str, Any]):
    if STATE.sparse_backend == SPARSE_BACKEND_TORCH:
        from .sparse import local_attention_2d_torch

        return local_attention_2d_torch(
            q,
            k,
            v,
            int(context["height"]),
            int(context["width"]),
            STATE.sparse_local_window,
            STATE.sparse_dilation,
        )
    if STATE.sparse_backend == SPARSE_BACKEND_NATTEN:
        from .sparse import local_attention_2d_natten

        message = _natten_unavailable_message()
        if message:
            raise RuntimeError(message)
        return local_attention_2d_natten(
            q,
            k,
            v,
            int(context["height"]),
            int(context["width"]),
            STATE.sparse_local_window,
            STATE.sparse_dilation,
        )
    raise RuntimeError(f"unknown sparse backend: {STATE.sparse_backend}")


def _natten_ready() -> bool:
    message = _natten_unavailable_message()
    if message:
        if STATE.sparse_unavailable_reason is None:
            STATE.sparse_unavailable_reason = message
            STATE.sparse_fallbacks += 1
            warning(f"sparse_attention_unavailable backend={STATE.sparse_backend} reason={message}")
        return False
    return True


def _natten_unavailable_message() -> str:
    from .sparse import natten_status

    status = natten_status()
    if not status["available"]:
        return f"NATTEN unavailable: {status['reason']}"
    return ""
