from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

from .logging import exception, info, warning
from .state import MODE_IDENTITY_PATCH, STATE


@dataclass
class PatchResult:
    ok: bool
    kind: str
    message: str = ""


def apply_patch(kind: str, context: Any = None) -> PatchResult:
    if kind == "cond_batch_trace":
        return _apply_cond_batch_trace_patch()
    if kind == "block_structure_trace":
        return _apply_block_structure_trace_patch()
    if kind == "block_forward_identity":
        return _apply_block_forward_identity_patch()
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
                "route=Nz-fast-anima->original_Block.forward"
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
    try:
        from modules import shared

        sd_model = getattr(shared, "sd_model", None)
        forge_objects = getattr(sd_model, "forge_objects", None)
        unet = getattr(forge_objects, "unet", None)
        model = getattr(unet, "model", None)
        diffusion_model = getattr(model, "diffusion_model", model)
        blocks = getattr(diffusion_model, "blocks", None)
        if blocks is None:
            return None
        return len(blocks)
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
