from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

from .logging import exception, info, warning
from .state import (
    ATTENTION_BACKEND_CURRENT,
    ATTENTION_TARGET_BOTH,
    ATTENTION_TARGET_CROSS,
    ATTENTION_TARGET_SELF,
    MODE_IDENTITY_PATCH,
    SPARSE_BACKEND_NATTEN,
    SPARSE_BACKEND_TORCH,
    STATE,
)


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
    if kind == "attention_kernel":
        return _apply_attention_kernel_patch()
    if kind == "sparse_attention":
        return _apply_sparse_attention_patch()
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

    backend_fn = _attention_backend_function(STATE.attention_backend)
    if backend_fn is None:
        return PatchResult(False, kind, f"attention backend not found: {STATE.attention_backend}")

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
                info(
                    "attention_kernel_call="
                    f"call={STATE.attention_kernel_calls} block={context.get('block_index')} "
                    f"type={attn_type} backend={STATE.attention_backend} "
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
    try:
        from backend import attention
    except Exception:
        return None
    return getattr(attention, name, None)


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
    return backend_fn(
        q_bhsd,
        k_bhsd,
        v_bhsd,
        in_q_shape[-2],
        skip_reshape=True,
        transformer_options=transformer_options,
    )


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
