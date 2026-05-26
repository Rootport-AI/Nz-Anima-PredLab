from __future__ import annotations

from typing import Any


def _safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    try:
        return str(value)
    except Exception:
        return f"<unprintable {type(value).__name__}>"


def processing_info(p: Any) -> dict[str, Any]:
    width = _safe_getattr(p, "width")
    height = _safe_getattr(p, "height")
    return {
        "sampler": _safe_str(_safe_getattr(p, "sampler_name", "")),
        "scheduler": _safe_str(_safe_getattr(p, "scheduler", "")),
        "steps": _safe_getattr(p, "steps"),
        "cfg_scale": _safe_getattr(p, "cfg_scale"),
        "width": width,
        "height": height,
        "is_img2img": _safe_getattr(p, "is_img2img", False),
        "enable_hr": _safe_getattr(p, "enable_hr", False),
    }


def attention_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "attention_backend": "unknown",
        "sage_available": False,
        "flash_available": False,
        "xformers_available": False,
        "pytorch_available": False,
        "anima_attention_path": False,
    }
    try:
        from backend import attention

        fn = _safe_getattr(attention, "attention_function")
        info["attention_backend"] = _safe_getattr(fn, "__name__", _safe_str(fn))
        for name in dir(attention):
            lname = name.lower()
            if "sage" in lname:
                info["sage_available"] = True
            if "flash" in lname:
                info["flash_available"] = True
            if "xformers" in lname:
                info["xformers_available"] = True
            if "pytorch" in lname or "sdpa" in lname:
                info["pytorch_available"] = True
    except Exception as exc:
        info["attention_error"] = _safe_str(exc)

    try:
        from backend.nn import anima

        info["anima_attention_path"] = all(
            hasattr(anima, attr) for attr in ("SelfCrossAttention", "Block")
        )
    except Exception as exc:
        info["anima_error"] = _safe_str(exc)

    return info


def lowbit_info(sd_model: Any) -> dict[str, Any]:
    info: dict[str, Any] = {}
    try:
        from backend.args import dynamic_args

        ops = _safe_getattr(dynamic_args, "ops")
        info["forge_ops"] = type(ops).__name__ if ops is not None else ""
    except Exception as exc:
        info["forge_ops_error"] = _safe_str(exc)

    for name in ("storage_dtype", "computation_dtype", "dtype"):
        value = _safe_getattr(sd_model, name)
        if value is not None:
            info[f"sd_model_{name}"] = _safe_str(value)

    try:
        forge_objects = _safe_getattr(sd_model, "forge_objects")
        unet = None
        if isinstance(forge_objects, dict):
            unet = forge_objects.get("unet")
        else:
            unet = _safe_getattr(forge_objects, "unet")
        model = _safe_getattr(unet, "model")
        diffusion_model = _safe_getattr(model, "diffusion_model")
        for name in ("storage_dtype", "computation_dtype", "dtype"):
            value = _safe_getattr(diffusion_model, name)
            if value is not None:
                info[f"diffusion_model_{name}"] = _safe_str(value)
    except Exception as exc:
        info["dtype_probe_error"] = _safe_str(exc)

    return info


def cond_info(params: Any) -> dict[str, Any]:
    denoiser = _safe_getattr(params, "denoiser")
    transformer_options = _safe_getattr(params, "transformer_options", {})
    if transformer_options is None:
        transformer_options = {}

    def opt(name: str) -> Any:
        if isinstance(transformer_options, dict):
            return transformer_options.get(name)
        return _safe_getattr(transformer_options, name)

    return {
        "text_uncond_is_none": _safe_getattr(params, "text_uncond") is None,
        "cfg_scale": _safe_getattr(params, "cond_scale", _safe_getattr(params, "cfg_scale")),
        "step": _safe_getattr(denoiser, "step"),
        "total_steps": _safe_getattr(denoiser, "total_steps"),
        "cond_or_uncond": opt("cond_or_uncond"),
        "cond_indices": opt("cond_indices"),
        "uncond_indices": opt("uncond_indices"),
    }
