from __future__ import annotations

from typing import Any

from . import __version__
from .forge_introspection import attention_info, cond_info, lowbit_info, processing_info
from .logging import info, warning
from .model_detect import ModelDetection
from .state import (
    MODE_DIAGNOSE,
    STATE,
)
from .timing import timing_summary


def _fmt(value: Any) -> str:
    if value is None:
        return "None"
    return str(value)


def log_generation_start(p: Any) -> None:
    if not STATE.active():
        return

    detection = STATE.model_detection
    proc = processing_info(p)
    info(
        f"version={__version__} enabled={STATE.enabled} mode={STATE.mode} "
        f"status={STATE.status} source={STATE.generation_start_source}"
    )

    if isinstance(detection, ModelDetection):
        info(
            "model_supported="
            f"{detection.supported} confidence={detection.confidence} "
            f"family={detection.family} reason={detection.reason}"
        )
        if STATE.verbose_diagnose_log:
            info(f"model_evidence={detection.evidence}")
        if not detection.supported:
            key = detection.key or detection.reason
            if key not in STATE.warned_model_keys:
                warning(f"unsupported model: {detection.reason}")
                STATE.warned_model_keys.add(key)
            STATE.status = "unsupported"

    info(
        "sampler="
        f"{proc['sampler']} scheduler={proc['scheduler']} steps={proc['steps']} "
        f"cfg={proc['cfg_scale']} resolution={proc['width']}x{proc['height']}"
    )

    log_attention_trace()
    log_lowbit_trace()

    STATE.generation_logged = True


def log_attention_trace() -> None:
    data = attention_info()
    info(f"attention_backend={data.get('attention_backend')}")
    info(
        "sage_available="
        f"{data.get('sage_available')} flash_available={data.get('flash_available')} "
        f"xformers_available={data.get('xformers_available')} "
        f"pytorch_available={data.get('pytorch_available')}"
    )
    info(f"anima_attention_path={data.get('anima_attention_path')}")
    if "attention_error" in data:
        warning(f"attention_trace_error={data['attention_error']}")
    if "anima_error" in data:
        warning(f"anima_trace_error={data['anima_error']}")


def log_lowbit_trace() -> None:
    data = lowbit_info(_current_sd_model())
    if not data:
        info("lowbit_info=unavailable")
        return
    info(" ".join(f"{key}={_fmt(value)}" for key, value in data.items()))


def log_cond_trace(params: Any) -> None:
    if not STATE.active():
        return
    if STATE.mode == "Off":
        return
    if STATE.cond_trace_logged and not STATE.verbose_diagnose_log:
        return
    data = cond_info(params)
    info(
        "cfg="
        f"{data.get('cfg_scale')} uncond_present={not data.get('text_uncond_is_none')} "
        f"sampling_step={data.get('sampling_step')}/{data.get('total_sampling_steps')} "
        f"denoiser_step={data.get('denoiser_step')}/{data.get('denoiser_total_steps')}"
    )
    info(
        "cond_or_uncond="
        f"{data.get('cond_or_uncond')} cond_indices={data.get('cond_indices')} "
        f"uncond_indices={data.get('uncond_indices')} "
        f"stage={data.get('transformer_options_stage')}"
    )
    info(
        "cond_shapes="
        f"x={data.get('x_shape')} sigma={data.get('sigma_shape')} "
        f"text_cond_type={data.get('text_cond_type')} "
        f"text_uncond_type={data.get('text_uncond_type')}"
    )
    STATE.cond_trace_logged = True


def log_timing_summary() -> None:
    if not STATE.active():
        return
    if not STATE.print_timing_log and STATE.mode != MODE_DIAGNOSE:
        return
    data = timing_summary()
    total = data["total_sampling_time"]
    avg = data["avg_step_time"]
    min_step = data["min_step_time"]
    max_step = data["max_step_time"]
    info(
        "denoiser_calls="
        f"{data['denoiser_calls']} avg_step_time={_seconds(avg)} "
        f"total_sampling_time={_seconds(total)} min_step_time={_seconds(min_step)} "
        f"max_step_time={_seconds(max_step)} status={STATE.status}"
    )


def _seconds(value: float | int | None) -> str:
    if value is None:
        return "None"
    return f"{float(value):.3f}s"


def _current_sd_model() -> Any:
    try:
        from modules import shared

        return getattr(shared, "sd_model", None)
    except Exception:
        return None
