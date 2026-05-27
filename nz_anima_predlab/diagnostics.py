from __future__ import annotations

from typing import Any

from . import __version__
from .forge_introspection import (
    attention_info,
    cond_info,
    lowbit_info,
    model_structure_info,
    processing_info,
)
from .logging import info, warning
from .model_detect import ModelDetection
from .state import (
    MODE_DIAGNOSE,
    MODE_IDENTITY_PATCH,
    SPARSE_BACKEND_NATTEN,
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
    if STATE.generation_steps is None:
        try:
            STATE.generation_steps = int(proc["steps"])
        except Exception:
            STATE.generation_steps = None

    log_attention_trace()
    log_lowbit_trace()
    log_model_structure_trace()
    log_experiment_snapshot()

    STATE.generation_logged = True


def log_experiment_snapshot() -> None:
    if STATE.attention_override_active() and not STATE.teacache_enabled and not STATE.sparse_enabled:
        info(
            "attention_kernel_config="
            f"enabled=True backend={STATE.attention_backend} "
            f"target={STATE.attention_target} "
            f"blocks={STATE.attention_block_start}..{STATE.attention_block_end}"
        )
    if STATE.sparse_enabled and not STATE.teacache_enabled:
        info(
            "sparse_config="
            f"enabled=True backend={STATE.sparse_backend} "
            f"blocks={STATE.sparse_block_start}..{STATE.sparse_block_end} "
            f"steps={STATE.sparse_step_start}..{STATE.sparse_step_end} "
            f"local_window={STATE.sparse_local_window} "
            f"dilation={STATE.sparse_dilation} "
            f"full_attention_interval={STATE.sparse_full_attention_interval}"
        )
        if STATE.sparse_backend == SPARSE_BACKEND_NATTEN:
            try:
                from .sparse import natten_status

                status = natten_status()
                info(
                    "natten_status="
                    f"available={status['available']} version={status['version']} "
                    f"reason={status['reason']}"
                )
            except Exception as exc:
                warning(f"natten_status_error={exc}")
    if STATE.teacache_enabled:
        info(
            "teacache_config="
            f"enabled=True preset={STATE.teacache_preset} "
            f"threshold={STATE.teacache_threshold:.4f} "
            f"progress={STATE.teacache_start_percent:.2f}..{STATE.teacache_end_percent:.2f} "
            f"cache_device={STATE.teacache_cache_device} "
            f"source={STATE.teacache_modulated_source} "
            f"coefficient_profile={STATE.teacache_coefficient_profile} "
            f"max_skip_streak={STATE.teacache_max_skip_streak} "
            f"force_full_interval={STATE.teacache_force_full_interval} "
            f"dry_run={STATE.teacache_dry_run}"
        )
    if STATE.cond_uncond_enabled:
        info(
            "cond_uncond_config="
            f"enabled=True skip_cfg1={STATE.cond_uncond_skip_cfg1} "
            f"schedule={STATE.cond_uncond_schedule_enabled} "
            f"guidance_interval={STATE.cond_uncond_guidance_interval}"
        )
    if STATE.lowbit_enabled or STATE.compile_enabled:
        info(
            "lowbit_compile_config="
            f"lowbit_enabled={STATE.lowbit_enabled} "
            f"compile_enabled={STATE.compile_enabled} "
            "reload_note=reload_model_after_changing_reload_required_settings"
        )


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


def log_model_structure_trace() -> None:
    data = model_structure_info(_current_sd_model())
    if not data:
        info("model_structure=unavailable")
        return
    info("model_structure=" + " ".join(f"{key}={_fmt(value)}" for key, value in data.items()))


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
    if (
        not STATE.print_timing_log
        and STATE.mode not in (MODE_DIAGNOSE, MODE_IDENTITY_PATCH)
        and not STATE.experimental_active()
    ):
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
    if STATE.mode == MODE_IDENTITY_PATCH:
        try:
            from .patcher import is_patched

            active = is_patched("block_forward_identity")
        except Exception:
            active = "unknown"
        info(
            "identity_patch_summary="
            f"calls={STATE.identity_patch_calls} "
            f"num_blocks={STATE.identity_patch_num_blocks} "
            f"logged_calls={STATE.identity_patch_logged_calls} "
            f"shape_mismatches={STATE.identity_patch_shape_mismatches} "
            f"errors={STATE.identity_patch_errors} active={active} "
            "target=backend.nn.anima.Block.forward behavior=call_original"
        )
    if STATE.sparse_enabled and not STATE.teacache_enabled:
        active = _is_patch_active("sparse_attention")
        info(
            "sparse_summary="
            f"block_calls={STATE.sparse_block_calls} "
            f"attention_calls={STATE.sparse_attention_calls} "
            f"fallbacks={STATE.sparse_fallbacks} errors={STATE.sparse_errors} "
            f"num_blocks={STATE.sparse_num_blocks} active={active} "
            f"backend={STATE.sparse_backend} "
            f"unavailable_reason={_fmt(STATE.sparse_unavailable_reason)}"
        )
    if STATE.attention_override_active() and not STATE.teacache_enabled and not STATE.sparse_enabled:
        active = _is_patch_active("attention_kernel")
        info(
            "attention_kernel_summary="
            f"calls={STATE.attention_kernel_calls} "
            f"block_calls={STATE.attention_kernel_block_calls} "
            f"fallbacks={STATE.attention_kernel_fallbacks} "
            f"errors={STATE.attention_kernel_errors} "
            f"internal_fallbacks={STATE.attention_kernel_internal_fallbacks} "
            f"internal_errors={STATE.attention_kernel_internal_errors} "
            f"actual_backends={_fmt_counts(STATE.attention_kernel_actual_counts)} "
            f"num_blocks={STATE.attention_kernel_num_blocks} "
            f"active={active} backend={STATE.attention_backend} "
            f"target={STATE.attention_target} "
            f"blocks={STATE.attention_block_start}..{STATE.attention_block_end}"
        )
    if STATE.teacache_enabled:
        active = _is_patch_active("teacache")
        total_decisions = STATE.teacache_full_calcs + STATE.teacache_skips
        skip_rate = (
            STATE.teacache_skips / total_decisions
            if total_decisions
            else 0.0
        )
        info(
            "teacache_summary="
            f"model_calls={STATE.teacache_model_calls} "
            f"full_calcs={STATE.teacache_full_calcs} "
            f"skips={STATE.teacache_skips} "
            f"dry_run_skips={STATE.teacache_dry_run_skips} "
            f"skip_rate={skip_rate:.3f} "
            f"first_full_calcs={STATE.teacache_first_full_calcs} "
            f"forced_full_calcs={STATE.teacache_forced_full_calcs} "
            f"fallbacks={STATE.teacache_fallbacks} "
            f"errors={STATE.teacache_errors} "
            f"num_blocks={STATE.teacache_num_blocks} "
            f"active={active} "
            f"dry_run={STATE.teacache_dry_run} "
            f"unavailable_reason={_fmt(STATE.teacache_unavailable_reason)}"
        )


def _seconds(value: float | int | None) -> str:
    if value is None:
        return "None"
    return f"{float(value):.3f}s"


def _fmt_counts(value: dict[str, int]) -> str:
    if not value:
        return "None"
    return ",".join(f"{key}:{value[key]}" for key in sorted(value))


def _current_sd_model() -> Any:
    try:
        from modules import shared

        return getattr(shared, "sd_model", None)
    except Exception:
        return None


def _is_patch_active(kind: str) -> Any:
    try:
        from .patcher import is_patched

        return is_patched(kind)
    except Exception:
        return "unknown"
