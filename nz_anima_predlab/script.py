from __future__ import annotations

import gradio as gr
import modules.scripts as scripts

from .callbacks import register_callbacks
from .diagnostics import log_generation_start, log_timing_summary
from .logging import exception, warning
from .model_detect import detect_model
from .patcher import apply_patch, remove_patch
from .state import (
    ATTENTION_BACKEND_CURRENT,
    ATTENTION_BACKENDS,
    ATTENTION_TARGET_BOTH,
    ATTENTION_TARGETS,
    MODE_DIAGNOSE,
    MODE_IDENTITY_PATCH,
    MODE_OFF,
    MODES,
    SPARSE_BACKEND_NATTEN,
    SPARSE_BACKENDS,
    SPECTRUM_PRESET_AGGRESSIVE,
    SPECTRUM_PRESET_BALANCED,
    SPECTRUM_PRESET_CUSTOM,
    SPECTRUM_PRESET_SAFE,
    SPECTRUM_PRESETS,
    STATE,
    TEACACHE_CACHE_DEVICE_CUDA,
    TEACACHE_CACHE_DEVICES,
    TEACACHE_COEFFICIENT_PROFILES,
    TEACACHE_MODULATED_SOURCES,
    TEACACHE_PRESET_AGGRESSIVE,
    TEACACHE_PRESET_BALANCED,
    TEACACHE_PRESET_CUSTOM,
    TEACACHE_PRESET_SAFE,
    TEACACHE_PRESETS,
    TEACACHE_PROFILE_ANIMA_2B_30STEP_FIRST_BLOCK_SHIFT,
    TEACACHE_SOURCE_FIRST_BLOCK_SHIFT,
)
from .timing import start_sampling

register_callbacks()


class Script(scripts.Script):
    def title(self):
        return "Nz-Anima-PredLab"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion("Nz-Anima-PredLab", open=False, elem_id="nzap-panel"):
            enabled = gr.Checkbox(
                label="Enable Nz-Anima-PredLab",
                value=_default_option("nzap_enable", False),
                elem_id="nzap-enable",
            )
            with gr.Accordion("Debug log mode", open=False, elem_id="nzap-debug-panel"):
                debug_log_enabled = gr.Checkbox(
                    label="Enable debug log mode",
                    value=_default_option("nzap_debug_log_enable", True),
                    elem_id="nzap-debug-enable",
                )
                mode = gr.Dropdown(
                    label="Debug log",
                    choices=MODES,
                    value=_default_mode_option(),
                    elem_id="nzap-mode",
                )
                print_timing_log = gr.Checkbox(
                    label="Print timing log",
                    value=_default_option("nzap_print_timing_log", True),
                    elem_id="nzap-print-timing-log",
                )
                verbose_diagnose_log = gr.Checkbox(
                    label="Verbose diagnose log",
                    value=_default_option("nzap_verbose_diagnose_log", False),
                    elem_id="nzap-verbose-diagnose-log",
                )
                debug_log_enabled.change(
                    fn=_enable_parent_if_child_enabled,
                    inputs=[debug_log_enabled],
                    outputs=[enabled],
                )
                mode.change(
                    fn=_debug_mode_selection_updates,
                    inputs=[mode],
                    outputs=[enabled, debug_log_enabled],
                )
            with gr.Accordion("Attention", open=False, elem_id="nzap-attention-panel"):
                attention_enabled = gr.Checkbox(
                    label="Enable attention backend override",
                    value=_default_option("nzap_attention_enable", False),
                    elem_id="nzap-attention-enable",
                )
                attention_backend = gr.Dropdown(
                    label="Attention backend",
                    choices=ATTENTION_BACKENDS,
                    value=ATTENTION_BACKEND_CURRENT,
                    elem_id="nzap-attention-backend",
                )
                attention_target = gr.Radio(
                    label="Attention target",
                    choices=ATTENTION_TARGETS,
                    value=ATTENTION_TARGET_BOTH,
                    elem_id="nzap-attention-target",
                )
                attention_block_start = gr.Slider(
                    label="Attention block start",
                    minimum=0,
                    maximum=27,
                    step=1,
                    value=0,
                    elem_id="nzap-attention-block-start",
                )
                attention_block_end = gr.Slider(
                    label="Attention block end",
                    minimum=0,
                    maximum=27,
                    step=1,
                    value=27,
                    elem_id="nzap-attention-block-end",
                )
                attention_enabled.change(
                    fn=_enable_parent_if_child_enabled,
                    inputs=[attention_enabled],
                    outputs=[enabled],
                )
            with gr.Accordion("TeaCache", open=False, elem_id="nzap-teacache-panel"):
                teacache_enabled = gr.Checkbox(
                    label="Enable TeaCache experiment",
                    value=False,
                    elem_id="nzap-teacache-enable",
                )
                teacache_preset = gr.Dropdown(
                    label="TeaCache preset",
                    choices=TEACACHE_PRESETS,
                    value=TEACACHE_PRESET_BALANCED,
                    elem_id="nzap-teacache-preset",
                )
                teacache_threshold = gr.Slider(
                    label="Rel L1 threshold",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.005,
                    value=0.07,
                    elem_id="nzap-teacache-threshold",
                )
                teacache_start_percent = gr.Slider(
                    label="Start progress",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    value=0.05,
                    elem_id="nzap-teacache-start-percent",
                )
                teacache_end_percent = gr.Slider(
                    label="End progress",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    value=0.95,
                    elem_id="nzap-teacache-end-percent",
                )
                teacache_cache_device = gr.Radio(
                    label="Cache device",
                    choices=TEACACHE_CACHE_DEVICES,
                    value=TEACACHE_CACHE_DEVICE_CUDA,
                    elem_id="nzap-teacache-cache-device",
                )
                teacache_modulated_source = gr.Dropdown(
                    label="Modulated source",
                    choices=TEACACHE_MODULATED_SOURCES,
                    value=TEACACHE_SOURCE_FIRST_BLOCK_SHIFT,
                    elem_id="nzap-teacache-modulated-source",
                )
                teacache_coefficient_profile = gr.Dropdown(
                    label="Coefficient profile",
                    choices=TEACACHE_COEFFICIENT_PROFILES,
                    value=TEACACHE_PROFILE_ANIMA_2B_30STEP_FIRST_BLOCK_SHIFT,
                    elem_id="nzap-teacache-coefficient-profile",
                )
                teacache_max_skip_streak = gr.Slider(
                    label="Max skip streak (0 = off)",
                    minimum=0,
                    maximum=64,
                    step=1,
                    value=0,
                    elem_id="nzap-teacache-max-skip-streak",
                )
                teacache_force_full_interval = gr.Slider(
                    label="Force full interval (0 = off)",
                    minimum=0,
                    maximum=64,
                    step=1,
                    value=0,
                    elem_id="nzap-teacache-force-full-interval",
                )
                teacache_dry_run = gr.Checkbox(
                    label="Dry run",
                    value=False,
                    elem_id="nzap-teacache-dry-run",
                )
                teacache_verbose_trace = gr.Checkbox(
                    label="Verbose TeaCache trace",
                    value=False,
                    elem_id="nzap-teacache-verbose-trace",
                )
                teacache_preset.change(
                    fn=_teacache_preset_updates,
                    inputs=[teacache_preset],
                    outputs=[
                        teacache_threshold,
                        teacache_start_percent,
                        teacache_end_percent,
                    ],
                )
                for control in (
                    teacache_threshold,
                    teacache_start_percent,
                    teacache_end_percent,
                ):
                    control.change(
                        fn=_teacache_mark_custom,
                        inputs=[],
                        outputs=[teacache_preset],
                    )
            with gr.Accordion("Spectrum", open=False, elem_id="nzap-spectrum-panel"):
                spectrum_enabled = gr.Checkbox(
                    label="Enable Spectrum experiment",
                    value=False,
                    elem_id="nzap-spectrum-enable",
                )
                spectrum_preset = gr.Dropdown(
                    label="Spectrum preset",
                    choices=SPECTRUM_PRESETS,
                    value=SPECTRUM_PRESET_BALANCED,
                    elem_id="nzap-spectrum-preset",
                )
                spectrum_w = gr.Slider(
                    label="Prediction weighting",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    value=0.20,
                    elem_id="nzap-spectrum-w",
                )
                spectrum_m = gr.Slider(
                    label="Polynomial degree",
                    minimum=1,
                    maximum=32,
                    step=1,
                    value=16,
                    elem_id="nzap-spectrum-m",
                )
                spectrum_lambda = gr.Slider(
                    label="Ridge lambda",
                    minimum=0.0,
                    maximum=100.0,
                    step=0.01,
                    value=0.50,
                    elem_id="nzap-spectrum-lambda",
                )
                spectrum_warmup_steps = gr.Slider(
                    label="Warmup steps",
                    minimum=0,
                    maximum=50,
                    step=1,
                    value=6,
                    elem_id="nzap-spectrum-warmup-steps",
                )
                spectrum_window_size = gr.Slider(
                    label="Window size",
                    minimum=1,
                    maximum=64,
                    step=1,
                    value=2,
                    elem_id="nzap-spectrum-window-size",
                )
                spectrum_flex_window = gr.Slider(
                    label="Flex window",
                    minimum=0.0,
                    maximum=2.0,
                    step=0.01,
                    value=0.0,
                    elem_id="nzap-spectrum-flex-window",
                )
                spectrum_stop_progress = gr.Slider(
                    label="Stop progress",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    value=0.80,
                    elem_id="nzap-spectrum-stop-progress",
                )
                spectrum_dry_run = gr.Checkbox(
                    label="Dry run",
                    value=False,
                    elem_id="nzap-spectrum-dry-run",
                )
                spectrum_verbose_trace = gr.Checkbox(
                    label="Verbose Spectrum trace",
                    value=False,
                    elem_id="nzap-spectrum-verbose-trace",
                )
                spectrum_preset.change(
                    fn=_spectrum_preset_updates,
                    inputs=[spectrum_preset],
                    outputs=[
                        spectrum_w,
                        spectrum_m,
                        spectrum_lambda,
                        spectrum_warmup_steps,
                        spectrum_window_size,
                        spectrum_flex_window,
                        spectrum_stop_progress,
                    ],
                )
                for control in (
                    spectrum_w,
                    spectrum_m,
                    spectrum_lambda,
                    spectrum_warmup_steps,
                    spectrum_window_size,
                    spectrum_flex_window,
                    spectrum_stop_progress,
                ):
                    control.change(
                        fn=_spectrum_mark_custom,
                        inputs=[],
                        outputs=[spectrum_preset],
                    )
                teacache_enabled.change(
                    fn=_teacache_enable_updates,
                    inputs=[teacache_enabled],
                    outputs=[spectrum_enabled, enabled],
                )
                spectrum_enabled.change(
                    fn=_spectrum_enable_updates,
                    inputs=[spectrum_enabled],
                    outputs=[teacache_enabled, enabled],
                )
            with gr.Accordion("2D Sparse", open=False, elem_id="nzap-sparse-panel"):
                sparse_enabled = gr.Checkbox(
                    label="Enable 2D sparse attention",
                    value=False,
                    elem_id="nzap-sparse-enable",
                )
                sparse_backend = gr.Radio(
                    label="Sparse backend",
                    choices=SPARSE_BACKENDS,
                    value=SPARSE_BACKEND_NATTEN,
                    elem_id="nzap-sparse-backend",
                )
                sparse_block_start = gr.Slider(
                    label="Block start",
                    minimum=0,
                    maximum=27,
                    step=1,
                    value=14,
                    elem_id="nzap-sparse-block-start",
                )
                sparse_block_end = gr.Slider(
                    label="Block end",
                    minimum=0,
                    maximum=27,
                    step=1,
                    value=27,
                    elem_id="nzap-sparse-block-end",
                )
                sparse_step_start = gr.Slider(
                    label="Step start",
                    minimum=0,
                    maximum=150,
                    step=1,
                    value=0,
                    elem_id="nzap-sparse-step-start",
                )
                sparse_step_end = gr.Slider(
                    label="Step end (-1 = last)",
                    minimum=-1,
                    maximum=150,
                    step=1,
                    value=-1,
                    elem_id="nzap-sparse-step-end",
                )
                sparse_local_window = gr.Slider(
                    label="Local attention window",
                    minimum=3,
                    maximum=63,
                    step=2,
                    value=15,
                    elem_id="nzap-sparse-local-window",
                )
                sparse_dilation = gr.Slider(
                    label="Dilation",
                    minimum=1,
                    maximum=8,
                    step=1,
                    value=1,
                    elem_id="nzap-sparse-dilation",
                )
                sparse_full_attention_interval = gr.Slider(
                    label="Full attention interval (0 = off)",
                    minimum=0,
                    maximum=64,
                    step=1,
                    value=0,
                    elem_id="nzap-sparse-full-attention-interval",
                )
                sparse_enabled.change(
                    fn=_enable_parent_if_child_enabled,
                    inputs=[sparse_enabled],
                    outputs=[enabled],
                )
            with gr.Accordion("Cond / Uncond", open=False, elem_id="nzap-cond-panel"):
                cond_uncond_enabled = gr.Checkbox(
                    label="Enable cond/uncond optimization",
                    value=False,
                    elem_id="nzap-cond-enable",
                )
                cond_uncond_skip_cfg1 = gr.Checkbox(
                    label="Skip uncond when CFG=1",
                    value=False,
                    elem_id="nzap-cond-skip-cfg1",
                )
                cond_uncond_schedule_enabled = gr.Checkbox(
                    label="Guidance step schedule",
                    value=False,
                    elem_id="nzap-cond-schedule",
                )
                cond_uncond_guidance_interval = gr.Slider(
                    label="Guidance interval",
                    minimum=1,
                    maximum=64,
                    step=1,
                    value=1,
                    elem_id="nzap-cond-guidance-interval",
                )
                cond_uncond_enabled.change(
                    fn=_enable_parent_if_child_enabled,
                    inputs=[cond_uncond_enabled],
                    outputs=[enabled],
                )
            with gr.Accordion("Low-bit / Compile", open=False, elem_id="nzap-lowbit-panel"):
                lowbit_enabled = gr.Checkbox(
                    label="Enable Nz low-bit experiment",
                    value=False,
                    elem_id="nzap-lowbit-enable",
                )
                compile_enabled = gr.Checkbox(
                    label="Enable torch.compile experiment",
                    value=False,
                    elem_id="nzap-compile-enable",
                )
                gr.Markdown(
                    "Reload the model after changing settings that require model reload.",
                    elem_id="nzap-reload-note",
                )
                lowbit_enabled.change(
                    fn=_enable_parent_if_child_enabled,
                    inputs=[lowbit_enabled],
                    outputs=[enabled],
                )
                compile_enabled.change(
                    fn=_enable_parent_if_child_enabled,
                    inputs=[compile_enabled],
                    outputs=[enabled],
                )
        return [
            enabled,
            mode,
            print_timing_log,
            verbose_diagnose_log,
            attention_backend,
            attention_target,
            attention_block_start,
            attention_block_end,
            sparse_enabled,
            sparse_backend,
            sparse_block_start,
            sparse_block_end,
            sparse_step_start,
            sparse_step_end,
            sparse_local_window,
            sparse_dilation,
            sparse_full_attention_interval,
            cond_uncond_enabled,
            cond_uncond_skip_cfg1,
            cond_uncond_schedule_enabled,
            cond_uncond_guidance_interval,
            lowbit_enabled,
            compile_enabled,
            teacache_enabled,
            teacache_preset,
            teacache_threshold,
            teacache_start_percent,
            teacache_end_percent,
            teacache_cache_device,
            teacache_modulated_source,
            teacache_coefficient_profile,
            teacache_max_skip_streak,
            teacache_force_full_interval,
            teacache_dry_run,
            teacache_verbose_trace,
            spectrum_enabled,
            spectrum_preset,
            spectrum_w,
            spectrum_m,
            spectrum_lambda,
            spectrum_warmup_steps,
            spectrum_window_size,
            spectrum_flex_window,
            spectrum_stop_progress,
            spectrum_dry_run,
            spectrum_verbose_trace,
            debug_log_enabled,
            attention_enabled,
        ]

    def process_before_every_sampling(self, p, *script_args, **kwargs):
        try:
            _begin_generation(p, script_args, "process_before_every_sampling")
        except Exception as exc:
            STATE.set_error(f"process_before_every_sampling failed: {exc}")
            exception("process_before_every_sampling failed")

    def postprocess(self, p, processed, *script_args):
        try:
            log_timing_summary()
        except Exception as exc:
            STATE.set_error(f"postprocess failed: {exc}")
            exception("postprocess failed")


def _apply_ui_args(script_args) -> None:
    if len(script_args) >= 48:
        STATE.apply_options(*script_args[:48])
        return
    if len(script_args) >= 46:
        STATE.apply_options(*script_args[:46])
        return
    if len(script_args) >= 35:
        STATE.apply_options(*script_args[:35])
        return
    if len(script_args) >= 23:
        STATE.apply_options(*script_args[:23])
        return
    if len(script_args) >= 4:
        STATE.apply_options(*script_args[:4])
        return
    STATE.refresh_settings()


def _begin_generation(p, script_args, source: str) -> None:
    _apply_ui_args(script_args)
    if not STATE.active():
        _remove_generation_patches()
        return

    start_sampling(source)
    try:
        STATE.generation_steps = int(getattr(p, "steps", 0) or 0) or None
    except Exception:
        STATE.generation_steps = None

    try:
        from modules import shared

        STATE.model_detection = detect_model(getattr(shared, "sd_model", None))
    except Exception:
        if STATE.model_detection is None:
            raise

    if not getattr(STATE.model_detection, "supported", False) and _requires_supported_model():
        _remove_generation_patches()
        log_generation_start(p)
        return

    _configure_generation_patches()
    log_generation_start(p)


def _configure_generation_patches() -> None:
    if STATE.mode == MODE_IDENTITY_PATCH:
        remove_patch("block_structure_trace")
        remove_patch("attention_kernel")
        remove_patch("sparse_attention")
        remove_patch("teacache")
        remove_patch("spectrum")
        apply_patch("block_forward_identity")
        return

    remove_patch("block_forward_identity")
    if STATE.teacache_enabled:
        remove_patch("block_structure_trace")
        remove_patch("attention_kernel")
        remove_patch("sparse_attention")
        remove_patch("spectrum")
        result = apply_patch("teacache")
        if not result.ok:
            STATE.teacache_unavailable_reason = result.message
            warning(f"teacache_patch_unavailable reason={result.message}")
    elif STATE.spectrum_enabled:
        remove_patch("teacache")
        remove_patch("block_structure_trace")
        remove_patch("sparse_attention")
        result = apply_patch("spectrum")
        if not result.ok:
            STATE.spectrum_unavailable_reason = result.message
            warning(f"spectrum_patch_unavailable reason={result.message}")
        if STATE.attention_override_active():
            apply_patch("attention_kernel")
        else:
            remove_patch("attention_kernel")
    elif STATE.sparse_enabled:
        remove_patch("teacache")
        remove_patch("spectrum")
        remove_patch("block_structure_trace")
        remove_patch("attention_kernel")
        apply_patch("sparse_attention")
    else:
        remove_patch("teacache")
        remove_patch("spectrum")
        remove_patch("sparse_attention")
        if STATE.attention_override_active():
            remove_patch("block_structure_trace")
            apply_patch("attention_kernel")
        else:
            remove_patch("attention_kernel")

    if (
        STATE.mode == MODE_DIAGNOSE
        and STATE.verbose_diagnose_log
        and not STATE.teacache_enabled
        and not STATE.spectrum_enabled
        and not STATE.sparse_enabled
        and not STATE.attention_override_active()
    ):
        apply_patch("block_structure_trace")
    else:
        remove_patch("block_structure_trace")


def _remove_generation_patches() -> None:
    remove_patch("block_structure_trace")
    remove_patch("block_forward_identity")
    remove_patch("attention_kernel")
    remove_patch("sparse_attention")
    remove_patch("teacache")
    remove_patch("spectrum")


def _requires_supported_model() -> bool:
    return (
        STATE.mode == MODE_IDENTITY_PATCH
        or STATE.teacache_enabled
        or STATE.sparse_enabled
        or STATE.attention_override_active()
    )


def _default_option(key: str, default):
    try:
        from modules import shared

        return getattr(shared.opts, key, default)
    except Exception:
        return default


def _default_mode_option() -> str:
    mode = str(_default_option("nzap_mode", MODE_OFF) or MODE_OFF)
    if mode == "Off":
        return MODE_OFF
    if mode == "Identity patch test":
        return MODE_IDENTITY_PATCH
    return mode if mode in MODES else MODE_OFF


def _teacache_preset_updates(preset: str):
    if preset == TEACACHE_PRESET_SAFE:
        return 0.06, 0.05, 0.95
    if preset == TEACACHE_PRESET_BALANCED:
        return 0.07, 0.05, 0.95
    if preset == TEACACHE_PRESET_AGGRESSIVE:
        return 0.08, 0.05, 0.95
    return gr.update(), gr.update(), gr.update()


def _teacache_mark_custom():
    return TEACACHE_PRESET_CUSTOM


def _spectrum_preset_updates(preset: str):
    if preset == SPECTRUM_PRESET_SAFE:
        return 0.20, 8, 0.50, 8, 2, 0.0, 0.80
    if preset == SPECTRUM_PRESET_BALANCED:
        return 0.20, 16, 0.50, 6, 2, 0.0, 0.80
    if preset == SPECTRUM_PRESET_AGGRESSIVE:
        return 0.30, 16, 0.50, 6, 2, 0.0, 0.90
    return (
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
    )


def _spectrum_mark_custom():
    return SPECTRUM_PRESET_CUSTOM


def _teacache_enable_updates(child_enabled: bool):
    if child_enabled:
        return False, True
    return gr.update(), gr.update()


def _spectrum_enable_updates(child_enabled: bool):
    if child_enabled:
        return False, True
    return gr.update(), gr.update()


def _enable_parent_if_child_enabled(child_enabled: bool):
    if child_enabled:
        return True
    return gr.update()


def _debug_mode_selection_updates(mode: str):
    if _mode_selected(mode):
        return True, True
    return gr.update(), gr.update()


def _mode_selected(mode: str) -> bool:
    value = str(mode or MODE_OFF)
    return value not in (MODE_OFF, "Off")
