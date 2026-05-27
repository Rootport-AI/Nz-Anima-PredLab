from __future__ import annotations

import gradio as gr
import modules.scripts as scripts

from .callbacks import register_callbacks
from .diagnostics import log_generation_start, log_timing_summary
from .logging import exception
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
    STATE,
    TEACACHE_CACHE_DEVICE_CUDA,
    TEACACHE_CACHE_DEVICES,
    TEACACHE_COEFFICIENT_PROFILES,
    TEACACHE_MODULATED_SOURCES,
    TEACACHE_PRESET_BALANCED,
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
            mode = gr.Dropdown(
                label="Debug log mode",
                choices=MODES,
                value=_default_option("nzap_mode", MODE_OFF),
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
            with gr.Accordion("Attention", open=False, elem_id="nzap-attention-panel"):
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
                    maximum=0.3,
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

    if not getattr(STATE.model_detection, "supported", False):
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
        apply_patch("block_forward_identity")
        return

    remove_patch("block_forward_identity")
    if STATE.teacache_enabled:
        remove_patch("block_structure_trace")
        remove_patch("attention_kernel")
        remove_patch("sparse_attention")
        apply_patch("teacache")
    elif STATE.sparse_enabled:
        remove_patch("teacache")
        remove_patch("block_structure_trace")
        remove_patch("attention_kernel")
        apply_patch("sparse_attention")
    else:
        remove_patch("teacache")
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


def _default_option(key: str, default):
    try:
        from modules import shared

        return getattr(shared.opts, key, default)
    except Exception:
        return default
