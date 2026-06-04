from __future__ import annotations

import gradio as gr
import modules.scripts as scripts

from .auto_ujicache import (
    AutoUjiCsvError,
    apply_auto_ujicache_row_to_state,
    parse_auto_ujicache_csv,
)
from .callbacks import register_callbacks
from .diagnostics import log_generation_start, log_timing_summary
from .logging import error, exception, info, warning
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
    UJICACHE_FORMULA_LINEAR,
    UJICACHE_FORMULA_TEACACHE,
    UJICACHE_FORMULA_TAYLOR2,
    UJICACHE_FORMULAS,
    UJICACHE_PRESET_CUSTOM,
    UJICACHE_PRESETS,
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
                dump_teacache_residual = gr.Checkbox(
                    label="Dump TeaCache residual",
                    value=False,
                    elem_id="nzap-dump-teacache-residual",
                )
                dump_ujicache_residual = gr.Checkbox(
                    label="Dump UjiCache residual",
                    value=False,
                    elem_id="nzap-dump-ujicache-residual",
                )
                dump_block_output = gr.Checkbox(
                    label="Dump block output",
                    value=False,
                    elem_id="nzap-dump-block-output",
                )
                dump_cross_attention_output = gr.Checkbox(
                    label="Dump cross-attention output",
                    value=False,
                    elem_id="nzap-dump-cross-attention-output",
                )
                dump_mlp_output = gr.Checkbox(
                    label="Dump MLP output",
                    value=False,
                    elem_id="nzap-dump-mlp-output",
                )
                dump_spectrum_final_output = gr.Checkbox(
                    label="Dump Spectrum final output",
                    value=False,
                    elem_id="nzap-dump-spectrum-final-output",
                )
                dump_baseline_final_output = gr.Checkbox(
                    label="Dump baseline final output",
                    value=False,
                    elem_id="nzap-dump-baseline-final-output",
                )
                gr.HTML(
                    '<div style="border-top: 3px solid var(--block-border-color, #4b5563); margin: 0.85rem 0 0.7rem;"></div>',
                    elem_id="nzap-debug-dump-divider",
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
                for control in (
                    dump_teacache_residual,
                    dump_ujicache_residual,
                    dump_block_output,
                    dump_cross_attention_output,
                    dump_mlp_output,
                    dump_spectrum_final_output,
                    dump_baseline_final_output,
                ):
                    control.change(
                        fn=_enable_parent_and_debug_if_child_enabled,
                        inputs=[control],
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
            with gr.Accordion("UjiCache", open=False, elem_id="nzap-ujicache-panel"):
                ujicache_enabled = gr.Checkbox(
                    label="Enable UjiCache experiment",
                    value=False,
                    elem_id="nzap-ujicache-enable",
                )
                ujicache_preset = gr.Dropdown(
                    label="UjiCache preset",
                    choices=UJICACHE_PRESETS,
                    value=UJICACHE_PRESET_CUSTOM,
                    elem_id="nzap-ujicache-preset",
                )
                ujicache_threshold = gr.Slider(
                    label="Rel L1 threshold",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.005,
                    value=0.07,
                    elem_id="nzap-ujicache-threshold",
                )
                ujicache_start_percent = gr.Slider(
                    label="Start progress",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    value=0.05,
                    elem_id="nzap-ujicache-start-percent",
                )
                ujicache_end_percent = gr.Slider(
                    label="End progress",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    value=0.95,
                    elem_id="nzap-ujicache-end-percent",
                )
                ujicache_modulated_source = gr.Dropdown(
                    label="Modulated source",
                    choices=TEACACHE_MODULATED_SOURCES,
                    value=TEACACHE_SOURCE_FIRST_BLOCK_SHIFT,
                    elem_id="nzap-ujicache-modulated-source",
                )
                ujicache_coefficient_profile = gr.Dropdown(
                    label="Coefficient profile",
                    choices=TEACACHE_COEFFICIENT_PROFILES,
                    value=TEACACHE_PROFILE_ANIMA_2B_30STEP_FIRST_BLOCK_SHIFT,
                    elem_id="nzap-ujicache-coefficient-profile",
                )
                ujicache_max_skip_streak = gr.Slider(
                    label="Max skip streak (0 = off)",
                    minimum=0,
                    maximum=64,
                    step=1,
                    value=0,
                    elem_id="nzap-ujicache-max-skip-streak",
                )
                ujicache_force_full_interval = gr.Slider(
                    label="Force full interval (0 = off)",
                    minimum=0,
                    maximum=64,
                    step=1,
                    value=0,
                    elem_id="nzap-ujicache-force-full-interval",
                )
                ujicache_formula = gr.Dropdown(
                    label="Prediction formula",
                    choices=UJICACHE_FORMULAS,
                    value=UJICACHE_FORMULA_TEACACHE,
                    elem_id="nzap-ujicache-formula",
                )
                ujicache_use_prediction_after_progress = gr.Slider(
                    label="Use prediction after progress",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    value=0.0,
                    interactive=False,
                    elem_id="nzap-ujicache-use-prediction-after-progress",
                )
                ujicache_apply_prediction_from_skip = gr.Slider(
                    label="Apply prediction from skip #",
                    minimum=1,
                    maximum=3,
                    step=1,
                    value=2,
                    interactive=False,
                    elem_id="nzap-ujicache-apply-prediction-from-skip",
                )
                ujicache_prediction_strength = gr.Slider(
                    label="Prediction strength",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    value=0.50,
                    interactive=False,
                    elem_id="nzap-ujicache-prediction-strength",
                )
                ujicache_taylor2_curve_strength = gr.Slider(
                    label="Taylor2 curve strength",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    value=0.25,
                    interactive=False,
                    elem_id="nzap-ujicache-taylor2-curve-strength",
                )
                ujicache_slope_ema_smoothing = gr.Slider(
                    label="Slope EMA Smoothing",
                    minimum=0.0,
                    maximum=0.99,
                    step=0.01,
                    value=0.0,
                    interactive=False,
                    elem_id="nzap-ujicache-slope-ema-smoothing",
                )
                ujicache_curve_ema_smoothing = gr.Slider(
                    label="Curve EMA Smoothing",
                    minimum=0.0,
                    maximum=0.99,
                    step=0.01,
                    value=0.0,
                    interactive=False,
                    elem_id="nzap-ujicache-curve-ema-smoothing",
                )
                with gr.Accordion(
                    "Auto Uji mode",
                    open=False,
                    elem_id="nzap-auto-uji-panel",
                ):
                    auto_ujicache_enabled = gr.Checkbox(
                        label="Enable Auto Uji mode",
                        value=False,
                        elem_id="nzap-auto-uji-enable",
                    )
                    auto_ujicache_csv = gr.Textbox(
                        label="Auto Uji CSV",
                        lines=6,
                        elem_id="nzap-auto-uji-csv",
                    )
                ujicache_cache_device = gr.Radio(
                    label="Cache device",
                    choices=TEACACHE_CACHE_DEVICES,
                    value=TEACACHE_CACHE_DEVICE_CUDA,
                    elem_id="nzap-ujicache-cache-device",
                )
                ujicache_dry_run = gr.Checkbox(
                    label="Dry run",
                    value=False,
                    elem_id="nzap-ujicache-dry-run",
                )
                ujicache_verbose_trace = gr.Checkbox(
                    label="Verbose UjiCache trace",
                    value=False,
                    elem_id="nzap-ujicache-verbose-trace",
                )
                ujicache_formula.change(
                    fn=_ujicache_prediction_control_updates,
                    inputs=[ujicache_formula, ujicache_slope_ema_smoothing],
                    outputs=[
                        ujicache_use_prediction_after_progress,
                        ujicache_apply_prediction_from_skip,
                        ujicache_prediction_strength,
                        ujicache_taylor2_curve_strength,
                        ujicache_slope_ema_smoothing,
                        ujicache_curve_ema_smoothing,
                    ],
                )
                ujicache_slope_ema_smoothing.change(
                    fn=_ujicache_prediction_control_updates,
                    inputs=[ujicache_formula, ujicache_slope_ema_smoothing],
                    outputs=[
                        ujicache_use_prediction_after_progress,
                        ujicache_apply_prediction_from_skip,
                        ujicache_prediction_strength,
                        ujicache_taylor2_curve_strength,
                        ujicache_slope_ema_smoothing,
                        ujicache_curve_ema_smoothing,
                    ],
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
                    outputs=[
                        spectrum_enabled,
                        ujicache_enabled,
                        enabled,
                        auto_ujicache_enabled,
                    ],
                )
                spectrum_enabled.change(
                    fn=_spectrum_enable_updates,
                    inputs=[spectrum_enabled],
                    outputs=[
                        teacache_enabled,
                        ujicache_enabled,
                        enabled,
                        auto_ujicache_enabled,
                    ],
                )
                ujicache_enabled.change(
                    fn=_ujicache_enable_updates,
                    inputs=[ujicache_enabled],
                    outputs=[
                        teacache_enabled,
                        spectrum_enabled,
                        enabled,
                        auto_ujicache_enabled,
                    ],
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
            ujicache_enabled,
            ujicache_preset,
            ujicache_threshold,
            ujicache_start_percent,
            ujicache_end_percent,
            ujicache_formula,
            ujicache_use_prediction_after_progress,
            ujicache_apply_prediction_from_skip,
            ujicache_prediction_strength,
            ujicache_taylor2_curve_strength,
            ujicache_slope_ema_smoothing,
            ujicache_curve_ema_smoothing,
            ujicache_cache_device,
            ujicache_modulated_source,
            ujicache_coefficient_profile,
            ujicache_max_skip_streak,
            ujicache_force_full_interval,
            ujicache_dry_run,
            ujicache_verbose_trace,
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
            dump_teacache_residual,
            dump_block_output,
            dump_cross_attention_output,
            dump_mlp_output,
            dump_spectrum_final_output,
            dump_baseline_final_output,
            dump_ujicache_residual,
            auto_ujicache_enabled,
            auto_ujicache_csv,
        ]

    def before_process(self, p, *script_args):
        try:
            _prepare_auto_ujicache_run(p, script_args)
        except AutoUjiCsvError as exc:
            STATE.auto_ujicache_parse_error = str(exc)
            STATE.set_error(f"auto uji csv error: {exc}")
            error(f"auto_uji_csv_error {exc}")
            raise RuntimeError(f"Auto Uji CSV error: {exc}") from exc
        except Exception as exc:
            STATE.set_error(f"auto uji setup failed: {exc}")
            exception("auto uji setup failed")
            raise

    def process(self, p, *script_args):
        try:
            _apply_auto_ujicache_seed_template(p)
        except Exception as exc:
            STATE.set_error(f"auto uji seed setup failed: {exc}")
            exception("auto uji seed setup failed")
            raise

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
        try:
            from .tensor_dump import flush_stats

            flush_stats()
        except Exception as exc:
            STATE.tensor_dump_errors += 1
            warning(f"tensor_dump_flush_failed reason={exc}")
            exception("tensor dump flush failed")
        _finish_auto_ujicache_run(p)


_AUTO_UJICACHE_P_ATTRS = (
    "_nzap_auto_ujicache_rows",
    "_nzap_auto_ujicache_original_n_iter",
    "_nzap_auto_ujicache_original_batch_size",
    "_nzap_auto_ujicache_seed_template_size",
    "_nzap_auto_ujicache_seed_template_ready",
    "_nzap_auto_ujicache_logged_row_index",
    "_nzap_auto_ujicache_iteration_counter",
)


def _prepare_auto_ujicache_run(p, script_args) -> None:
    _apply_ui_args(script_args)
    _clear_auto_ujicache_p_attrs(p)
    STATE.auto_ujicache_active = False
    STATE.auto_ujicache_row_index = None
    STATE.auto_ujicache_row_name = None
    STATE.auto_ujicache_row_count = 0
    STATE.auto_ujicache_original_n_iter = 1
    STATE.auto_ujicache_parse_error = None

    if not (STATE.enabled and STATE.ujicache_enabled and STATE.auto_ujicache_enabled):
        return

    result = parse_auto_ujicache_csv(STATE.auto_ujicache_csv)
    for message in result.warnings:
        warning(f"auto_uji_csv_warning {message}")

    rows = result.rows
    original_n_iter = _positive_int(getattr(p, "n_iter", 1), 1)
    batch_size = _positive_int(getattr(p, "batch_size", 1), 1)
    total_n_iter = len(rows) * original_n_iter

    setattr(p, "_nzap_auto_ujicache_rows", rows)
    setattr(p, "_nzap_auto_ujicache_original_n_iter", original_n_iter)
    setattr(p, "_nzap_auto_ujicache_original_batch_size", batch_size)
    setattr(p, "n_iter", total_n_iter)

    STATE.auto_ujicache_active = True
    STATE.auto_ujicache_row_count = len(rows)
    STATE.auto_ujicache_original_n_iter = original_n_iter

    info(
        "auto_uji_prepare "
        f"rows={len(rows)} original_n_iter={original_n_iter} "
        f"batch_size={batch_size} total_n_iter={total_n_iter}"
    )


def _apply_auto_ujicache_seed_template(p) -> None:
    rows = getattr(p, "_nzap_auto_ujicache_rows", None)
    if not rows:
        return

    row_count = len(rows)
    original_n_iter = _positive_int(
        getattr(p, "_nzap_auto_ujicache_original_n_iter", 1),
        1,
    )
    batch_size = _positive_int(getattr(p, "batch_size", 1), 1)
    template_size = original_n_iter * batch_size
    setattr(p, "_nzap_auto_ujicache_seed_template_size", template_size)

    raw_seed_values = getattr(p, "all_seeds", None)
    if not raw_seed_values and _safe_int(getattr(p, "seed", -1), -1) < 0:
        return

    was_ready = bool(getattr(p, "_nzap_auto_ujicache_seed_template_ready", False))
    seed_template = _seed_template(
        raw_seed_values,
        getattr(p, "seed", 0),
        template_size,
    )
    setattr(p, "all_seeds", seed_template * row_count)

    if hasattr(p, "all_subseeds"):
        subseed_template = _seed_template(
            getattr(p, "all_subseeds", None),
            getattr(p, "subseed", 0),
            template_size,
        )
        setattr(p, "all_subseeds", subseed_template * row_count)

    setattr(p, "_nzap_auto_ujicache_seed_template_ready", True)
    if not was_ready:
        info(
            "auto_uji_seed_template "
            f"rows={row_count} template_size={template_size} seeds={_seed_label(seed_template)}"
        )


def _apply_auto_ujicache_row_if_needed(p) -> None:
    rows = getattr(p, "_nzap_auto_ujicache_rows", None)
    if not rows or not STATE.auto_ujicache_active or not STATE.ujicache_enabled:
        return

    original_n_iter = _positive_int(
        getattr(p, "_nzap_auto_ujicache_original_n_iter", 1),
        1,
    )
    iteration = _current_auto_ujicache_iteration(p)
    row_index = max(0, min(len(rows) - 1, iteration // original_n_iter))
    repeat_index = (iteration % original_n_iter) + 1
    row = rows[row_index]

    apply_auto_ujicache_row_to_state(row)
    STATE.auto_ujicache_row_index = row.index
    STATE.auto_ujicache_row_name = row.name

    logged_row_index = getattr(p, "_nzap_auto_ujicache_logged_row_index", None)
    if logged_row_index != row_index:
        setattr(p, "_nzap_auto_ujicache_logged_row_index", row_index)
        info(
            "auto_uji_row_start "
            f"index={row_index + 1}/{len(rows)} name={row.name} "
            f"repeat={repeat_index}/{original_n_iter} "
            f"threshold={STATE.ujicache_threshold:.4f} "
            f"formula={STATE.ujicache_formula} "
            f"prediction_strength={STATE.ujicache_prediction_strength:.2f} "
            f"batch_size={_positive_int(getattr(p, 'batch_size', 1), 1)} "
            f"seeds={_row_seed_label(p, row_index)}"
        )


def _finish_auto_ujicache_run(p) -> None:
    _clear_auto_ujicache_p_attrs(p)
    STATE.auto_ujicache_active = False
    STATE.auto_ujicache_row_index = None
    STATE.auto_ujicache_row_name = None
    STATE.auto_ujicache_row_count = 0
    STATE.auto_ujicache_original_n_iter = 1


def _clear_auto_ujicache_p_attrs(p) -> None:
    for attr in _AUTO_UJICACHE_P_ATTRS:
        try:
            if hasattr(p, attr):
                delattr(p, attr)
        except Exception:
            setattr(p, attr, None)


def _current_auto_ujicache_iteration(p) -> int:
    value = getattr(p, "iteration", None)
    if value is not None:
        try:
            return max(0, int(value))
        except Exception:
            pass
    counter = _positive_int(getattr(p, "_nzap_auto_ujicache_iteration_counter", 0), 0)
    setattr(p, "_nzap_auto_ujicache_iteration_counter", counter + 1)
    return counter


def _row_seed_label(p, row_index: int) -> str:
    template_size = _positive_int(
        getattr(p, "_nzap_auto_ujicache_seed_template_size", 0),
        0,
    )
    if template_size <= 0:
        original_n_iter = _positive_int(
            getattr(p, "_nzap_auto_ujicache_original_n_iter", 1),
            1,
        )
        template_size = original_n_iter * _positive_int(getattr(p, "batch_size", 1), 1)
    try:
        seeds = list(getattr(p, "all_seeds", []) or [])
    except Exception:
        seeds = []
    start = row_index * template_size
    return _seed_label(seeds[start : start + template_size])


def _seed_template(values, base_seed, size: int) -> list[int]:
    try:
        template = list(values or [])[:size]
    except Exception:
        template = []
    if len(template) >= size:
        return template

    if template:
        base = _safe_int(template[0], 0)
    else:
        base = _safe_int(base_seed, 0)
        if base < 0:
            base = 0

    while len(template) < size:
        template.append(base + len(template))
    return template


def _seed_label(seeds) -> str:
    values = list(seeds or [])
    if not values:
        return "None"
    if len(values) == 1:
        return str(values[0])
    return f"{values[0]}..{values[-1]}"


def _positive_int(value, minimum: int) -> int:
    try:
        number = int(value)
    except Exception:
        number = minimum
    return max(minimum, number)


def _safe_int(value, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _apply_ui_args(script_args) -> None:
    if len(script_args) >= 76:
        STATE.apply_options(*script_args[:76])
        return
    if len(script_args) >= 75:
        STATE.apply_options(*list(script_args[:75]), "")
        return
    if len(script_args) >= 74:
        STATE.apply_options(*script_args[:74])
        return
    if len(script_args) >= 73:
        STATE.apply_options(*script_args[:73])
        return
    if len(script_args) >= 72:
        STATE.apply_options(*_insert_default_ujicache_ema_args(script_args[:72]))
        return
    if len(script_args) >= 71:
        STATE.apply_options(*_insert_default_ujicache_ema_args(script_args[:71]))
        return
    if len(script_args) >= 54:
        STATE.apply_options(*_insert_default_ujicache_args(script_args[:54]))
        return
    if len(script_args) >= 53:
        STATE.apply_options(*_insert_default_ujicache_args(script_args[:53]))
        return
    if len(script_args) >= 48:
        STATE.apply_options(*_insert_default_ujicache_args(script_args[:48]))
        return
    if len(script_args) >= 46:
        STATE.apply_options(*_insert_default_ujicache_args(script_args[:46]))
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


def _insert_default_ujicache_args(script_args):
    ujicache_defaults = [
        False,
        UJICACHE_PRESET_CUSTOM,
        0.07,
        0.05,
        0.95,
        UJICACHE_FORMULA_TEACACHE,
        0.0,
        2,
        0.50,
        0.25,
        0.0,
        0.0,
        TEACACHE_CACHE_DEVICE_CUDA,
        TEACACHE_SOURCE_FIRST_BLOCK_SHIFT,
        TEACACHE_PROFILE_ANIMA_2B_30STEP_FIRST_BLOCK_SHIFT,
        0,
        0,
        False,
        False,
    ]
    args = list(script_args)
    return args[:35] + ujicache_defaults + args[35:]


def _insert_default_ujicache_ema_args(script_args):
    args = list(script_args)
    return args[:45] + [0.0, 0.0] + args[45:]


def _begin_generation(p, script_args, source: str) -> None:
    _apply_ui_args(script_args)
    _apply_auto_ujicache_seed_template(p)
    _apply_auto_ujicache_row_if_needed(p)
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

    if _tensor_dump_will_save():
        try:
            from .tensor_dump import initialize_run_if_needed

            initialize_run_if_needed(p)
        except Exception as exc:
            STATE.tensor_dump_errors += 1
            warning(f"tensor_dump_initialize_failed reason={exc}")

    _configure_generation_patches()
    _apply_infotext_metadata(p)
    log_generation_start(p)


def _apply_infotext_metadata(p) -> None:
    if not STATE.ujicache_enabled:
        return
    try:
        params = getattr(p, "extra_generation_params", None)
        if not isinstance(params, dict):
            params = {}
            setattr(p, "extra_generation_params", params)
        _clear_legacy_ujicache_metadata(params)
        params["Uji enabled"] = True
        params["Uji formula"] = STATE.ujicache_formula
        params["Uji threshold"] = f"{STATE.ujicache_threshold:.4f}"
        params["Uji progress"] = (
            f"{STATE.ujicache_start_percent:.2f}..{STATE.ujicache_end_percent:.2f}"
        )
        params["Uji use_prediction_after_progress"] = (
            f"{STATE.ujicache_use_prediction_after_progress:.2f}"
        )
        params["Uji apply_prediction_from_skip"] = STATE.ujicache_apply_prediction_from_skip
        params["Uji prediction_strength"] = f"{STATE.ujicache_prediction_strength:.2f}"
        params["Uji taylor2_curve_strength"] = (
            f"{STATE.ujicache_taylor2_curve_strength:.2f}"
        )
        params["Uji slope_ema_smoothing"] = f"{STATE.ujicache_slope_ema_smoothing:.2f}"
        params["Uji curve_ema_smoothing"] = f"{STATE.ujicache_curve_ema_smoothing:.2f}"
        params["Uji modulated_source"] = STATE.ujicache_modulated_source
        params["Uji coefficient_profile"] = STATE.ujicache_coefficient_profile
        params["Uji max_skip_streak"] = STATE.ujicache_max_skip_streak
        params["Uji force_full_interval"] = STATE.ujicache_force_full_interval
        if STATE.auto_ujicache_active and STATE.auto_ujicache_row_index is not None:
            if STATE.auto_ujicache_row_count > 0:
                params["Uji auto_row_index"] = (
                    f"{STATE.auto_ujicache_row_index}/{STATE.auto_ujicache_row_count}"
                )
            else:
                params["Uji auto_row_index"] = STATE.auto_ujicache_row_index
            params["Uji auto_row_name"] = STATE.auto_ujicache_row_name or ""
    except Exception as exc:
        warning(f"ujicache_metadata_failed reason={exc}")


def _clear_legacy_ujicache_metadata(params: dict) -> None:
    for key in (
        "UjiCache enabled",
        "UjiCache formula",
        "UjiCache use_prediction_after_progress",
        "UjiCache apply_prediction_from_skip",
        "UjiCache prediction_strength",
        "UjiCache taylor2_curve_strength",
        "UjiCache slope_ema_smoothing",
        "UjiCache curve_ema_smoothing",
        "Uji auto_row_index",
        "Uji auto_row_name",
    ):
        params.pop(key, None)


def _configure_generation_patches() -> None:
    remove_patch("tensor_dump")
    remove_patch("tensor_dump_output")

    if STATE.mode == MODE_IDENTITY_PATCH:
        remove_patch("block_structure_trace")
        remove_patch("attention_kernel")
        remove_patch("sparse_attention")
        remove_patch("teacache")
        remove_patch("ujicache")
        remove_patch("spectrum")
        apply_patch("block_forward_identity")
        return

    remove_patch("block_forward_identity")
    if STATE.ujicache_enabled:
        remove_patch("block_structure_trace")
        remove_patch("attention_kernel")
        remove_patch("sparse_attention")
        remove_patch("teacache")
        remove_patch("spectrum")
        result = apply_patch("ujicache")
        if not result.ok:
            STATE.ujicache_unavailable_reason = result.message
            warning(f"ujicache_patch_unavailable reason={result.message}")
    elif STATE.teacache_enabled:
        remove_patch("block_structure_trace")
        remove_patch("attention_kernel")
        remove_patch("sparse_attention")
        remove_patch("ujicache")
        remove_patch("spectrum")
        result = apply_patch("teacache")
        if not result.ok:
            STATE.teacache_unavailable_reason = result.message
            warning(f"teacache_patch_unavailable reason={result.message}")
    elif STATE.spectrum_enabled:
        remove_patch("teacache")
        remove_patch("ujicache")
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
        remove_patch("ujicache")
        remove_patch("spectrum")
        remove_patch("block_structure_trace")
        remove_patch("attention_kernel")
        apply_patch("sparse_attention")
    else:
        remove_patch("teacache")
        remove_patch("ujicache")
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
        and not STATE.ujicache_enabled
        and not STATE.spectrum_enabled
        and not STATE.sparse_enabled
        and not STATE.attention_override_active()
        and not STATE.tensor_dump_block_level_active()
    ):
        apply_patch("block_structure_trace")
    else:
        remove_patch("block_structure_trace")

    if STATE.tensor_dump_active() and STATE.dump_baseline_final_output and not STATE.spectrum_enabled:
        apply_patch("tensor_dump_output")
    else:
        remove_patch("tensor_dump_output")

    if (
        STATE.tensor_dump_active()
        and STATE.dump_spectrum_final_output
        and not STATE.spectrum_enabled
        and "spectrum_final_output_requires_spectrum" not in STATE.tensor_dump_warned_reasons
    ):
        STATE.tensor_dump_warned_reasons.add("spectrum_final_output_requires_spectrum")
        warning("tensor_dump_spectrum_final_output_inactive reason=spectrum_disabled")

    if (
        STATE.tensor_dump_active()
        and STATE.dump_baseline_final_output
        and STATE.spectrum_enabled
        and "baseline_final_output_requires_spectrum_off" not in STATE.tensor_dump_warned_reasons
    ):
        STATE.tensor_dump_warned_reasons.add("baseline_final_output_requires_spectrum_off")
        warning("tensor_dump_baseline_final_output_inactive reason=spectrum_enabled")

    if (
        STATE.tensor_dump_active()
        and STATE.dump_teacache_residual
        and not STATE.teacache_enabled
        and "teacache_residual_requires_teacache" not in STATE.tensor_dump_warned_reasons
    ):
        STATE.tensor_dump_warned_reasons.add("teacache_residual_requires_teacache")
        warning("tensor_dump_teacache_residual_inactive reason=teacache_disabled")

    if (
        STATE.tensor_dump_active()
        and STATE.dump_ujicache_residual
        and not STATE.ujicache_enabled
        and "ujicache_residual_requires_ujicache" not in STATE.tensor_dump_warned_reasons
    ):
        STATE.tensor_dump_warned_reasons.add("ujicache_residual_requires_ujicache")
        warning("tensor_dump_ujicache_residual_inactive reason=ujicache_disabled")

    if STATE.tensor_dump_block_level_active():
        if (
            STATE.teacache_enabled
            or STATE.ujicache_enabled
            or STATE.spectrum_enabled
            or STATE.sparse_enabled
            or STATE.attention_override_active()
        ):
            remove_patch("tensor_dump")
            warning("tensor_dump_unavailable reason=conflicts_with_active_generation_patch")
        else:
            apply_patch("tensor_dump")
    else:
        remove_patch("tensor_dump")


def _tensor_dump_will_save() -> bool:
    if not STATE.tensor_dump_active():
        return False
    if STATE.dump_teacache_residual and STATE.teacache_enabled:
        return True
    if STATE.dump_ujicache_residual and STATE.ujicache_enabled:
        return True
    if STATE.dump_spectrum_final_output and STATE.spectrum_enabled:
        return True
    if STATE.dump_baseline_final_output and not STATE.spectrum_enabled:
        return True
    if STATE.tensor_dump_block_level_active():
        return not (
            STATE.teacache_enabled
            or STATE.ujicache_enabled
            or STATE.spectrum_enabled
            or STATE.sparse_enabled
            or STATE.attention_override_active()
        )
    return False


def _remove_generation_patches() -> None:
    remove_patch("block_structure_trace")
    remove_patch("block_forward_identity")
    remove_patch("attention_kernel")
    remove_patch("sparse_attention")
    remove_patch("teacache")
    remove_patch("ujicache")
    remove_patch("spectrum")
    remove_patch("tensor_dump")
    remove_patch("tensor_dump_output")


def _requires_supported_model() -> bool:
    return (
        STATE.mode == MODE_IDENTITY_PATCH
        or STATE.teacache_enabled
        or STATE.ujicache_enabled
        or STATE.sparse_enabled
        or STATE.attention_override_active()
        or STATE.tensor_dump_block_level_active()
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


def _ujicache_prediction_control_updates(formula: str, slope_ema_smoothing: float):
    uses_prediction = formula in (UJICACHE_FORMULA_LINEAR, UJICACHE_FORMULA_TAYLOR2)
    uses_taylor = formula == UJICACHE_FORMULA_TAYLOR2
    try:
        slope = float(slope_ema_smoothing)
    except Exception:
        slope = 0.0
    return (
        gr.update(interactive=uses_prediction),
        gr.update(interactive=uses_prediction),
        gr.update(interactive=uses_prediction),
        gr.update(interactive=uses_taylor),
        gr.update(interactive=uses_prediction),
        gr.update(interactive=uses_taylor and slope > 0.0),
    )


def _teacache_enable_updates(child_enabled: bool):
    if child_enabled:
        return False, False, True, False
    return gr.update(), gr.update(), gr.update(), gr.update()


def _spectrum_enable_updates(child_enabled: bool):
    if child_enabled:
        return False, False, True, False
    return gr.update(), gr.update(), gr.update(), gr.update()


def _ujicache_enable_updates(child_enabled: bool):
    if child_enabled:
        return False, False, True, gr.update()
    return gr.update(), gr.update(), gr.update(), False


def _enable_parent_if_child_enabled(child_enabled: bool):
    if child_enabled:
        return True
    return gr.update()


def _debug_mode_selection_updates(mode: str):
    if _mode_selected(mode):
        return True, True
    return gr.update(), gr.update()


def _enable_parent_and_debug_if_child_enabled(child_enabled: bool):
    if child_enabled:
        return True, True
    return gr.update(), gr.update()


def _mode_selected(mode: str) -> bool:
    value = str(mode or MODE_OFF)
    return value not in (MODE_OFF, "Off")
