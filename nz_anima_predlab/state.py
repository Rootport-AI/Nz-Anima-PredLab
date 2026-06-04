from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any


MODE_OFF = ""
MODE_DIAGNOSE = "Diagnose only"
MODE_IDENTITY_PATCH = "Identity Patch test"
MODE_TRACE_ATTENTION = "Trace attention"
MODE_TRACE_COND = "Trace cond/uncond"
MODE_TRACE_LOWBIT = "Trace low-bit / compile"

MODES = [
    MODE_OFF,
    MODE_DIAGNOSE,
    MODE_IDENTITY_PATCH,
]

SPARSE_BACKEND_NATTEN = "NATTEN (optional)"
SPARSE_BACKEND_TORCH = "Torch prototype"
SPARSE_BACKENDS = [SPARSE_BACKEND_NATTEN, SPARSE_BACKEND_TORCH]

ATTENTION_BACKEND_CURRENT = "Forge current/default"
ATTENTION_BACKENDS = [
    ATTENTION_BACKEND_CURRENT,
    "attention_sage",
    "attention_flash",
    "attention_xformers",
    "attention_pytorch",
]

ATTENTION_TARGET_BOTH = "self + cross"
ATTENTION_TARGET_SELF = "self only"
ATTENTION_TARGET_CROSS = "cross only"
ATTENTION_TARGETS = [
    ATTENTION_TARGET_BOTH,
    ATTENTION_TARGET_SELF,
    ATTENTION_TARGET_CROSS,
]

TEACACHE_PRESET_SAFE = "Safe"
TEACACHE_PRESET_BALANCED = "Balanced"
TEACACHE_PRESET_AGGRESSIVE = "Aggressive"
TEACACHE_PRESET_CUSTOM = "Custom"
TEACACHE_PRESETS = [
    TEACACHE_PRESET_SAFE,
    TEACACHE_PRESET_BALANCED,
    TEACACHE_PRESET_AGGRESSIVE,
    TEACACHE_PRESET_CUSTOM,
]

TEACACHE_CACHE_DEVICE_CUDA = "cuda"
TEACACHE_CACHE_DEVICE_CPU = "cpu"
TEACACHE_CACHE_DEVICES = [
    TEACACHE_CACHE_DEVICE_CUDA,
    TEACACHE_CACHE_DEVICE_CPU,
]

TEACACHE_SOURCE_FIRST_BLOCK_SHIFT = "first_block_shift"
TEACACHE_SOURCE_TIMESTEP_EMBEDDING = "timestep_embedding"
TEACACHE_MODULATED_SOURCES = [
    TEACACHE_SOURCE_FIRST_BLOCK_SHIFT,
    TEACACHE_SOURCE_TIMESTEP_EMBEDDING,
]

TEACACHE_PROFILE_ANIMA_2B_30STEP_FIRST_BLOCK_SHIFT = "Anima 2B 30step first_block_shift"
TEACACHE_PROFILE_IDENTITY = "Identity estimate"
TEACACHE_COEFFICIENT_PROFILES = [
    TEACACHE_PROFILE_ANIMA_2B_30STEP_FIRST_BLOCK_SHIFT,
    TEACACHE_PROFILE_IDENTITY,
]

TEACACHE_COEFFICIENTS_ANIMA_2B_30STEP_FIRST_BLOCK_SHIFT = [
    5954.035087553969,
    -2410.0426539290293,
    349.24023850217395,
    -17.264742642375417,
    0.31229336331906893,
]

UJICACHE_PRESET_CUSTOM = "Custom"
UJICACHE_PRESETS = [UJICACHE_PRESET_CUSTOM]

UJICACHE_FORMULA_TEACACHE = "TeaCache (residual only)"
UJICACHE_FORMULA_LINEAR = "Linear extrapolation"
UJICACHE_FORMULA_TAYLOR2 = "Taylor2 curve"
UJICACHE_FORMULAS = [
    UJICACHE_FORMULA_TEACACHE,
    UJICACHE_FORMULA_LINEAR,
    UJICACHE_FORMULA_TAYLOR2,
]

SPECTRUM_PRESET_SAFE = "Safe"
SPECTRUM_PRESET_BALANCED = "Balanced"
SPECTRUM_PRESET_AGGRESSIVE = "Aggressive"
SPECTRUM_PRESET_CUSTOM = "Custom"
SPECTRUM_PRESETS = [
    SPECTRUM_PRESET_SAFE,
    SPECTRUM_PRESET_BALANCED,
    SPECTRUM_PRESET_AGGRESSIVE,
    SPECTRUM_PRESET_CUSTOM,
]


@dataclass
class RuntimeState:
    enabled: bool = False
    debug_log_enabled: bool = True
    mode: str = MODE_OFF
    print_timing_log: bool = True
    verbose_diagnose_log: bool = False
    attention_enabled: bool = False
    attention_backend: str = ATTENTION_BACKEND_CURRENT
    attention_target: str = ATTENTION_TARGET_BOTH
    attention_block_start: int = 0
    attention_block_end: int = 27
    sparse_enabled: bool = False
    sparse_backend: str = SPARSE_BACKEND_NATTEN
    sparse_block_start: int = 14
    sparse_block_end: int = 27
    sparse_step_start: int = 0
    sparse_step_end: int = -1
    sparse_local_window: int = 15
    sparse_dilation: int = 1
    sparse_full_attention_interval: int = 0
    cond_uncond_enabled: bool = False
    cond_uncond_skip_cfg1: bool = False
    cond_uncond_schedule_enabled: bool = False
    cond_uncond_guidance_interval: int = 1
    lowbit_enabled: bool = False
    compile_enabled: bool = False
    teacache_enabled: bool = False
    teacache_preset: str = TEACACHE_PRESET_BALANCED
    teacache_threshold: float = 0.07
    teacache_start_percent: float = 0.05
    teacache_end_percent: float = 0.95
    teacache_cache_device: str = TEACACHE_CACHE_DEVICE_CUDA
    teacache_modulated_source: str = TEACACHE_SOURCE_FIRST_BLOCK_SHIFT
    teacache_coefficient_profile: str = TEACACHE_PROFILE_ANIMA_2B_30STEP_FIRST_BLOCK_SHIFT
    teacache_max_skip_streak: int = 0
    teacache_force_full_interval: int = 0
    teacache_dry_run: bool = False
    teacache_verbose_trace: bool = False
    ujicache_enabled: bool = False
    ujicache_preset: str = UJICACHE_PRESET_CUSTOM
    ujicache_threshold: float = 0.07
    ujicache_start_percent: float = 0.05
    ujicache_end_percent: float = 0.95
    ujicache_formula: str = UJICACHE_FORMULA_TEACACHE
    ujicache_use_prediction_after_progress: float = 0.0
    ujicache_apply_prediction_from_skip: int = 2
    ujicache_prediction_strength: float = 0.50
    ujicache_taylor2_curve_strength: float = 0.25
    ujicache_slope_ema_smoothing: float = 0.0
    ujicache_curve_ema_smoothing: float = 0.0
    ujicache_cache_device: str = TEACACHE_CACHE_DEVICE_CUDA
    ujicache_modulated_source: str = TEACACHE_SOURCE_FIRST_BLOCK_SHIFT
    ujicache_coefficient_profile: str = TEACACHE_PROFILE_ANIMA_2B_30STEP_FIRST_BLOCK_SHIFT
    ujicache_max_skip_streak: int = 0
    ujicache_force_full_interval: int = 0
    ujicache_dry_run: bool = False
    ujicache_verbose_trace: bool = False
    auto_ujicache_enabled: bool = False
    auto_ujicache_csv: str = ""
    auto_ujicache_active: bool = False
    auto_ujicache_row_index: int | None = None
    auto_ujicache_row_name: str | None = None
    auto_ujicache_row_count: int = 0
    auto_ujicache_original_n_iter: int = 1
    auto_ujicache_parse_error: str | None = None
    spectrum_enabled: bool = False
    spectrum_preset: str = SPECTRUM_PRESET_BALANCED
    spectrum_w: float = 0.20
    spectrum_m: int = 16
    spectrum_lambda: float = 0.50
    spectrum_warmup_steps: int = 6
    spectrum_window_size: int = 2
    spectrum_flex_window: float = 0.0
    spectrum_stop_progress: float = 0.80
    spectrum_dry_run: bool = False
    spectrum_verbose_trace: bool = False
    dump_teacache_residual: bool = False
    dump_ujicache_residual: bool = False
    dump_block_output: bool = False
    dump_cross_attention_output: bool = False
    dump_mlp_output: bool = False
    dump_spectrum_final_output: bool = False
    dump_baseline_final_output: bool = False
    status: str = "disabled"
    error_message: str | None = None
    model_detection: Any | None = None
    warned_model_keys: set[str] = field(default_factory=set)
    generation_index: int = 0
    generation_steps: int | None = None
    generation_start: float | None = None
    step_start: float | None = None
    step_durations: list[float] = field(default_factory=list)
    denoiser_calls: int = 0
    cond_trace_logged: bool = False
    cond_batch_trace_logged: bool = False
    block_trace_call_count: int = 0
    block_trace_qkv_logged: set[tuple[int, str]] = field(default_factory=set)
    current_block_index: int | None = None
    identity_patch_calls: int = 0
    identity_patch_logged_calls: int = 0
    identity_patch_shape_mismatches: int = 0
    identity_patch_errors: int = 0
    identity_patch_num_blocks: int | None = None
    attention_kernel_calls: int = 0
    attention_kernel_block_calls: int = 0
    attention_kernel_fallbacks: int = 0
    attention_kernel_errors: int = 0
    attention_kernel_logged_calls: int = 0
    attention_kernel_num_blocks: int | None = None
    attention_kernel_current_context: dict[str, Any] | None = None
    attention_kernel_actual_counts: dict[str, int] = field(default_factory=dict)
    attention_kernel_internal_fallbacks: int = 0
    attention_kernel_internal_errors: int = 0
    attention_kernel_last_trace: dict[str, Any] | None = None
    sparse_block_calls: int = 0
    sparse_attention_calls: int = 0
    sparse_fallbacks: int = 0
    sparse_errors: int = 0
    sparse_logged_calls: int = 0
    sparse_num_blocks: int | None = None
    sparse_current_context: dict[str, Any] | None = None
    sparse_unavailable_reason: str | None = None
    teacache_model_calls: int = 0
    teacache_full_calcs: int = 0
    teacache_skips: int = 0
    teacache_dry_run_skips: int = 0
    teacache_first_full_calcs: int = 0
    teacache_forced_full_calcs: int = 0
    teacache_fallbacks: int = 0
    teacache_errors: int = 0
    teacache_logged_calls: int = 0
    teacache_num_blocks: int | None = None
    teacache_unavailable_reason: str | None = None
    ujicache_model_calls: int = 0
    ujicache_full_calcs: int = 0
    ujicache_skips: int = 0
    ujicache_skipped_steps: list[int] = field(default_factory=list)
    ujicache_prediction_used: int = 0
    ujicache_fallback_used: int = 0
    ujicache_dry_run_predictions: int = 0
    ujicache_first_full_calcs: int = 0
    ujicache_forced_full_calcs: int = 0
    ujicache_fallbacks: int = 0
    ujicache_errors: int = 0
    ujicache_logged_calls: int = 0
    ujicache_num_blocks: int | None = None
    ujicache_unavailable_reason: str | None = None
    ujicache_fallback_reasons: dict[str, int] = field(default_factory=dict)
    spectrum_model_calls: int = 0
    spectrum_actual_forwards: int = 0
    spectrum_forecasts: int = 0
    spectrum_dry_run_forecasts: int = 0
    spectrum_fallbacks: int = 0
    spectrum_errors: int = 0
    spectrum_logged_calls: int = 0
    spectrum_unavailable_reason: str | None = None
    tensor_dump_run_dir: str | None = None
    tensor_dump_initialized: bool = False
    tensor_dump_records: int = 0
    tensor_dump_errors: int = 0
    tensor_dump_unavailable_reason: str | None = None
    tensor_dump_block_call_index: int = 0
    tensor_dump_block_local_call_index: int = 0
    tensor_dump_cross_attention_local_call_index: int = 0
    tensor_dump_mlp_local_call_index: int = 0
    tensor_dump_teacache_local_call_index: int = 0
    tensor_dump_ujicache_local_call_index: int = 0
    tensor_dump_spectrum_local_call_index: int = 0
    tensor_dump_baseline_local_call_index: int = 0
    tensor_dump_current_context: dict[str, Any] | None = None
    tensor_dump_num_blocks: int | None = None
    tensor_dump_warned_reasons: set[str] = field(default_factory=set)
    generation_logged: bool = False
    generation_start_source: str | None = None
    patches: dict[str, Any] = field(default_factory=dict)

    def refresh_settings(self) -> None:
        try:
            from modules import shared

            opts = shared.opts
            self.enabled = bool(getattr(opts, "nzap_enable", False))
            self.debug_log_enabled = bool(
                getattr(opts, "nzap_debug_log_enable", True)
            )
            self.mode = (
                _normalize_mode(getattr(opts, "nzap_mode", MODE_OFF))
                if self.debug_log_enabled
                else MODE_OFF
            )
            self.print_timing_log = bool(getattr(opts, "nzap_print_timing_log", True))
            self.verbose_diagnose_log = bool(
                getattr(opts, "nzap_verbose_diagnose_log", False)
            )
        except Exception as exc:
            self.enabled = False
            self.mode = MODE_OFF
            self.status = "error"
            self.error_message = f"failed to read settings: {exc}"

    def apply_options(
        self,
        enabled: bool,
        mode: str,
        print_timing_log: bool,
        verbose_diagnose_log: bool,
        attention_backend: str = ATTENTION_BACKEND_CURRENT,
        attention_target: str = ATTENTION_TARGET_BOTH,
        attention_block_start: int = 0,
        attention_block_end: int = 27,
        sparse_enabled: bool = False,
        sparse_backend: str = SPARSE_BACKEND_NATTEN,
        sparse_block_start: int = 14,
        sparse_block_end: int = 27,
        sparse_step_start: int = 0,
        sparse_step_end: int = -1,
        sparse_local_window: int = 15,
        sparse_dilation: int = 1,
        sparse_full_attention_interval: int = 0,
        cond_uncond_enabled: bool = False,
        cond_uncond_skip_cfg1: bool = False,
        cond_uncond_schedule_enabled: bool = False,
        cond_uncond_guidance_interval: int = 1,
        lowbit_enabled: bool = False,
        compile_enabled: bool = False,
        teacache_enabled: bool = False,
        teacache_preset: str = TEACACHE_PRESET_BALANCED,
        teacache_threshold: float = 0.07,
        teacache_start_percent: float = 0.05,
        teacache_end_percent: float = 0.95,
        teacache_cache_device: str = TEACACHE_CACHE_DEVICE_CUDA,
        teacache_modulated_source: str = TEACACHE_SOURCE_FIRST_BLOCK_SHIFT,
        teacache_coefficient_profile: str = TEACACHE_PROFILE_ANIMA_2B_30STEP_FIRST_BLOCK_SHIFT,
        teacache_max_skip_streak: int = 0,
        teacache_force_full_interval: int = 0,
        teacache_dry_run: bool = False,
        teacache_verbose_trace: bool = False,
        ujicache_enabled: bool = False,
        ujicache_preset: str = UJICACHE_PRESET_CUSTOM,
        ujicache_threshold: float = 0.07,
        ujicache_start_percent: float = 0.05,
        ujicache_end_percent: float = 0.95,
        ujicache_formula: str = UJICACHE_FORMULA_TEACACHE,
        ujicache_use_prediction_after_progress: float = 0.0,
        ujicache_apply_prediction_from_skip: int = 2,
        ujicache_prediction_strength: float = 0.50,
        ujicache_taylor2_curve_strength: float = 0.25,
        ujicache_slope_ema_smoothing: float = 0.0,
        ujicache_curve_ema_smoothing: float = 0.0,
        ujicache_cache_device: str = TEACACHE_CACHE_DEVICE_CUDA,
        ujicache_modulated_source: str = TEACACHE_SOURCE_FIRST_BLOCK_SHIFT,
        ujicache_coefficient_profile: str = TEACACHE_PROFILE_ANIMA_2B_30STEP_FIRST_BLOCK_SHIFT,
        ujicache_max_skip_streak: int = 0,
        ujicache_force_full_interval: int = 0,
        ujicache_dry_run: bool = False,
        ujicache_verbose_trace: bool = False,
        spectrum_enabled: bool = False,
        spectrum_preset: str = SPECTRUM_PRESET_BALANCED,
        spectrum_w: float = 0.20,
        spectrum_m: int = 16,
        spectrum_lambda: float = 0.50,
        spectrum_warmup_steps: int = 6,
        spectrum_window_size: int = 2,
        spectrum_flex_window: float = 0.0,
        spectrum_stop_progress: float = 0.80,
        spectrum_dry_run: bool = False,
        spectrum_verbose_trace: bool = False,
        debug_log_enabled: bool = True,
        attention_enabled: bool | None = None,
        dump_teacache_residual: bool = False,
        dump_block_output: bool = False,
        dump_cross_attention_output: bool = False,
        dump_mlp_output: bool = False,
        dump_spectrum_final_output: bool = False,
        dump_baseline_final_output: bool = False,
        dump_ujicache_residual: bool = False,
        auto_ujicache_enabled: bool = False,
        auto_ujicache_csv: str = "",
    ) -> None:
        self.enabled = bool(enabled)
        self.debug_log_enabled = bool(debug_log_enabled)
        self.mode = _normalize_mode(mode) if self.debug_log_enabled else MODE_OFF
        self.print_timing_log = bool(print_timing_log)
        self.verbose_diagnose_log = bool(verbose_diagnose_log)
        self.attention_enabled = (
            bool(attention_enabled)
            if attention_enabled is not None
            else attention_backend != ATTENTION_BACKEND_CURRENT
        )
        self.attention_backend = (
            attention_backend if attention_backend in ATTENTION_BACKENDS else ATTENTION_BACKEND_CURRENT
        )
        self.attention_target = (
            attention_target if attention_target in ATTENTION_TARGETS else ATTENTION_TARGET_BOTH
        )
        self.attention_block_start = _clamp_int(attention_block_start, 0, 27)
        self.attention_block_end = _clamp_int(attention_block_end, 0, 27)
        self.sparse_enabled = bool(sparse_enabled)
        self.sparse_backend = sparse_backend if sparse_backend in SPARSE_BACKENDS else SPARSE_BACKEND_NATTEN
        self.sparse_block_start = _clamp_int(sparse_block_start, 0, 27)
        self.sparse_block_end = _clamp_int(sparse_block_end, 0, 27)
        self.sparse_step_start = _clamp_int(sparse_step_start, 0, 150)
        self.sparse_step_end = _clamp_int(sparse_step_end, -1, 150)
        self.sparse_local_window = _odd_int(sparse_local_window, 3, 63)
        self.sparse_dilation = _clamp_int(sparse_dilation, 1, 8)
        self.sparse_full_attention_interval = _clamp_int(sparse_full_attention_interval, 0, 64)
        self.cond_uncond_enabled = bool(cond_uncond_enabled)
        self.cond_uncond_skip_cfg1 = bool(cond_uncond_skip_cfg1)
        self.cond_uncond_schedule_enabled = bool(cond_uncond_schedule_enabled)
        self.cond_uncond_guidance_interval = _clamp_int(cond_uncond_guidance_interval, 1, 64)
        self.lowbit_enabled = bool(lowbit_enabled)
        self.compile_enabled = bool(compile_enabled)
        self.teacache_enabled = bool(teacache_enabled)
        self.teacache_preset = (
            teacache_preset if teacache_preset in TEACACHE_PRESETS else TEACACHE_PRESET_BALANCED
        )
        self.teacache_threshold = _clamp_float(teacache_threshold, 0.0, 1.0)
        self.teacache_start_percent = _clamp_float(teacache_start_percent, 0.0, 1.0)
        self.teacache_end_percent = _clamp_float(teacache_end_percent, 0.0, 1.0)
        self.teacache_cache_device = (
            teacache_cache_device if teacache_cache_device in TEACACHE_CACHE_DEVICES else TEACACHE_CACHE_DEVICE_CUDA
        )
        self.teacache_modulated_source = (
            teacache_modulated_source
            if teacache_modulated_source in TEACACHE_MODULATED_SOURCES
            else TEACACHE_SOURCE_FIRST_BLOCK_SHIFT
        )
        self.teacache_coefficient_profile = (
            teacache_coefficient_profile
            if teacache_coefficient_profile in TEACACHE_COEFFICIENT_PROFILES
            else TEACACHE_PROFILE_ANIMA_2B_30STEP_FIRST_BLOCK_SHIFT
        )
        self.teacache_max_skip_streak = _clamp_int(teacache_max_skip_streak, 0, 64)
        self.teacache_force_full_interval = _clamp_int(teacache_force_full_interval, 0, 64)
        self.teacache_dry_run = bool(teacache_dry_run)
        self.teacache_verbose_trace = bool(teacache_verbose_trace)
        self._apply_teacache_preset()
        self.ujicache_enabled = bool(ujicache_enabled)
        self.ujicache_preset = (
            ujicache_preset if ujicache_preset in UJICACHE_PRESETS else UJICACHE_PRESET_CUSTOM
        )
        self.ujicache_threshold = _clamp_float(ujicache_threshold, 0.0, 1.0)
        self.ujicache_start_percent = _clamp_float(ujicache_start_percent, 0.0, 1.0)
        self.ujicache_end_percent = _clamp_float(ujicache_end_percent, 0.0, 1.0)
        self.ujicache_formula = (
            ujicache_formula if ujicache_formula in UJICACHE_FORMULAS else UJICACHE_FORMULA_TEACACHE
        )
        self.ujicache_use_prediction_after_progress = _clamp_float(
            ujicache_use_prediction_after_progress,
            0.0,
            1.0,
        )
        self.ujicache_apply_prediction_from_skip = _clamp_int(
            ujicache_apply_prediction_from_skip,
            1,
            3,
        )
        self.ujicache_prediction_strength = _clamp_float(ujicache_prediction_strength, 0.0, 1.0)
        self.ujicache_taylor2_curve_strength = _clamp_float(ujicache_taylor2_curve_strength, 0.0, 1.0)
        self.ujicache_slope_ema_smoothing = _clamp_float(ujicache_slope_ema_smoothing, 0.0, 0.99)
        self.ujicache_curve_ema_smoothing = _clamp_float(ujicache_curve_ema_smoothing, 0.0, 0.99)
        self.ujicache_cache_device = (
            ujicache_cache_device if ujicache_cache_device in TEACACHE_CACHE_DEVICES else TEACACHE_CACHE_DEVICE_CUDA
        )
        self.ujicache_modulated_source = (
            ujicache_modulated_source
            if ujicache_modulated_source in TEACACHE_MODULATED_SOURCES
            else TEACACHE_SOURCE_FIRST_BLOCK_SHIFT
        )
        self.ujicache_coefficient_profile = (
            ujicache_coefficient_profile
            if ujicache_coefficient_profile in TEACACHE_COEFFICIENT_PROFILES
            else TEACACHE_PROFILE_ANIMA_2B_30STEP_FIRST_BLOCK_SHIFT
        )
        self.ujicache_max_skip_streak = _clamp_int(ujicache_max_skip_streak, 0, 64)
        self.ujicache_force_full_interval = _clamp_int(ujicache_force_full_interval, 0, 64)
        self.ujicache_dry_run = bool(ujicache_dry_run)
        self.ujicache_verbose_trace = bool(ujicache_verbose_trace)
        self.auto_ujicache_enabled = bool(auto_ujicache_enabled) and self.ujicache_enabled
        self.auto_ujicache_csv = str(auto_ujicache_csv or "")
        if self.ujicache_enabled:
            self.teacache_enabled = False
        self.spectrum_enabled = bool(spectrum_enabled)
        if (self.teacache_enabled or self.ujicache_enabled) and self.spectrum_enabled:
            self.spectrum_enabled = False
        self.spectrum_preset = (
            spectrum_preset if spectrum_preset in SPECTRUM_PRESETS else SPECTRUM_PRESET_BALANCED
        )
        self.spectrum_w = _clamp_float(spectrum_w, 0.0, 1.0)
        self.spectrum_m = _clamp_int(spectrum_m, 1, 32)
        self.spectrum_lambda = _clamp_float(spectrum_lambda, 0.0, 100.0)
        self.spectrum_warmup_steps = _clamp_int(spectrum_warmup_steps, 0, 50)
        self.spectrum_window_size = _clamp_int(spectrum_window_size, 1, 64)
        self.spectrum_flex_window = _clamp_float(spectrum_flex_window, 0.0, 2.0)
        self.spectrum_stop_progress = _clamp_float(spectrum_stop_progress, 0.0, 1.0)
        self.spectrum_dry_run = bool(spectrum_dry_run)
        self.spectrum_verbose_trace = bool(spectrum_verbose_trace)
        self._apply_spectrum_preset()
        self.dump_teacache_residual = bool(dump_teacache_residual)
        self.dump_ujicache_residual = bool(dump_ujicache_residual)
        self.dump_block_output = bool(dump_block_output)
        self.dump_cross_attention_output = bool(dump_cross_attention_output)
        self.dump_mlp_output = bool(dump_mlp_output)
        self.dump_spectrum_final_output = bool(dump_spectrum_final_output)
        self.dump_baseline_final_output = bool(dump_baseline_final_output)
        if self.sparse_block_start > self.sparse_block_end:
            self.sparse_block_start, self.sparse_block_end = (
                self.sparse_block_end,
                self.sparse_block_start,
            )
        if self.attention_block_start > self.attention_block_end:
            self.attention_block_start, self.attention_block_end = (
                self.attention_block_end,
                self.attention_block_start,
            )
        if self.teacache_start_percent > self.teacache_end_percent:
            self.teacache_start_percent, self.teacache_end_percent = (
                self.teacache_end_percent,
                self.teacache_start_percent,
            )
        if self.ujicache_start_percent > self.ujicache_end_percent:
            self.ujicache_start_percent, self.ujicache_end_percent = (
                self.ujicache_end_percent,
                self.ujicache_start_percent,
            )

    def active(self) -> bool:
        return self.enabled and (
            self.mode != MODE_OFF
            or self.attention_override_active()
            or self.sparse_enabled
            or self.teacache_enabled
            or self.ujicache_enabled
            or self.spectrum_enabled
            or self.cond_uncond_enabled
            or self.lowbit_enabled
            or self.compile_enabled
            or self.tensor_dump_active()
        )

    def experimental_active(self) -> bool:
        return (
            self.attention_override_active()
            or self.sparse_enabled
            or self.teacache_enabled
            or self.ujicache_enabled
            or self.spectrum_enabled
            or self.cond_uncond_enabled
            or self.lowbit_enabled
            or self.compile_enabled
        )

    def attention_override_active(self) -> bool:
        return self.attention_enabled and self.attention_backend != ATTENTION_BACKEND_CURRENT

    def tensor_dump_requested(self) -> bool:
        return (
            self.dump_teacache_residual
            or self.dump_ujicache_residual
            or self.dump_block_output
            or self.dump_cross_attention_output
            or self.dump_mlp_output
            or self.dump_spectrum_final_output
            or self.dump_baseline_final_output
        )

    def tensor_dump_active(self) -> bool:
        return self.enabled and self.debug_log_enabled and self.tensor_dump_requested()

    def tensor_dump_block_level_active(self) -> bool:
        return self.tensor_dump_active() and (
            self.dump_block_output
            or self.dump_cross_attention_output
            or self.dump_mlp_output
        )

    def _apply_teacache_preset(self) -> None:
        if self.teacache_preset == TEACACHE_PRESET_CUSTOM:
            return
        if self.teacache_preset == TEACACHE_PRESET_SAFE:
            self.teacache_threshold = 0.06
        elif self.teacache_preset == TEACACHE_PRESET_AGGRESSIVE:
            self.teacache_threshold = 0.08
        else:
            self.teacache_threshold = 0.07
        self.teacache_start_percent = 0.05
        self.teacache_end_percent = 0.95

    def _apply_spectrum_preset(self) -> None:
        if self.spectrum_preset == SPECTRUM_PRESET_CUSTOM:
            return
        if self.spectrum_preset == SPECTRUM_PRESET_SAFE:
            self.spectrum_w = 0.20
            self.spectrum_m = 8
            self.spectrum_lambda = 0.50
            self.spectrum_warmup_steps = 8
            self.spectrum_window_size = 2
            self.spectrum_flex_window = 0.0
            self.spectrum_stop_progress = 0.80
        elif self.spectrum_preset == SPECTRUM_PRESET_AGGRESSIVE:
            self.spectrum_w = 0.30
            self.spectrum_m = 16
            self.spectrum_lambda = 0.50
            self.spectrum_warmup_steps = 6
            self.spectrum_window_size = 2
            self.spectrum_flex_window = 0.0
            self.spectrum_stop_progress = 0.90
        else:
            self.spectrum_w = 0.20
            self.spectrum_m = 16
            self.spectrum_lambda = 0.50
            self.spectrum_warmup_steps = 6
            self.spectrum_window_size = 2
            self.spectrum_flex_window = 0.0
            self.spectrum_stop_progress = 0.80

    def reset_generation(self, source: str = "unknown") -> None:
        self.generation_index += 1
        self.generation_steps = None
        self.generation_start = perf_counter()
        self.generation_start_source = source
        self.step_start = None
        self.step_durations.clear()
        self.denoiser_calls = 0
        self.cond_trace_logged = False
        self.cond_batch_trace_logged = False
        self.block_trace_call_count = 0
        self.block_trace_qkv_logged.clear()
        self.current_block_index = None
        self.identity_patch_calls = 0
        self.identity_patch_logged_calls = 0
        self.identity_patch_shape_mismatches = 0
        self.identity_patch_errors = 0
        self.identity_patch_num_blocks = None
        self.attention_kernel_calls = 0
        self.attention_kernel_block_calls = 0
        self.attention_kernel_fallbacks = 0
        self.attention_kernel_errors = 0
        self.attention_kernel_logged_calls = 0
        self.attention_kernel_num_blocks = None
        self.attention_kernel_current_context = None
        self.attention_kernel_actual_counts.clear()
        self.attention_kernel_internal_fallbacks = 0
        self.attention_kernel_internal_errors = 0
        self.attention_kernel_last_trace = None
        self.sparse_block_calls = 0
        self.sparse_attention_calls = 0
        self.sparse_fallbacks = 0
        self.sparse_errors = 0
        self.sparse_logged_calls = 0
        self.sparse_num_blocks = None
        self.sparse_current_context = None
        self.sparse_unavailable_reason = None
        self.teacache_model_calls = 0
        self.teacache_full_calcs = 0
        self.teacache_skips = 0
        self.teacache_dry_run_skips = 0
        self.teacache_first_full_calcs = 0
        self.teacache_forced_full_calcs = 0
        self.teacache_fallbacks = 0
        self.teacache_errors = 0
        self.teacache_logged_calls = 0
        self.teacache_num_blocks = None
        self.teacache_unavailable_reason = None
        self.ujicache_model_calls = 0
        self.ujicache_full_calcs = 0
        self.ujicache_skips = 0
        self.ujicache_skipped_steps.clear()
        self.ujicache_prediction_used = 0
        self.ujicache_fallback_used = 0
        self.ujicache_dry_run_predictions = 0
        self.ujicache_first_full_calcs = 0
        self.ujicache_forced_full_calcs = 0
        self.ujicache_fallbacks = 0
        self.ujicache_errors = 0
        self.ujicache_logged_calls = 0
        self.ujicache_num_blocks = None
        self.ujicache_unavailable_reason = None
        self.ujicache_fallback_reasons.clear()
        self.spectrum_model_calls = 0
        self.spectrum_actual_forwards = 0
        self.spectrum_forecasts = 0
        self.spectrum_dry_run_forecasts = 0
        self.spectrum_fallbacks = 0
        self.spectrum_errors = 0
        self.spectrum_logged_calls = 0
        self.spectrum_unavailable_reason = None
        self.tensor_dump_run_dir = None
        self.tensor_dump_initialized = False
        self.tensor_dump_records = 0
        self.tensor_dump_errors = 0
        self.tensor_dump_unavailable_reason = None
        self.tensor_dump_block_call_index = 0
        self.tensor_dump_block_local_call_index = 0
        self.tensor_dump_cross_attention_local_call_index = 0
        self.tensor_dump_mlp_local_call_index = 0
        self.tensor_dump_teacache_local_call_index = 0
        self.tensor_dump_ujicache_local_call_index = 0
        self.tensor_dump_spectrum_local_call_index = 0
        self.tensor_dump_baseline_local_call_index = 0
        self.tensor_dump_current_context = None
        self.tensor_dump_num_blocks = None
        self.tensor_dump_warned_reasons.clear()
        self.generation_logged = False
        self.error_message = None
        if not self.enabled:
            self.status = "disabled"
        elif self.mode == MODE_IDENTITY_PATCH:
            self.status = "identity-patch"
        elif self.teacache_enabled:
            self.status = "experimental-teacache"
        elif self.ujicache_enabled:
            self.status = "experimental-ujicache"
        elif self.spectrum_enabled:
            self.status = "experimental-spectrum"
        elif self.attention_override_active():
            self.status = "experimental-attention"
        elif self.sparse_enabled:
            self.status = "experimental-sparse"
        elif self.mode == MODE_OFF:
            self.status = "disabled"
        elif self.mode == MODE_DIAGNOSE:
            self.status = "diagnosing"
        else:
            self.status = "trace"

    def mark_step_start(self) -> None:
        self.step_start = perf_counter()
        self.denoiser_calls += 1

    def mark_step_end(self) -> None:
        if self.step_start is None:
            return
        self.step_durations.append(perf_counter() - self.step_start)
        self.step_start = None

    def total_sampling_time(self) -> float | None:
        if self.generation_start is None:
            return None
        return perf_counter() - self.generation_start

    def avg_step_time(self) -> float | None:
        if not self.step_durations:
            return None
        return sum(self.step_durations) / len(self.step_durations)

    def set_error(self, message: str) -> None:
        self.status = "error"
        self.error_message = message


STATE = RuntimeState()


def _clamp_int(value: Any, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except Exception:
        number = minimum
    return max(minimum, min(maximum, number))


def _normalize_mode(value: Any) -> str:
    mode = str(value or MODE_OFF)
    if mode == "Off":
        return MODE_OFF
    if mode == "Identity patch test":
        return MODE_IDENTITY_PATCH
    return mode if mode in MODES else MODE_OFF


def _clamp_float(value: Any, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except Exception:
        number = minimum
    return max(minimum, min(maximum, number))


def _odd_int(value: Any, minimum: int, maximum: int) -> int:
    number = _clamp_int(value, minimum, maximum)
    if number % 2 == 0:
        number += 1
    if number > maximum:
        number -= 2
    return max(minimum, number)
