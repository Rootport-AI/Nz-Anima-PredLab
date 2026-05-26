from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any


MODE_OFF = "Off"
MODE_DIAGNOSE = "Diagnose only"
MODE_IDENTITY_PATCH = "Identity patch test"
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


@dataclass
class RuntimeState:
    enabled: bool = False
    mode: str = MODE_OFF
    print_timing_log: bool = True
    verbose_diagnose_log: bool = False
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
    status: str = "disabled"
    error_message: str | None = None
    model_detection: Any | None = None
    warned_model_keys: set[str] = field(default_factory=set)
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
    sparse_block_calls: int = 0
    sparse_attention_calls: int = 0
    sparse_fallbacks: int = 0
    sparse_errors: int = 0
    sparse_logged_calls: int = 0
    sparse_num_blocks: int | None = None
    sparse_current_context: dict[str, Any] | None = None
    sparse_unavailable_reason: str | None = None
    generation_logged: bool = False
    generation_start_source: str | None = None
    patches: dict[str, Any] = field(default_factory=dict)

    def refresh_settings(self) -> None:
        try:
            from modules import shared

            opts = shared.opts
            self.enabled = bool(getattr(opts, "nzfa_enable", False))
            self.mode = str(getattr(opts, "nzfa_mode", MODE_OFF))
            self.print_timing_log = bool(getattr(opts, "nzfa_print_timing_log", True))
            self.verbose_diagnose_log = bool(
                getattr(opts, "nzfa_verbose_diagnose_log", False)
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
    ) -> None:
        self.enabled = bool(enabled)
        self.mode = mode if mode in MODES else MODE_OFF
        self.print_timing_log = bool(print_timing_log)
        self.verbose_diagnose_log = bool(verbose_diagnose_log)
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

    def active(self) -> bool:
        return self.enabled and (
            self.mode != MODE_OFF
            or self.sparse_enabled
            or self.cond_uncond_enabled
            or self.lowbit_enabled
            or self.compile_enabled
        )

    def experimental_active(self) -> bool:
        return (
            self.sparse_enabled
            or self.cond_uncond_enabled
            or self.lowbit_enabled
            or self.compile_enabled
        )

    def reset_generation(self, source: str = "unknown") -> None:
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
        self.sparse_block_calls = 0
        self.sparse_attention_calls = 0
        self.sparse_fallbacks = 0
        self.sparse_errors = 0
        self.sparse_logged_calls = 0
        self.sparse_num_blocks = None
        self.sparse_current_context = None
        self.sparse_unavailable_reason = None
        self.generation_logged = False
        self.error_message = None
        if not self.enabled:
            self.status = "disabled"
        elif self.mode == MODE_IDENTITY_PATCH:
            self.status = "identity-patch"
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


def _odd_int(value: Any, minimum: int, maximum: int) -> int:
    number = _clamp_int(value, minimum, maximum)
    if number % 2 == 0:
        number += 1
    if number > maximum:
        number -= 2
    return max(minimum, number)
