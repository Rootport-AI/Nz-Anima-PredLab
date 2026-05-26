from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any


MODE_OFF = "Off"
MODE_DIAGNOSE = "Diagnose only"
MODE_TRACE_ATTENTION = "Trace attention"
MODE_TRACE_COND = "Trace cond/uncond"
MODE_TRACE_LOWBIT = "Trace low-bit / compile"

MODES = [
    MODE_OFF,
    MODE_DIAGNOSE,
    MODE_TRACE_ATTENTION,
    MODE_TRACE_COND,
    MODE_TRACE_LOWBIT,
]


@dataclass
class RuntimeState:
    enabled: bool = False
    mode: str = MODE_OFF
    print_timing_log: bool = True
    verbose_diagnose_log: bool = False
    status: str = "disabled"
    error_message: str | None = None
    model_detection: Any | None = None
    warned_model_keys: set[str] = field(default_factory=set)
    generation_start: float | None = None
    step_start: float | None = None
    step_durations: list[float] = field(default_factory=list)
    denoiser_calls: int = 0
    cond_trace_logged: bool = False
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
    ) -> None:
        self.enabled = bool(enabled)
        self.mode = mode if mode in MODES else MODE_OFF
        self.print_timing_log = bool(print_timing_log)
        self.verbose_diagnose_log = bool(verbose_diagnose_log)

    def active(self) -> bool:
        return self.enabled and self.mode != MODE_OFF

    def reset_generation(self, source: str = "unknown") -> None:
        self.generation_start = perf_counter()
        self.generation_start_source = source
        self.step_start = None
        self.step_durations.clear()
        self.denoiser_calls = 0
        self.cond_trace_logged = False
        self.generation_logged = False
        self.error_message = None
        if not self.enabled or self.mode == MODE_OFF:
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
