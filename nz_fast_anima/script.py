from __future__ import annotations

import gradio as gr
import modules.scripts as scripts

from .callbacks import register_callbacks
from .diagnostics import log_generation_start, log_timing_summary
from .logging import exception
from .model_detect import detect_model
from .state import MODE_OFF, MODES, STATE
from .timing import start_sampling

register_callbacks()


class Script(scripts.Script):
    def title(self):
        return "Nz-fast-anima"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion("Nz-fast-anima", open=False, elem_id="nzfa-panel"):
            enabled = gr.Checkbox(
                label="Enable Nz-fast-anima",
                value=_default_option("nzfa_enable", False),
                elem_id="nzfa-enable",
            )
            mode = gr.Dropdown(
                label="Nz-fast-anima mode",
                choices=MODES,
                value=_default_option("nzfa_mode", MODE_OFF),
                elem_id="nzfa-mode",
            )
            print_timing_log = gr.Checkbox(
                label="Print timing log",
                value=_default_option("nzfa_print_timing_log", True),
                elem_id="nzfa-print-timing-log",
            )
            verbose_diagnose_log = gr.Checkbox(
                label="Verbose diagnose log",
                value=_default_option("nzfa_verbose_diagnose_log", False),
                elem_id="nzfa-verbose-diagnose-log",
            )
        return [enabled, mode, print_timing_log, verbose_diagnose_log]

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
    if len(script_args) >= 4:
        STATE.apply_options(
            script_args[0],
            script_args[1],
            script_args[2],
            script_args[3],
        )
        return
    STATE.refresh_settings()


def _begin_generation(p, script_args, source: str) -> None:
    _apply_ui_args(script_args)
    if not STATE.active():
        return

    start_sampling(source)

    try:
        from modules import shared

        STATE.model_detection = detect_model(getattr(shared, "sd_model", None))
    except Exception:
        if STATE.model_detection is None:
            raise

    log_generation_start(p)


def _default_option(key: str, default):
    try:
        from modules import shared

        return getattr(shared.opts, key, default)
    except Exception:
        return default
