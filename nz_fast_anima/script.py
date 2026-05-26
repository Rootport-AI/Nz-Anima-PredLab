from __future__ import annotations

import gradio as gr
import modules.scripts as scripts

from .callbacks import register_callbacks
from .diagnostics import log_generation_start, log_timing_summary
from .logging import exception
from .model_detect import detect_model
from .state import STATE
from .timing import start_sampling

register_callbacks()


class Script(scripts.Script):
    def title(self):
        return "Nz-fast-anima"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion("Nz-fast-anima", open=False, elem_id="nzfa-panel"):
            gr.Markdown(
                "Diagnostics are configured in Settings > Nz-fast-anima. "
                "Logs are printed to the Forge Neo console.",
                elem_id="nzfa-status",
            )
        return []

    def process_before_every_sampling(self, p, *script_args, **kwargs):
        try:
            STATE.refresh_settings()
            start_sampling()
            if not STATE.active():
                return

            try:
                from modules import shared

                STATE.model_detection = detect_model(getattr(shared, "sd_model", None))
            except Exception:
                if STATE.model_detection is None:
                    raise

            log_generation_start(p)
        except Exception as exc:
            STATE.set_error(f"process_before_every_sampling failed: {exc}")
            exception("process_before_every_sampling failed")

    def postprocess(self, p, processed, *script_args):
        try:
            log_timing_summary()
        except Exception as exc:
            STATE.set_error(f"postprocess failed: {exc}")
            exception("postprocess failed")
