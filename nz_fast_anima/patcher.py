from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .logging import info, warning
from .state import STATE


@dataclass
class PatchResult:
    ok: bool
    kind: str
    message: str = ""


def apply_patch(kind: str, context: Any = None) -> PatchResult:
    warning(f"patch '{kind}' is not implemented in the diagnostic build")
    return PatchResult(False, kind, "not implemented")


def remove_patch(kind: str) -> PatchResult:
    patch = STATE.patches.pop(kind, None)
    if patch is None:
        return PatchResult(True, kind, "not patched")
    try:
        restore = patch["restore"]
        restore()
        info(f"removed patch kind={kind}")
        return PatchResult(True, kind, "removed")
    except Exception as exc:
        STATE.set_error(f"failed to remove patch {kind}: {exc}")
        return PatchResult(False, kind, str(exc))


def remove_all_patches() -> PatchResult:
    ok = True
    messages: list[str] = []
    for kind in list(STATE.patches.keys()):
        result = remove_patch(kind)
        ok = ok and result.ok
        messages.append(f"{kind}:{result.message}")
    return PatchResult(ok, "all", ", ".join(messages))


def is_patched(kind: str) -> bool:
    return kind in STATE.patches
