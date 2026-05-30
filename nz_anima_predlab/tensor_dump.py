from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from .logging import info, warning
from .model_detect import ModelDetection
from .state import STATE

DUMP_RAW_BLOCKS = {0, 7, 14, 21, 27}
DUMP_MAX_RECORDS_PER_TYPE = 100000
DUMP_SAVE_DTYPE = "float16"

_deps_checked = False
_deps_available = False
_deps_error = ""
_zarr = None
_pd = None
_root_group = None
_stats: list[dict[str, Any]] = []
_arrays: dict[str, Any] = {}
_record_counts_by_path: dict[str, int] = {}
_record_counts_by_type: dict[str, int] = {}


def any_tensor_dump_enabled() -> bool:
    return STATE.tensor_dump_active()


def initialize_run_if_needed(p: Any = None) -> None:
    global _root_group, _stats, _arrays, _record_counts_by_path, _record_counts_by_type

    if not any_tensor_dump_enabled():
        return
    if STATE.tensor_dump_initialized:
        return
    if not _ensure_dependencies():
        _mark_unavailable(_deps_error)
        return

    run_dir = ensure_run_dir(p)
    if run_dir is None:
        _mark_unavailable("run_dir_unavailable")
        return

    try:
        _stats = []
        _arrays = {}
        _record_counts_by_path = {}
        _record_counts_by_type = {}
        tensors_dir = run_dir / "tensors.zarr"
        _root_group = _zarr.open_group(str(tensors_dir), mode="a")
        STATE.tensor_dump_run_dir = str(run_dir)
        STATE.tensor_dump_initialized = True
        _write_meta(run_dir, p)
        info(f"tensor_dump_initialized run_dir={run_dir}")
    except Exception as exc:
        STATE.tensor_dump_errors += 1
        _mark_unavailable(f"initialize_failed:{_short_error(exc)}")


def ensure_run_dir(p: Any = None) -> Path | None:
    try:
        now = datetime.now().astimezone()
        base_dir = _infer_log_base_dir(p, now)
        run_id = f"run_{now:%Y%m%d_%H%M%S}_gen{STATE.generation_index:04d}"
        run_dir = base_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
    except Exception as exc:
        STATE.tensor_dump_errors += 1
        warning(f"tensor_dump_run_dir_failed reason={_short_error(exc)}")
        return None


def dump_tensor(
    tensor_type: str,
    tensor: Any,
    *,
    logical_step_index: int | None = None,
    step_index: int | None = None,
    local_call_index: int | None = None,
    call_index: int | None = None,
    block_call_index: int | None = None,
    block_index: int | None = None,
    slot: int | None = None,
    decision: str | None = None,
    attn_type: str | None = None,
    timestep_value: Any = None,
    teacache_model_call: int | None = None,
    spectrum_cnt: int | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    if not any_tensor_dump_enabled():
        return
    if not _ensure_dependencies():
        _mark_unavailable(_deps_error)
        return

    try:
        import torch

        if not torch.is_tensor(tensor):
            return
        initialize_run_if_needed()
        if not STATE.tensor_dump_initialized:
            return

        type_count = _record_counts_by_type.get(tensor_type, 0)
        if type_count >= DUMP_MAX_RECORDS_PER_TYPE:
            _warn_once(
                f"max_records_{tensor_type}",
                f"tensor_dump_skipped type={tensor_type} reason=max_records",
            )
            return
        _record_counts_by_type[tensor_type] = type_count + 1

        logical_step = (
            int(logical_step_index)
            if logical_step_index is not None
            else int(step_index)
            if step_index is not None
            else None
        )
        tensor_cpu = tensor.detach().cpu()
        stats = _tensor_stats(tensor_cpu)
        zarr_path = _zarr_path(tensor_type, block_index, slot, decision)
        save_raw = _should_save_raw(tensor_type, block_index)
        record_index = _record_counts_by_path.get(zarr_path, 0)
        saved_dtype = ""

        if save_raw:
            saved = _tensor_for_save(tensor_cpu)
            saved_dtype = str(saved.dtype).replace("torch.", "")
            _append_zarr(zarr_path, saved)
            _record_counts_by_path[zarr_path] = record_index + 1
        else:
            zarr_path = ""
            record_index = -1

        _stats.append(
            {
                "schema_version": 1,
                "run_id": _run_id(),
                "generation_index": STATE.generation_index,
                "tensor_type": tensor_type,
                "record_index": record_index,
                "logical_step_index": logical_step,
                "local_call_index": local_call_index,
                "call_index": call_index,
                "block_call_index": block_call_index,
                "block_index": block_index,
                "slot": slot,
                "attn_type": attn_type,
                "decision": decision,
                "timestep_value": _safe_float(timestep_value),
                "teacache_model_call": teacache_model_call,
                "spectrum_cnt": spectrum_cnt,
                "shape": "x".join(str(part) for part in tensor_cpu.shape),
                "dtype": str(tensor_cpu.dtype).replace("torch.", ""),
                "saved_dtype": saved_dtype,
                "numel": int(tensor_cpu.numel()),
                **stats,
                "zarr_path": zarr_path,
                "extra_json": json.dumps(extra or {}, default=str, sort_keys=True),
            }
        )
        STATE.tensor_dump_records += 1
    except Exception as exc:
        STATE.tensor_dump_errors += 1
        warning(f"tensor_dump_failed type={tensor_type} reason={_short_error(exc)}")


def flush_stats() -> None:
    if not STATE.tensor_dump_initialized or not STATE.tensor_dump_run_dir:
        return
    if not _stats:
        return
    if not _ensure_dependencies():
        _mark_unavailable(_deps_error)
        return
    try:
        run_dir = Path(STATE.tensor_dump_run_dir)
        frame = _pd.DataFrame(_stats)
        frame.to_parquet(run_dir / "stats.parquet", index=False)
        info(
            "tensor_dump_summary="
            f"records={STATE.tensor_dump_records} errors={STATE.tensor_dump_errors} "
            f"run_dir={run_dir}"
        )
    except Exception as exc:
        STATE.tensor_dump_errors += 1
        warning(f"tensor_dump_flush_failed reason={_short_error(exc)}")


def _ensure_dependencies() -> bool:
    global _deps_checked, _deps_available, _deps_error, _zarr, _pd

    if _deps_checked:
        return _deps_available
    _deps_checked = True
    try:
        import pandas as pd
        import pyarrow  # noqa: F401
        import zarr

        _pd = pd
        _zarr = zarr
        _deps_available = True
        return True
    except Exception as first_exc:
        install_result = _install_dependencies()
        if install_result:
            try:
                import pandas as pd
                import pyarrow  # noqa: F401
                import zarr

                _pd = pd
                _zarr = zarr
                _deps_available = True
                return True
            except Exception as second_exc:
                _deps_error = f"dependency_import_failed_after_install:{_short_error(second_exc)}"
        else:
            _deps_error = f"dependency_import_failed:{_short_error(first_exc)}"
        _deps_available = False
        return False


def _install_dependencies() -> bool:
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "zarr",
                "pandas",
                "pyarrow",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode == 0:
            info("tensor_dump_dependencies_installed packages=zarr,pandas,pyarrow")
            return True
        stderr = (result.stderr or result.stdout or "").strip()
        _mark_unavailable(f"dependency_install_failed:{_short_error(stderr)}")
        return False
    except Exception as exc:
        _mark_unavailable(f"dependency_install_failed:{_short_error(exc)}")
        return False


def _infer_log_base_dir(p: Any, now: datetime) -> Path:
    candidates = []
    for attr in ("outpath_samples", "outpath_grids", "outdir_samples"):
        value = getattr(p, attr, None) if p is not None else None
        if value:
            candidates.append(value)
    try:
        from modules import shared

        opts = shared.opts
        for key in ("outdir_txt2img_samples", "outdir_samples", "outdir_save"):
            value = getattr(opts, key, None)
            if value:
                candidates.append(value)
    except Exception:
        pass

    for value in candidates:
        try:
            path = Path(str(value)).expanduser()
            if path.suffix:
                path = path.parent
            images_dir = _images_dir_from_path(path)
            return images_dir / "logs" / f"{now:%Y-%m-%d}"
        except Exception:
            continue
    return Path.cwd() / "logs" / f"{now:%Y-%m-%d}"


def _images_dir_from_path(path: Path) -> Path:
    parts = path.parts
    for index, part in enumerate(parts):
        if part.lower() == "images":
            return Path(*parts[: index + 1])
    return path


def _write_meta(run_dir: Path, p: Any) -> None:
    now = datetime.now().astimezone()
    detection = STATE.model_detection
    evidence = detection.evidence if isinstance(detection, ModelDetection) else {}
    meta = {
        "schema_version": 1,
        "extension": "Nz-Anima-PredLab",
        "model": evidence.get("checkpoint_name") or evidence.get("filename") or "unknown",
        "generation_index": STATE.generation_index,
        "created_at": now.isoformat(),
        "steps": _safe_int(getattr(p, "steps", None)),
        "width": _safe_int(getattr(p, "width", None)),
        "height": _safe_int(getattr(p, "height", None)),
        "batch_size": _safe_int(getattr(p, "batch_size", None)),
        "sampler": str(getattr(p, "sampler_name", "unknown")),
        "scheduler": str(getattr(p, "scheduler", "unknown")),
        "cfg_scale": _safe_float(getattr(p, "cfg_scale", None)),
        "dump_flags": {
            "teacache_residual": STATE.dump_teacache_residual,
            "block_output": STATE.dump_block_output,
            "cross_attention_output": STATE.dump_cross_attention_output,
            "mlp_output": STATE.dump_mlp_output,
            "spectrum_final_output": STATE.dump_spectrum_final_output,
            "baseline_final_output": STATE.dump_baseline_final_output,
        },
        "raw_blocks": sorted(DUMP_RAW_BLOCKS),
        "save_dtype": DUMP_SAVE_DTYPE,
    }
    (run_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _tensor_stats(tensor_cpu: Any) -> dict[str, float]:
    import torch

    if tensor_cpu.numel() == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "abs_mean": 0.0,
            "l1_norm": 0.0,
            "l2_norm": 0.0,
            "max_abs": 0.0,
        }
    x = tensor_cpu.float()
    abs_x = x.abs()
    return {
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "abs_mean": float(abs_x.mean().item()),
        "l1_norm": float(abs_x.sum().item()),
        "l2_norm": float(torch.linalg.vector_norm(x).item()),
        "max_abs": float(abs_x.max().item()),
    }


def _tensor_for_save(tensor_cpu: Any):
    import torch

    if DUMP_SAVE_DTYPE == "float16" and tensor_cpu.is_floating_point():
        return tensor_cpu.to(torch.float16).contiguous()
    return tensor_cpu.contiguous()


def _append_zarr(zarr_path: str, tensor_cpu: Any) -> None:
    import numpy as np

    array = _array_for_path(zarr_path, tuple(tensor_cpu.shape), tensor_cpu.numpy().dtype)
    index = int(array.shape[0])
    array.resize((index + 1, *array.shape[1:]))
    array[index] = np.asarray(tensor_cpu.numpy())


def _array_for_path(zarr_path: str, tensor_shape: tuple[int, ...], dtype: Any):
    if zarr_path in _arrays:
        return _arrays[zarr_path]
    if _root_group is None:
        raise RuntimeError("zarr root group is not initialized")
    group = _root_group
    parts = zarr_path.split("/")
    for part in parts[:-1]:
        group = _require_group(group, part)
    name = parts[-1]
    try:
        array = group[name]
    except Exception:
        shape = (0, *tensor_shape)
        chunks = (1, *tensor_shape)
        array = _create_array(group, name, shape, chunks, dtype)
    _arrays[zarr_path] = array
    return array


def _require_group(group: Any, name: str):
    try:
        return group.require_group(name)
    except Exception:
        try:
            return group[name]
        except Exception:
            return group.create_group(name)


def _create_array(group: Any, name: str, shape: tuple[int, ...], chunks: tuple[int, ...], dtype: Any):
    try:
        return group.create_dataset(name, shape=shape, chunks=chunks, dtype=dtype)
    except TypeError:
        return group.create_array(name, shape=shape, chunks=chunks, dtype=dtype)


def _zarr_path(
    tensor_type: str,
    block_index: int | None,
    slot: int | None,
    decision: str | None,
) -> str:
    if tensor_type == "teacache_residual":
        return f"{tensor_type}/slot_{slot if slot is not None else 'unknown'}"
    if tensor_type in {"spectrum_final_output", "baseline_final_output"}:
        return f"{tensor_type}/{decision or 'actual'}"
    if tensor_type in {"block_output", "cross_attention_output", "mlp_output"}:
        suffix = f"{int(block_index):02d}" if block_index is not None else "unknown"
        return f"{tensor_type}/block_{suffix}"
    return f"{tensor_type}/all"


def _should_save_raw(tensor_type: str, block_index: int | None) -> bool:
    if tensor_type in {
        "teacache_residual",
        "spectrum_final_output",
        "baseline_final_output",
    }:
        return True
    if tensor_type in {"block_output", "cross_attention_output", "mlp_output"}:
        return isinstance(block_index, int) and block_index in DUMP_RAW_BLOCKS
    return True


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        import torch

        if torch.is_tensor(value):
            if value.numel() == 0:
                return None
            return float(value.detach().float().flatten()[0].cpu().item())
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return None


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _run_id() -> str:
    if not STATE.tensor_dump_run_dir:
        return ""
    return Path(STATE.tensor_dump_run_dir).name


def _mark_unavailable(reason: str) -> None:
    if not reason:
        reason = "unknown"
    STATE.tensor_dump_unavailable_reason = reason
    _warn_once(reason, f"tensor_dump_unavailable reason={reason}")


def _warn_once(key: str, message: str) -> None:
    if key in STATE.tensor_dump_warned_reasons:
        return
    STATE.tensor_dump_warned_reasons.add(key)
    warning(message)


def _short_error(value: Any) -> str:
    text = str(value).replace("\n", " ").strip()
    if len(text) > 220:
        return text[:217] + "..."
    return text
