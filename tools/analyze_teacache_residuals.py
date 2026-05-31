from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import zarr


DEFAULT_ROOT = Path(r"T:\StabilityMatrix\Images\logs\2026-05-30")

RUN_PAIRS = {
    "C_promptA": ("runC1_20260530_193131_gen0001", "runC2_20260530_193212_gen0002"),
    "D_promptB": ("runD1_20260530_193822_gen0004", "runD2_20260530_193908_gen0005"),
    "E_promptC": ("runE1_20260530_194052_gen0006", "runE2_20260530_194225_gen0007"),
    "F_seed": ("runF1_20260530_194533_gen0008", "runF2_20260530_194919_gen0010"),
    "G_resolution": ("runG1_20260530_195128_gen0011", "runG2_20260530_195319_gen0012"),
}

ALPHAS = [0.25, 0.5, 0.75, 1.0]
CURVE_BETAS = [0.25, 0.5, 0.75, 1.0]
CHEB_K_VALUES = [3, 4, 5, 6, 8]
CHEB_DEGREES = [1, 2, 3]
CHEB_RIDGES = [0.0, 1e-4, 1e-2]
BASELINE_ERROR_TOP_N = 4


@dataclass(frozen=True)
class TensorRef:
    step: int
    slot: int
    timestep: float
    zarr_path: str
    record_index: int


@dataclass
class RunData:
    name: str
    path: Path
    meta: dict
    frame: pd.DataFrame
    root: zarr.Group
    refs: dict[tuple[int, int], TensorRef]

    @property
    def steps(self) -> list[int]:
        return sorted({step for step, _slot in self.refs})

    @property
    def slots(self) -> list[int]:
        return sorted({slot for _step, slot in self.refs})

    def tensor(self, step: int, slot: int) -> np.ndarray:
        ref = self.refs[(step, slot)]
        array = self.root[ref.zarr_path]
        return np.asarray(array[ref.record_index], dtype=np.float32)

    def timestep(self, step: int, slot: int) -> float:
        return self.refs[(step, slot)].timestep


def load_run(path: Path) -> RunData:
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    frame = pd.read_parquet(path / "stats.parquet")
    root = zarr.open_group(str(path / "tensors.zarr"), mode="r")
    residuals = frame[frame["tensor_type"] == "teacache_residual"].copy()
    refs: dict[tuple[int, int], TensorRef] = {}
    for row in residuals.itertuples(index=False):
        step = int(row.logical_step_index)
        slot = int(row.slot)
        refs[(step, slot)] = TensorRef(
            step=step,
            slot=slot,
            timestep=float(row.timestep_value),
            zarr_path=str(row.zarr_path),
            record_index=int(row.record_index),
        )
    return RunData(path.name, path, meta, frame, root, refs)


def rel_l2(pred: np.ndarray, actual: np.ndarray) -> float:
    denom = float(np.linalg.norm(actual.ravel()))
    if denom == 0.0:
        return 0.0
    return float(np.linalg.norm((pred - actual).ravel()) / denom)


def rel_l1(pred: np.ndarray, actual: np.ndarray) -> float:
    denom = float(np.mean(np.abs(actual)))
    if denom == 0.0:
        return 0.0
    return float(np.mean(np.abs(pred - actual)) / denom)


def lagrange_weights(xs: list[float], target: float) -> list[float]:
    weights: list[float] = []
    for i, xi in enumerate(xs):
        weight = 1.0
        for j, xj in enumerate(xs):
            if i == j:
                continue
            denom = xi - xj
            if denom == 0.0:
                return []
            weight *= (target - xj) / denom
        weights.append(weight)
    return weights


def combine(points: list[tuple[float, np.ndarray]], target: float) -> np.ndarray | None:
    weights = lagrange_weights([x for x, _tensor in points], target)
    if not weights:
        return None
    result = np.zeros_like(points[-1][1], dtype=np.float32)
    for weight, (_x, tensor) in zip(weights, points):
        result += np.float32(weight) * tensor
    return result


def chebyshev_design(xs: np.ndarray, degree: int) -> np.ndarray:
    columns = [np.ones_like(xs, dtype=np.float64)]
    if degree >= 1:
        columns.append(xs)
    if degree >= 2:
        columns.append(2.0 * xs * xs - 1.0)
    if degree >= 3:
        columns.append(4.0 * xs * xs * xs - 3.0 * xs)
    return np.stack(columns, axis=1)


def scaled_positions(xs: list[int], target: int) -> tuple[np.ndarray, float]:
    lo = float(min(xs))
    hi = float(max(xs))
    scale = (hi - lo) * 0.5
    if scale == 0.0:
        return np.zeros(len(xs), dtype=np.float64), 0.0
    center = (lo + hi) * 0.5
    x_scaled = (np.asarray(xs, dtype=np.float64) - center) / scale
    target_scaled = (float(target) - center) / scale
    return x_scaled, target_scaled


def cheb_step_weights(
    history_steps: list[int],
    target_step: int,
    *,
    k: int,
    degree: int,
    ridge: float,
) -> dict[int, float] | None:
    if len(history_steps) < k:
        return None
    selected_steps = history_steps[-k:]
    x_scaled, target_scaled = scaled_positions(selected_steps, target_step)
    design = chebyshev_design(x_scaled, degree)
    target_design = chebyshev_design(np.asarray([target_scaled], dtype=np.float64), degree)[0]
    lhs = design.T @ design
    if ridge > 0.0:
        lhs = lhs + ridge * np.eye(lhs.shape[0], dtype=np.float64)
    try:
        inv_lhs = np.linalg.pinv(lhs)
    except np.linalg.LinAlgError:
        return None
    raw_weights = target_design @ inv_lhs @ design.T
    return {
        step: float(weight)
        for step, weight in zip(selected_steps, raw_weights)
    }


def merge_weights(*weighted_maps: tuple[float, dict[int, float]]) -> dict[int, float]:
    merged: dict[int, float] = {}
    for scale, weights in weighted_maps:
        for step, weight in weights.items():
            merged[step] = merged.get(step, 0.0) + scale * weight
    return {step: weight for step, weight in merged.items() if abs(weight) > 1e-12}


def damp_from_previous(
    previous_weights: dict[int, float],
    extrapolated_weights: dict[int, float],
    alpha: float,
) -> dict[int, float]:
    return merge_weights(
        (1.0 - alpha, previous_weights),
        (alpha, extrapolated_weights),
    )


def format_float(value: float) -> str:
    if value == 0.0:
        return "0"
    return f"{value:g}".replace("-", "m")


def candidate_weight_maps(history_steps: list[int], target_step: int) -> dict[str, dict[int, float]]:
    if not history_steps:
        return {}

    candidates: dict[str, dict[int, float]] = {}
    previous_weights = {history_steps[-1]: 1.0}
    candidates["previous"] = previous_weights

    linear_weights = None
    if len(history_steps) >= 2:
        weights = lagrange_weights([float(step) for step in history_steps[-2:]], float(target_step))
        linear_weights = {
            step: float(weight)
            for step, weight in zip(history_steps[-2:], weights)
        }
        for alpha in ALPHAS:
            candidates[f"linear_step_a{format_float(alpha)}"] = damp_from_previous(
                previous_weights,
                linear_weights,
                alpha,
            )

    taylor2_weights = None
    if len(history_steps) >= 3:
        weights = lagrange_weights([float(step) for step in history_steps[-3:]], float(target_step))
        taylor2_weights = {
            step: float(weight)
            for step, weight in zip(history_steps[-3:], weights)
        }
        for alpha in ALPHAS:
            candidates[f"taylor2_step_a{format_float(alpha)}"] = damp_from_previous(
                previous_weights,
                taylor2_weights,
                alpha,
            )
        if linear_weights is not None:
            for beta in CURVE_BETAS:
                candidates[f"taylor2_curve_b{format_float(beta)}"] = merge_weights(
                    (1.0 - beta, linear_weights),
                    (beta, taylor2_weights),
                )

    for k in CHEB_K_VALUES:
        for degree in CHEB_DEGREES:
            for ridge in CHEB_RIDGES:
                cheb_weights = cheb_step_weights(
                    history_steps,
                    target_step,
                    k=k,
                    degree=degree,
                    ridge=ridge,
                )
                if cheb_weights is None:
                    continue
                for alpha in ALPHAS:
                    candidates[
                        f"cheb_step_k{k}_d{degree}_r{format_float(ridge)}_a{format_float(alpha)}"
                    ] = damp_from_previous(
                        previous_weights,
                        cheb_weights,
                        alpha,
                    )

    return candidates


def interval_for_skip_order(order: int) -> str:
    if order < 4:
        return "skip_first4"
    if order < 8:
        return "skip_mid4"
    return "skip_last4"


def skip_streak_positions(skipped_steps: list[int]) -> dict[int, int]:
    positions: dict[int, int] = {}
    previous_step: int | None = None
    streak_pos = 0
    for step in skipped_steps:
        if previous_step is not None and step == previous_step + 1:
            streak_pos += 1
        else:
            streak_pos = 1
        positions[step] = streak_pos
        previous_step = step
    return positions


def predict_previous(history: list[tuple[int, np.ndarray]], _target_step: int) -> np.ndarray | None:
    if not history:
        return None
    return history[-1][1]


def predict_linear_step(history: list[tuple[int, np.ndarray]], target_step: int) -> np.ndarray | None:
    if len(history) < 2:
        return None
    points = [(float(step), tensor) for step, tensor in history[-2:]]
    return combine(points, float(target_step))


def predict_quadratic_step(history: list[tuple[int, np.ndarray]], target_step: int) -> np.ndarray | None:
    if len(history) < 3:
        return None
    points = [(float(step), tensor) for step, tensor in history[-3:]]
    return combine(points, float(target_step))


def predict_linear_timestep(
    baseline: RunData,
    slot: int,
    history_steps: list[int],
    target_step: int,
) -> np.ndarray | None:
    if len(history_steps) < 2:
        return None
    points = [
        (baseline.timestep(step, slot), baseline.tensor(step, slot))
        for step in history_steps[-2:]
    ]
    return combine(points, baseline.timestep(target_step, slot))


def predict_quadratic_timestep(
    baseline: RunData,
    slot: int,
    history_steps: list[int],
    target_step: int,
) -> np.ndarray | None:
    if len(history_steps) < 3:
        return None
    points = [
        (baseline.timestep(step, slot), baseline.tensor(step, slot))
        for step in history_steps[-3:]
    ]
    return combine(points, baseline.timestep(target_step, slot))


def evaluate_weighted_predictions(
    baseline: RunData,
    slot: int,
    target_step: int,
    history_steps: list[int],
    candidates: dict[str, dict[int, float]],
) -> list[dict[str, float | str]]:
    used_steps = sorted({step for weights in candidates.values() for step in weights})
    vectors = {
        step: baseline.tensor(step, slot).ravel()
        for step in used_steps
    }
    actual = baseline.tensor(target_step, slot).ravel()
    actual_norm_sq = float(np.vdot(actual, actual).real)
    if actual_norm_sq <= 0.0:
        return []

    dot_history: dict[tuple[int, int], float] = {}
    for i, step_i in enumerate(used_steps):
        tensor_i = vectors[step_i]
        for step_j in used_steps[i:]:
            value = float(np.vdot(tensor_i, vectors[step_j]).real)
            dot_history[(step_i, step_j)] = value
            dot_history[(step_j, step_i)] = value
    dot_actual = {
        step: float(np.vdot(vectors[step], actual).real)
        for step in used_steps
    }

    rows: list[dict[str, float | str]] = []
    actual_norm = actual_norm_sq**0.5
    for formula, weights in candidates.items():
        pred_norm_sq = 0.0
        pred_actual_dot = 0.0
        items = list(weights.items())
        for step_i, weight_i in items:
            pred_actual_dot += weight_i * dot_actual[step_i]
            for step_j, weight_j in items:
                pred_norm_sq += weight_i * weight_j * dot_history[(step_i, step_j)]
        err_sq = max(0.0, pred_norm_sq - 2.0 * pred_actual_dot + actual_norm_sq)
        rows.append(
            {
                "formula": formula,
                "rel_l2": (err_sq**0.5) / actual_norm,
                "rel_l1": np.nan,
            }
        )
    return rows


def evaluate_pair(label: str, baseline: RunData, threshold_run: RunData) -> list[dict]:
    baseline_steps = baseline.steps
    actual_steps = threshold_run.steps
    skipped_steps = [step for step in baseline_steps if step not in set(actual_steps)]
    streak_positions = skip_streak_positions(skipped_steps)
    rows: list[dict] = []

    for skip_order, step in enumerate(skipped_steps):
        interval = interval_for_skip_order(skip_order)
        progress = float(step) / float(max(1, int(baseline.meta.get("steps") or 1) - 1))
        available_steps = [actual_step for actual_step in actual_steps if actual_step < step]
        if not available_steps:
            continue
        for slot in baseline.slots:
            history_steps = available_steps[-max(CHEB_K_VALUES):]
            candidates = candidate_weight_maps(history_steps, step)
            for result in evaluate_weighted_predictions(
                baseline,
                slot,
                step,
                history_steps,
                candidates,
            ):
                rows.append(
                    {
                        "pair": label,
                        "baseline_run": baseline.name,
                        "threshold_run": threshold_run.name,
                        "width": baseline.meta.get("width"),
                        "height": baseline.meta.get("height"),
                        "step": step,
                        "progress": progress,
                        "skip_order": skip_order,
                        "skip_streak_pos": streak_positions[step],
                        "interval": interval,
                        "slot": slot,
                        "formula": result["formula"],
                        "rel_l2": result["rel_l2"],
                        "rel_l1": result["rel_l1"],
                    }
                )
    return rows


def summarize(rows: list[dict], keys: list[str]) -> list[dict]:
    frame = pd.DataFrame(rows)
    group_keys = keys + ["formula"]
    grouped = frame.groupby(group_keys, dropna=False)
    summary = grouped.agg(
        mean_rel_l2=("rel_l2", "mean"),
        median_rel_l2=("rel_l2", "median"),
        mean_rel_l1=("rel_l1", "mean"),
        n=("rel_l2", "count"),
    ).reset_index()
    if keys:
        summary["rank_l2"] = summary.groupby(keys)["mean_rel_l2"].rank(method="min")
    else:
        summary["rank_l2"] = summary["mean_rel_l2"].rank(method="min")
    return summary.sort_values(keys + ["rank_l2", "mean_rel_l2"]).to_dict("records")


def baseline_error_top_steps(rows: list[dict], top_n: int = BASELINE_ERROR_TOP_N) -> list[int]:
    frame = pd.DataFrame(rows)
    previous = frame[frame["formula"] == "previous"]
    summary = (
        previous.groupby("step")
        .agg(mean_rel_l2=("rel_l2", "mean"))
        .reset_index()
        .sort_values("mean_rel_l2", ascending=False)
    )
    return [int(step) for step in summary.head(top_n)["step"].tolist()]


def summarize_conditions(rows: list[dict], top_steps: list[int]) -> list[dict]:
    frame = pd.DataFrame(rows)
    conditions = {
        "all_skip": frame.index == frame.index,
        "progress_ge_0.70": frame["progress"] >= 0.70,
        "skip_streak_pos_eq_1": frame["skip_streak_pos"] == 1,
        "skip_streak_pos_ge_2": frame["skip_streak_pos"] >= 2,
        "baseline_error_top_steps": frame["step"].isin(top_steps),
    }
    summaries: list[dict] = []
    for condition_name, mask in conditions.items():
        condition_rows = frame[mask].to_dict("records")
        for row in summarize(condition_rows, []):
            row["condition"] = condition_name
            row["steps"] = " ".join(str(step) for step in sorted(frame.loc[mask, "step"].unique()))
            summaries.append(row)
    return sorted(summaries, key=lambda row: (row["condition"], row["rank_l2"], row["mean_rel_l2"]))


def summarize_steps(rows: list[dict]) -> list[dict]:
    return summarize(rows, ["step"])


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=Path("analysis/teacache_threshold021_expanded"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    schedule_rows: list[dict] = []

    for label, (baseline_name, threshold_name) in RUN_PAIRS.items():
        baseline = load_run(args.root / baseline_name)
        threshold_run = load_run(args.root / threshold_name)
        baseline_steps = baseline.steps
        actual_steps = threshold_run.steps
        skipped_steps = [step for step in baseline_steps if step not in set(actual_steps)]
        schedule_rows.append(
            {
                "pair": label,
                "baseline_run": baseline.name,
                "threshold_run": threshold_run.name,
                "width": baseline.meta.get("width"),
                "height": baseline.meta.get("height"),
                "full_steps": " ".join(map(str, actual_steps)),
                "skipped_steps": " ".join(map(str, skipped_steps)),
                "skip_count": len(skipped_steps),
            }
        )
        all_rows.extend(evaluate_pair(label, baseline, threshold_run))

    write_csv(args.out / "skip_schedule.csv", schedule_rows)
    write_csv(args.out / "formula_errors.csv", all_rows)
    top_steps = baseline_error_top_steps(all_rows)
    write_csv(args.out / "summary_overall.csv", summarize(all_rows, []))
    write_csv(args.out / "summary_by_pair.csv", summarize(all_rows, ["pair"]))
    write_csv(args.out / "summary_by_interval.csv", summarize(all_rows, ["interval"]))
    write_csv(args.out / "summary_by_pair_interval.csv", summarize(all_rows, ["pair", "interval"]))
    write_csv(args.out / "summary_by_step.csv", summarize_steps(all_rows))
    write_csv(args.out / "summary_by_condition.csv", summarize_conditions(all_rows, top_steps))
    write_csv(
        args.out / "condition_membership.csv",
        [
            {
                "condition": "baseline_error_top_steps",
                "steps": " ".join(str(step) for step in top_steps),
                "top_n": len(top_steps),
                "basis": "highest mean rel_l2 for previous",
            }
        ],
    )

    print(f"wrote {args.out}")
    print(pd.DataFrame(schedule_rows).to_string(index=False))
    print()
    print(pd.DataFrame(summarize(all_rows, [])).head(20).to_string(index=False))
    print()
    print(pd.DataFrame(summarize_conditions(all_rows, top_steps)).groupby("condition").head(10).to_string(index=False))


if __name__ == "__main__":
    main()
