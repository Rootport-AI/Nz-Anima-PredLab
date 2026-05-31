# TeaCache Residual Analysis: Threshold 0.21, runC-G

This document records the runC-G analysis only. It is intentionally separate from
the earlier runA-B notes so that the threshold 0.21 / skips=12 experiment remains
easy to read on its own.

## Purpose

Anima base v1.0 is substantially slower than SDXL. A practical acceleration
target requires skipping roughly one third to one half of the DiT block-stack
work. In user testing, TeaCache threshold `0.21` skipped 12 of 32 steps while
preserving the overall art style well enough for practical use.

This analysis asks:

- Whether skipped step positions change with prompt, seed, or resolution.
- Whether a residual replacement formula that beats TeaCache's existing
  `previous_residual` changes with prompt, seed, resolution, or threshold.
- Whether the best formula changes by skip interval or skip streak position.

## Dataset

All runs were collected on 2026-05-30 from:

```text
T:\StabilityMatrix\Images\logs\2026-05-30
```

The analysis pairs are:

| Pair | Baseline run | Threshold run | Purpose |
| --- | --- | --- | --- |
| C_promptA | `runC1_20260530_193131_gen0001` | `runC2_20260530_193212_gen0002` | prompt A |
| D_promptB | `runD1_20260530_193822_gen0004` | `runD2_20260530_193908_gen0005` | prompt B |
| E_promptC | `runE1_20260530_194052_gen0006` | `runE2_20260530_194225_gen0007` | prompt C |
| F_seed | `runF1_20260530_194533_gen0008` | `runF2_20260530_194919_gen0010` | seed change, prompt A |
| G_resolution | `runG1_20260530_195128_gen0011` | `runG2_20260530_195319_gen0012` | 1536 resolution, prompt A |

Baseline runs use TeaCache threshold `0.0`, which produced residual records for
all 32 steps. Threshold runs use TeaCache threshold `0.21`, producing residual
records only for full-calculation steps. Missing residual records in the
threshold run are treated as skipped steps.

## Analysis Artifacts

Script:

```text
tools/analyze_teacache_residuals.py
```

Primary output directory:

```text
analysis/teacache_threshold021_expanded/
```

Important CSV files:

| File | Contents |
| --- | --- |
| `skip_schedule.csv` | full and skipped step lists per pair |
| `formula_errors.csv` | per pair / step / slot / formula error rows |
| `summary_overall.csv` | formula ranking over all skipped steps |
| `summary_by_interval.csv` | formula ranking by skip interval |
| `summary_by_condition.csv` | formula ranking by requested application condition |
| `summary_by_step.csv` | best formula candidates per skipped step |
| `condition_membership.csv` | membership for derived conditions such as baseline-error top steps |

The expanded analysis evaluated 193 formula variants and 23,160 formula rows.
For this expanded run, the exact metric is relative L2 error computed from
history tensor inner products. Relative L1 is intentionally not populated in the
expanded CSV because materializing every candidate tensor would be much slower.

## Skip Schedule

The skipped steps were identical across prompt, seed, and resolution:

```text
skipped steps = 10, 12, 14, 16, 18, 20, 21, 23, 24, 26, 27, 28
skip count    = 12 / 32
```

The full-calculation steps were also identical:

```text
full steps = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 15, 17, 19, 22, 25, 29, 30, 31
```

Interpretation:

- In this dataset, skip positions appear to be determined mainly by the timestep
  schedule and TeaCache threshold.
- Prompt, seed, and resolution did not change the skip schedule at threshold
  `0.21`.
- This does not prove the schedule is universally invariant, but it is stable
  across the collected runC-G conditions.

## Formula Families

The expanded search includes:

- `previous`: existing TeaCache behavior, reusing the most recent residual.
- `linear_step_aX`: linear extrapolation in step index, damped from previous by
  alpha `X`.
- `taylor2_step_aX`: quadratic extrapolation in step index, damped from previous
  by alpha `X`.
- `taylor2_curve_bX`: linear extrapolation plus only a damped curvature term.
- `cheb_step_kK_dD_rR_aX`: Chebyshev regression in step index using history
  window `K`, degree `D`, ridge `R`, and alpha damping `X`.

Search ranges:

```text
linear alpha = 0.25, 0.5, 0.75, 1.0
taylor2 alpha = 0.25, 0.5, 0.75, 1.0
taylor2 curvature beta = 0.25, 0.5, 0.75, 1.0
cheb k = 3, 4, 5, 6, 8
cheb degree = 1, 2, 3
cheb ridge = 0, 1e-4, 1e-2
cheb alpha = 0.25, 0.5, 0.75, 1.0
```

## Overall Ranking

Across all skipped steps, all pairs, and both slots:

| Rank | Formula | Mean rel L2 |
| --- | --- | --- |
| 1 | `linear_step_a0.75` | 0.075315 |
| 2 | `linear_step_a0.5` | 0.075842 |
| 3 | `linear_step_a1` | 0.076447 |
| 4 | `cheb_step_k4_d2_r0.01_a0.25` | 0.077699 |
| 5 | `cheb_step_k4_d2_r0.01_a0.5` | 0.077711 |

Interpretation:

- Damped linear extrapolation is the best overall family.
- The best all-skip alpha is `0.75`, not the undamped `1.0`.
- Chebyshev comes close but does not beat damped linear in the overall ranking.

## Condition Rankings

### All Skips

Best formula:

```text
linear_step_a0.75
mean rel L2 = 0.075315
```

### Progress >= 0.70

Condition membership:

```text
steps = 23, 24, 26, 27, 28
```

Top formulas:

| Rank | Formula | Mean rel L2 |
| --- | --- | --- |
| 1 | `taylor2_curve_b0.25` | 0.097202 |
| 2 | `linear_step_a1` | 0.098260 |
| 3 | `taylor2_curve_b0.5` | 0.098568 |
| 4 | `cheb_step_k4_d2_r0.01_a1` | 0.101826 |
| 5 | `cheb_step_k4_d2_r0.01_a0.75` | 0.101976 |

Interpretation:

- Late steps benefit from curvature, but only when the curvature term is damped.
- Fully damped or undamped quadratic extrapolation is not preferred. The useful
  signal is a small curvature correction on top of linear extrapolation.

### skip_streak_pos == 1

Condition membership:

```text
steps = 10, 12, 14, 16, 18, 20, 23, 26
```

Top formulas:

| Rank | Formula | Mean rel L2 |
| --- | --- | --- |
| 1 | `linear_step_a0.25` | 0.047767 |
| 2 | `cheb_step_k3_d1_r0.01_a0.25` | 0.048209 |
| 3 | `previous` | 0.048266 |
| 4 | `cheb_step_k3_d1_r0.0001_a0.25` | 0.048397 |
| 5 | `cheb_step_k3_d1_r0_a0.25` | 0.048399 |

Interpretation:

- First skips in a streak are already close to the previous residual.
- Very lightly damped linear extrapolation wins, but the margin over `previous`
  is small.
- This region is not the strongest place to spend implementation complexity.

### skip_streak_pos >= 2

Condition membership:

```text
steps = 21, 24, 27, 28
```

Top formulas:

| Rank | Formula | Mean rel L2 |
| --- | --- | --- |
| 1 | `linear_step_a1` | 0.118641 |
| 2 | `taylor2_curve_b0.25` | 0.119741 |
| 3 | `linear_step_a0.75` | 0.122910 |
| 4 | `taylor2_curve_b0.5` | 0.124144 |
| 5 | `cheb_step_k4_d2_r0.01_a0.75` | 0.124598 |

Interpretation:

- Consecutive skips should not simply reuse `previous_residual`.
- Undamped linear extrapolation is strongest here, with small-curvature Taylor
  close behind.

### Baseline Error Top Steps

Defined as the four skipped steps with the highest mean `previous` error:

```text
steps = 28, 27, 26, 24
```

Top formulas:

| Rank | Formula | Mean rel L2 |
| --- | --- | --- |
| 1 | `taylor2_curve_b0.25` | 0.111444 |
| 2 | `taylor2_curve_b0.5` | 0.112415 |
| 3 | `linear_step_a1` | 0.113261 |
| 4 | `cheb_step_k4_d2_r0.01_a1` | 0.115950 |
| 5 | `taylor2_curve_b0.75` | 0.116021 |

Interpretation:

- The hardest steps are late steps.
- A damped curvature term is the best candidate for repairing the worst
  previous-residual failures.

## Interval Rankings

The skip-order intervals are:

```text
skip_first4 = skipped order 0-3
skip_mid4   = skipped order 4-7
skip_last4  = skipped order 8-11
```

Best formulas:

| Interval | Best formula | Mean rel L2 |
| --- | --- | --- |
| `skip_first4` | `previous` | 0.050827 |
| `skip_mid4` | `linear_step_a0.25` | 0.040899 |
| `skip_last4` | `taylor2_curve_b0.25` | 0.111444 |

Interpretation:

- Early skipped steps are best left as the existing TeaCache previous residual.
- Mid skipped steps can benefit from very light linear extrapolation, but the
  difference from `previous` is small.
- Late skipped steps clearly want extrapolation, especially a damped curvature
  correction.

## Step-Level Best Formulas

| Step | Best formula | Mean rel L2 |
| --- | --- | --- |
| 10 | `previous` | 0.057741 |
| 12 | `previous` | 0.055639 |
| 14 | `previous` | 0.047832 |
| 16 | `previous` | 0.042095 |
| 18 | `previous` | 0.038251 |
| 20 | `linear_step_a0.25` | 0.035548 |
| 21 | `cheb_step_k3_d1_r0.01_a0.25` | 0.052487 |
| 23 | `linear_step_a0.5` | 0.036530 |
| 24 | `linear_step_a0.5` | 0.061868 |
| 26 | `taylor2_curve_b0.5` | 0.047770 |
| 27 | `taylor2_curve_b0.5` | 0.108295 |
| 28 | `taylor2_curve_b0.5` | 0.212031 |

Step 28 remains difficult even after the expanded search. It is a candidate for
either a special late-step formula or a forced full calculation if image quality
needs more safety.

## Practical Implementation Hypotheses

The simplest useful policy is:

```text
steps <= 18:
  use previous_residual

steps 20-21:
  use linear_step alpha=0.25
  or keep previous_residual if simplicity is preferred

steps 23-24:
  use linear_step alpha=0.5

steps >= 26:
  use taylor2 curve-only beta=0.25 or 0.5
```

A condition-based policy may be easier to generalize than hard-coded step
numbers:

```text
if progress >= 0.70:
  use taylor2_curve beta=0.25
elif skip_streak_pos >= 2:
  use linear_step alpha=1.0
elif skip_streak_pos == 1:
  use linear_step alpha=0.25 or previous
else:
  use previous
```

However, the condition-based policy has an overlap issue: late consecutive skips
belong to both `progress >= 0.70` and `skip_streak_pos >= 2`. The current
analysis suggests late-progress rules should take priority, because the largest
previous-residual failures are late.

## Current Interpretation

The runC-G results support the earlier runA-B intuition:

- The best residual replacement is not uniform across the denoising trajectory.
- Early and mid skips are relatively safe with the existing previous residual.
- The late region is where previous residual reuse breaks down and extrapolation
  matters most.
- Chebyshev regression is not the leading candidate in this dataset. It is close
  in some conditions, but damped linear or damped Taylor curvature is simpler and
  generally better.

The most promising next implementation target is therefore not a global formula,
but a piecewise TeaCache residual forecaster:

```text
early: previous
middle: previous or lightly damped linear
late: damped curvature Taylor, with optional force-full guard around the final hard step
```

