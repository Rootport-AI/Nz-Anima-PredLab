# UjiCache EMA Prediction 追加仕様案

## 1. 目的

UjiCacheに、residual履歴から代替residualを予測するEMA補正を追加する。

最終目的は、DiT / Anima block stackをskipしたステップにおいて、

```text
y_pred = x_now + r_pred
```

の `r_pred` を、通常TeaCache residual onlyより高精度に予測することである。

UIの簡潔さよりも、既存式との比較実験の信頼性を優先する。

---

## 2. 既存仕様の前提

UjiCacheでは、full計算時に以下を得る。

```text
x_before = Anima block stack 入力中間テンソル
x_after  = Anima block stack 出力中間テンソル

residual = x_after - x_before
```

skip時には、現在のblock stack入力 `x_now` に対して、

```text
x_now = x_now + residual
```

を行う。

既存formulaは以下の3つ。

```text
TeaCache (residual only)
Linear extrapolation
Taylor2 curve
```

既存formulaの挙動は、EMA追加後も維持する。

---

## 3. UI追加項目

UjiCacheパネルに以下の2つのスライダーを追加する。

### 3.1 Slope EMA Smoothing

```text
Label: Slope EMA Smoothing
Minimum: 0.00
Maximum: 0.99
Step: 0.01
Default: 0.00
```

意味:

```text
residualの傾き velocity をどれくらいEMA平滑化するか
```

内部変数名案:

```text
ujicache_slope_ema_smoothing
```

---

### 3.2 Curve EMA Smoothing

```text
Label: Curve EMA Smoothing
Minimum: 0.00
Maximum: 0.99
Step: 0.01
Default: 0.00
```

意味:

```text
Taylor2の曲率・加速度 acceleration をどれくらいEMA平滑化するか
```

内部変数名案:

```text
ujicache_curve_ema_smoothing
```

---

## 4. UI有効・無効ルール

UI項目は無効化しても値をリセットしない。
グレーアウトは「現在の条件では計算に使われない」ことだけを示す。

### 4.1 Formula = TeaCache (residual only)

以下をグレーアウトする。

```text
Use prediction after progress
Apply prediction from skip #
Prediction strength
Taylor2 curve strength
Slope EMA Smoothing
Curve EMA Smoothing
```

理由:

```text
TeaCache residual onlyではprediction式を使わず、previous_residualをそのまま使うため。
```

以下は有効のままにする。

```text
Rel L1 threshold
Start progress
End progress
Cache device
Modulated source
Coefficient profile
Max skip streak
Force full interval
Dry run
Verbose UjiCache trace
```

これらはskip判定、キャッシュ配置、強制full計算、ログ出力に関係するため。

---

### 4.2 Formula = Linear extrapolation

以下を有効にする。

```text
Use prediction after progress
Apply prediction from skip #
Prediction strength
Slope EMA Smoothing
```

以下をグレーアウトする。

```text
Taylor2 curve strength
Curve EMA Smoothing
```

理由:

```text
Linearでは曲率・加速度成分を使わないため。
```

---

### 4.3 Formula = Taylor2 curve

以下を有効にする。

```text
Use prediction after progress
Apply prediction from skip #
Prediction strength
Taylor2 curve strength
Slope EMA Smoothing
```

`Curve EMA Smoothing` は条件付きで有効化する。

```text
Formula = Taylor2 curve
かつ
Slope EMA Smoothing > 0
```

のときだけ、`Curve EMA Smoothing` を操作可能にする。

`Slope EMA Smoothing = 0` のときは、`Curve EMA Smoothing` はグレーアウトし、内部計算でも無視する。

---

## 5. 計算式の分岐方針

重要方針:

```text
Slope EMA Smoothing = 0 の場合は、既存Linear / 既存Taylor2の計算経路をそのまま使う。
```

EMA対応のために既存式を共通化・書き換えない。
比較実験の基準を壊さないためである。

裏側では以下の4経路を持つ。

```text
Linear extrapolation + Slope EMA Smoothing = 0
  → 既存Linear

Linear extrapolation + Slope EMA Smoothing > 0
  → Linear-EMA-damping

Taylor2 curve + Slope EMA Smoothing = 0
  → 既存Taylor2

Taylor2 curve + Slope EMA Smoothing > 0
  → Taylor2-EMA-damping
```

疑似コード:

```python
if formula == UJICACHE_FORMULA_TEACACHE:
    return previous_residual

if formula == UJICACHE_FORMULA_LINEAR:
    if slope_ema_smoothing <= 0.0:
        return _ujicache_predict_linear(slot, step_index, previous)
    else:
        return _ujicache_predict_linear_ema(slot, step_index, previous)

if formula == UJICACHE_FORMULA_TAYLOR2:
    if slope_ema_smoothing <= 0.0:
        return _ujicache_predict_taylor2(slot, step_index, previous)
    else:
        return _ujicache_predict_taylor2_ema(slot, step_index, previous)
```

---

## 6. 記号定義

full計算で得られたtrue residual履歴を以下のように定義する。

```text
t_i = full計算されたlogical step index
r_i = そのstepで得られたtrue residual
```

直近のtrue residualを、

```text
t_n
r_n
```

とする。

skip対象の現在stepを、

```text
t_target
```

とする。

予測距離:

```text
dt_pred = t_target - t_n
```

UjiCacheではskipが入るため、full計算されたstepは等間隔とは限らない。
したがって、傾きは単純差分ではなくstep差で割る。

---

## 7. EMA状態

UjiCacheのslotごとに、以下の状態を追加する。

```text
previous_velocity
previous_velocity_time
velocity_ema
acceleration_ema
```

既存のslotには以下がある前提。

```text
previous_residual
residual_history
```

追加状態はcond/uncond slotごとに独立して持つ。

---

## 8. EMA更新タイミング

EMAは、full計算でtrue residualが得られたときだけ更新する。

skip時に予測したresidualをEMA更新に使ってはならない。

理由:

```text
予測residualを履歴やEMA更新に混ぜると、予測誤差を自己学習してしまうため。
```

EMA状態は、`Slope EMA Smoothing = 0` の場合でも常に更新する。
ただし、EMA状態をpredictionに使うのは `Slope EMA Smoothing > 0` の場合だけである。
`Slope EMA Smoothing = 0` の場合は既存Linear / 既存Taylor2関数へ直接分岐するため、EMA状態の更新は出力に影響しない。

---

## 9. full計算時のEMA更新式

新しいtrue residualが得られたとき、前回true residualが存在すればvelocityを計算する。

```text
dt = t_n - t_prev
v_obs = (r_n - r_prev) / dt
v_time = (t_prev + t_n) / 2
```

ここで、

```text
t_prev = 1つ前のtrue residualのstep index
r_prev = 1つ前のtrue residual
```

`dt <= 0` の場合は、EMA更新を行わない。

---

### 9.1 velocity EMA

```text
beta_v = Slope EMA Smoothing
```

初回velocityの場合:

```text
velocity_ema = v_obs
```

2回目以降:

```text
velocity_ema = beta_v * velocity_ema + (1 - beta_v) * v_obs
```

---

### 9.2 acceleration EMA

前回velocityが存在する場合、観測velocity同士の差分から加速度を計算する。

velocityの時刻は、対応するresidual区間の中点として扱う。
現在の観測velocity `v_obs` の時刻は以下である。

```text
v_time = (t_prev + t_n) / 2
```

前回観測velocityの時刻は `previous_velocity_time` として保持する。

```text
dt_v = v_time - previous_velocity_time
a_obs = (v_obs - previous_velocity) / dt_v
```

`dt_v <= 0` の場合は、acceleration EMAを更新しない。

```text
beta_a = Curve EMA Smoothing
```

初回accelerationの場合:

```text
acceleration_ema = a_obs
```

2回目以降:

```text
acceleration_ema = beta_a * acceleration_ema + (1 - beta_a) * a_obs
```

最後に、

```text
previous_velocity = v_obs
previous_velocity_time = v_time
```

を更新する。

---

## 10. Linear-EMA-damping の計算式

条件:

```text
Formula = Linear extrapolation
Slope EMA Smoothing > 0
velocity_ema が存在する
```

raw prediction:

```text
raw_prediction = r_n + dt_pred * velocity_ema
```

damping後のprediction:

```text
r_pred = r_n + Prediction strength * (raw_prediction - r_n)
```

同値な式:

```text
r_pred = r_n + Prediction strength * dt_pred * velocity_ema
```

`velocity_ema` が存在しない場合はfallbackする。

推奨fallback:

```text
previous_residualをそのまま返す
fallback reason = insufficient_ema_velocity
```

---

## 11. Taylor2-EMA-damping の計算式

条件:

```text
Formula = Taylor2 curve
Slope EMA Smoothing > 0
velocity_ema が存在する
```

まずLinear-EMA成分を計算する。

```text
linear_ema_prediction = r_n + dt_pred * velocity_ema
```

Taylor2の曲率項は、`acceleration_ema` が存在する場合のみ使う。

```text
curve_term = 0.5 * (dt_pred ** 2) * acceleration_ema
```

Taylor2 curve strengthをかける。

```text
raw_prediction = linear_ema_prediction + Taylor2 curve strength * curve_term
```

最後にPrediction strengthでdampingする。

```text
r_pred = r_n + Prediction strength * (raw_prediction - r_n)
```

展開形:

```text
r_pred = r_n
       + Prediction strength * (
             dt_pred * velocity_ema
           + Taylor2 curve strength * 0.5 * (dt_pred ** 2) * acceleration_ema
         )
```

`acceleration_ema` が存在しない場合は、曲率項を0として扱う。

```text
raw_prediction = linear_ema_prediction
```

この場合、ログ上では以下のように記録してよい。

```text
decision = prediction
prediction_note = taylor2_ema_without_acceleration
```

これはfallbackではなく、velocity_emaのみを用いたLinear-EMA相当のpredictionとして扱う。
そのため `fallback reason` には入れない。

ただし `velocity_ema` も存在しない場合はfallbackする。

```text
fallback reason = insufficient_ema_velocity
```

---

## 12. Curve EMA Smoothing = 0 の扱い

`Formula = Taylor2 curve` かつ `Slope EMA Smoothing > 0` の場合、`Curve EMA Smoothing = 0` は以下を意味する。

```text
加速度EMAを平滑化しない。
a_ema = a_obs と同等。
```

ただし、UI上では `Curve EMA Smoothing` は有効なままにする。
これは、Taylor2-EMAのうち「傾きはEMA、曲率は非EMA」という条件を検証可能にするためである。

一方、`Slope EMA Smoothing = 0` の場合は、EMA全体を無効化し、既存Taylor2経路を使う。
このとき `Curve EMA Smoothing` はグレーアウトし、内部計算でも無視する。

---

## 13. prediction開始条件との関係

既存UjiCacheには以下の条件がある。

```text
Use prediction after progress
Apply prediction from skip #
```

これらは、Linear-EMA / Taylor2-EMAでも既存predictionと同じように適用する。

つまり、EMA版predictionを使う条件は、

```text
既存のprediction_allowed条件を満たす
かつ
FormulaがLinearまたはTaylor2
かつ
Slope EMA Smoothing > 0
かつ
必要なEMA状態が存在する
```

である。

prediction_allowed条件を満たさない場合は、従来通りprevious_residualを使う。

---

## 14. validation / guard

EMA版predictionも、既存のprediction validationを必ず通す。

必要なチェック:

```text
shape一致
finite値チェック
dtype変換
norm guard
```

predictionのshapeがprevious_residualまたはtarget_sliceと一致しない場合はfallbackする。

NaN / Infがある場合はfallbackする。

prediction normが異常に大きい場合はfallbackする。

既存の `_ujicache_validate_prediction()` を再利用すること。

---

## 15. ログ

Verbose UjiCache traceでは、EMA版を使った場合に以下を出せるようにする。

```text
formula
slope_ema_smoothing
curve_ema_smoothing
ema_velocity_ready
ema_acceleration_ready
dt_pred
decision
prediction_note
fallback reason
```

decision候補:

```text
fallback
prediction
```

fallback reason候補:

```text
formula
streak
insufficient_history
insufficient_ema_velocity
shape_mismatch
numeric_error
norm_guard
prediction_error
```

Taylor2-EMAでacceleration_emaが未初期化だがLinear-EMA成分だけ使った場合は、fallbackではなくprediction扱いでよい。
この場合は `fallback reason` ではなく、以下の補足情報として記録する。

```text
prediction_note=taylor2_ema_without_acceleration
```

追加情報:

```text
ema_curve_ready=False
curve_term_used=False
```

---

## 16. infotext metadata

UjiCacheが有効な場合、生成メタデータに以下を追加する。

```text
UjiCache slope_ema_smoothing
UjiCache curve_ema_smoothing
```

値は小数2桁程度でよい。

例:

```text
UjiCache slope_ema_smoothing: 0.70
UjiCache curve_ema_smoothing: 0.90
```

---

## 17. backward compatibility

以下を守る。

```text
Slope EMA Smoothing default = 0.00
Curve EMA Smoothing default = 0.00
```

既存のUI引数数に対応するため、古いscript_argsから読み込む場合は、EMA関連引数をデフォルト値0.00で補う。

`STATE.apply_options()` に以下を追加する。

```text
ujicache_slope_ema_smoothing: float = 0.0
ujicache_curve_ema_smoothing: float = 0.0
```

範囲は以下でclampする。

```text
0.0 <= value <= 0.99
```

---

## 18. 実装対象ファイル候補

主な変更候補:

```text
nz_anima_predlab/state.py
nz_anima_predlab/script.py
nz_anima_predlab/patcher.py
```

変更内容:

```text
state.py:
  - EMA smoothing用定数またはstate fieldを追加
  - RuntimeState.apply_optionsに引数追加
  - clamp処理追加

script.py:
  - UjiCache Accordionに2スライダー追加
  - Formula / Slope EMA値に応じたグレーアウトUI更新処理追加
  - return script_argsリストに追加
  - backward compatibility用のdefault挿入処理更新
  - infotext metadata追加

patcher.py:
  - slot初期化にEMA状態を追加
  - full計算時のresidual記録時にEMA更新
  - Linear EMA prediction関数追加
  - Taylor2 EMA prediction関数追加
  - formula + smoothing値による分岐追加
  - validation / logging更新
```

---

## 19. 推奨初期実験値

Linear-EMA-damping:

```text
Formula: Linear extrapolation
Slope EMA Smoothing: 0.30 / 0.50 / 0.70 / 0.85
Prediction strength: 既存探索値を使用
Curve EMA Smoothing: disabled
```

Taylor2-EMA-damping:

```text
Formula: Taylor2 curve
Slope EMA Smoothing: 0.30 / 0.50 / 0.70
Curve EMA Smoothing: 0.50 / 0.70 / 0.85 / 0.95
Taylor2 curve strength: 0.05 / 0.10 / 0.20 / 0.25
Prediction strength: 既存探索値を使用
```

基本仮説:

```text
Slope EMA Smoothingは中程度
Curve EMA Smoothingは高め
Taylor2 curve strengthは小さめ
```

ただし、最終判断はLPIPS / SSIM / 視覚比較の実験結果で行う。

---

## 20. 実験上の注意

EMA版は既存式の上位互換とは限らない。

特に以下を比較対象として必ず残す。

```text
TeaCache residual only
既存Linear
既存Taylor2
Linear-EMA
Taylor2-EMA
```

`Slope EMA Smoothing = 0` のときに既存式と同じ結果になるかを厳密検証する必要はない。
実装上は、`Slope EMA Smoothing = 0` なら既存関数へ直接分岐するためである。

EMA版が悪化する可能性もある。
その場合でも、どの成分が悪化要因かを切り分けられるように、ログとUI条件を明確にする。
