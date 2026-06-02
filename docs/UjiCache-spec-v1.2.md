# UjiCache 仕様書 v1.2

## 1. 概要

`UjiCache` は、TeaCache系のキャッシュ高速化において、skipされたdenoising stepで使用する `previous residual` を、複数の予測式に差し替えて比較実験するための機能である。

名称の由来は、TeaCacheの `Tea` から連想した日本茶・宇治茶の `Uji`。
技術的には以下の略称とする。

```text
UJI = Unified Jump-step Imputation
```

本機能は、TeaCacheの「どのstepをskipするか」という判定ロジックは変更しない。
変更するのは、TeaCacheがskip stepで使用する residual の計算方法のみである。

---

## 2. 目的

UjiCacheの目的は以下である。

```text
1. TeaCacheのskip判定をそのまま利用する。
2. skip stepで使うprevious residualを、別の予測式に置き換える。
3. 品質劣化を減らす。
4. 同じ品質を保ったまま、TeaCacheのthresholdをより攻められる可能性を探る。
5. Anima / Cosmos系モデルでの高速推論実験に使う。
```

UjiCacheは、低ステップLoRAではない。
モデル重みを変更しない。
schedulerも変更しない。
step数も変更しない。

---

## 3. UI仕様

### UIの構成  
　土台となるTeaCacheのUIを活かす。  

UjiCache  
 [ ]Enable UjiCache （チェックボックス）  
  Preset mode （ドロップダウンリスト）  
  Rel L1 Threshold（TeaCacheのものを流用）  
  Start progress（TeaCacheのものを流用）  
  End progress（TeaCacheのものを流用）  
 
  Prediction formula:   
   - TeaCache (residual only)  
   - Linear extrapolation   
   - Taylor2 curve   
   
  Use prediction after progress:   
    slider 0.00 - 1.00  
    default: 0.00  
  
  Apply prediction from skip #:  
   slider 1 - 3   
   default: 2   
   
  Prediction strength:   
   slider 0.00 - 1.00   
   default: 0.50 
    
  Taylor2 curve strength:   
   slider 0.00 - 1.00   
   default: 0.25   
   
  Fallback: previous residual  
 
  Cache device: cuda / cpu （TeaCacheのものを流用）  
  Modulated source: （TeaCacheのものを流用）  
  Coefficient profile: （TeaCacheのものを流用）  
  Max skip streak (0=off) （TeaCacheのものを流用）  
  Force full interval(0=off) （TeaCacheのものを流用）  
  [ ]Dry-run （TeaCacheのものを流用）  
  [ ]Verbose UjiCache Trace （TeaCacheのものを流用）  

### 3.1 UjiCache

```text
UjiCache
```

まずはNz-Anima-PredLabのサブアコーディオンの1つとして実装する。
将来的には、Top-level accordionとして独立させることも視野に入れる。
現状では、TeaCacheのサブアコーディオンを丸ごとコピーして名称だけ変更したものが用意されている。これを土台として機能を追加していく。

---

### 3.2 Enable UjiCache

```text
Enable UjiCache:
  checkbox
```

#### 挙動

* OFFの場合、UjiCacheは何もしない。
* ONの場合、TeaCacheがskip stepで使用するresidualをUjiCache側で差し替える。UjiCacheのサブアコーディオンの裏側には、すでにTeaCacheのskip判定機構が丸ごとコピー済みである。これを使う。
* UjiCacheはTeaCache判定ロジックを内包するため、通常TeaCache patchとは相互排他にする。親のNz-Anima-PredLabの仕様に従い、同時起動が危険な機能が同時にEnableにならないようにする。

---

### 3.3 Preset mode

```text
Preset mode:
  - Custom
```

初期実装では `Custom` のみ。

UjiCacheのpresetはTeaCacheの `Safe` / `Balanced` / `Aggressive` とは独立させる。
現状のUI scaffoldはTeaCache UIを複製したものだが、UjiCache実装時にはpreset候補をUjiCache専用に置き換え、初期状態では `Custom` のみ選択可能にする。

将来的に以下のようなpresetを追加できる設計にしておく。

```text
- ER SDE-Beta Shift=3 32steps 1024*1024 Threshold=0.21
- ER SDE-Beta Shift=3 32steps 1536*1536 Threshold=0.21
- ER SDE-Beta Shift=3 32steps 1024*1024 Threshold=0.25
- ER SDE-Beta Shift=3 32steps 1536*1536 Threshold=0.25
- Euler a-Simple Shift=3 32steps 1024*1024 Threshold=0.21
- Euler a-Simple Shift=3 32steps 1024*1024 Threshold=0.25
```

---

### 3.4 Prediction formula

```text
Prediction formula:
  - TeaCache (residual only)
  - Linear extrapolation
  - Taylor2 curve
```

#### 3.4.1 TeaCache (residual only)

TeaCache互換モード。

skip stepでは常に直近のactual forwardで得られたprevious residualを使用する。

```text
r_pred = r_last
```

このモードでは以下の設定は実質的に使用しない。

```text
Use prediction after progress
Apply prediction from skip #
Prediction strength
Taylor2 curve strength
```

ただし、比較実験用のbaselineとしてUIには残す。

---

#### 3.4.2 Linear extrapolation

直近2つのactual residualから、現在のskip stepにおけるresidualを線形外挿する。

必要な履歴:

```text
r_prevprev
r_prev
```

ここで、

```text
r_prevprev = 2回前のactual forwardで得たresidual
r_prev     = 直近のactual forwardで得たresidual
```

時間情報:

```text
t_prevprev = r_prevprevを得たstep index
t_prev     = r_prevを得たstep index
t_now      = 現在のskip step index
```

正規化された外挿距離:

```text
dt = max(t_prev - t_prevprev, epsilon)
k  = (t_now - t_prev) / dt
```

予測式:

```text
delta = r_prev - r_prevprev

raw_prediction = r_prev + k * delta

r_pred = r_prev + prediction_strength * (raw_prediction - r_prev)
```

`prediction_strength` は 0.0〜1.0。

```text
0.0 = TeaCacheと同じ。外挿しない。
1.0 = 完全な線形外挿。
0.5 = 半分だけ外挿する。
```

履歴が2点未満の場合はfallbackする。

---

#### 3.4.3 Taylor2 curve

直近3つのactual residualから、2次曲線的にresidualを外挿する。

必要な履歴:

```text
r0
r1
r2
```

ここで、

```text
r0 = 3回前のactual residual
r1 = 2回前のactual residual
r2 = 直近のactual residual
```

時間情報:

```text
t0 = r0を得たstep index
t1 = r1を得たstep index
t2 = r2を得たstep index
t_now = 現在のskip step index
```

実験結果の再現性を優先し、実装では `analysis` で使った計算と同じく、actual residualを得たstep indexに基づく外挿重みを使う。
等間隔stepの場合は、以下の差分形式と同じ意味になる。

正規化された外挿距離:

```text
dt = max(t2 - t1, epsilon)
k  = (t_now - t2) / dt
```

1次差分:

```text
d1 = r2 - r1
```

2次差分:

```text
d2 = r2 - 2 * r1 + r0
```

予測式:

```text
linear_term = k * d1

quadratic_term = 0.5 * k * (k + 1) * d2

linear_prediction = r2 + linear_term

quadratic_prediction = r2 + linear_term + quadratic_term

raw_prediction =
    (1 - taylor2_curve_strength) * linear_prediction
    + taylor2_curve_strength * quadratic_prediction

r_pred = r2 + prediction_strength * (raw_prediction - r2)
```

実際のstep間隔が等間隔でない場合は、`linear_prediction` は `(t1, r1), (t2, r2)` からの線形Lagrange外挿、`quadratic_prediction` は `(t0, r0), (t1, r1), (t2, r2)` からの2次Lagrange外挿として計算する。
実験結果と矛盾する場合は、実験結果で使ったLagrange外挿を正とする。

`Taylor2 curve strength` は2次項の効き具合を調整する。

```text
0.0 = 2次項を使わない。実質damped linear。
1.0 = 2次項を完全に使う。
0.25 = 2次項を弱めに使う。
```

履歴が3点未満の場合はfallbackする。

---

### 3.5 Use prediction after progress

```text
Use prediction after progress:
  slider 0.00 - 1.00
  default: 0.00
```

#### 意味

samplingの進行度がこの値を超えた後は、連続skip条件に関係なく、skip 1回目からprediction formulaを使用する。

#### progress定義

内部では、既存TeaCache判定で使っているprogress値と同じものを使用する。
UjiCache側で独自にprogressを再定義しない。

```text
progress = 土台となるTeaCache側のprogress値
```

現行TeaCache実装では以下を使う。

```text
progress = step_index / (total_steps - 1)
```

これにより、TeaCacheのskip判定とUjiCacheのlate phase判定で、進行度の解釈がズレないようにする。

#### 判定

```text
late_phase = progress > use_prediction_after_progress
```

`>` を使用する。
これにより、30 stepsで `0.70` の場合、現行TeaCache progress定義ではstep 20まではlate phaseに入らず、step 21以降でlate phaseに入る。

#### 例

```text
total_steps = 30
use_prediction_after_progress = 0.70

step 20:
  progress = 20 / 29 = 0.689...
  late_phase = False

step 21:
  progress = 21 / 29 = 0.724...
  late_phase = True
```

---

### 3.6 Apply prediction from skip

```text
Apply prediction from skip #:
  slider 1 - 3
  default: 2
```

#### 意味

late phaseより前の区間において、連続skipの何回目からprediction formulaを使うかを指定する。

```text
1 = skip 1回目からpredictionを使う
2 = skip 2回目からpredictionを使う
3 = skip 3回目からpredictionを使う
```

#### 重要

この設定は、`Use prediction after progress` より前の区間でのみ効く。

late phaseでは、この値に関係なく、skip 1回目からprediction formulaを使う。

---

### 3.7 Prediction strength

```text
Prediction strength:
  slider 0.00 - 1.00
  default: 0.50
```

#### 使用対象

以下のprediction formulaで使用する。

```text
Linear extrapolation
Taylor2 curve
```

#### 意味

予測式の出力をどれくらい強く反映するかを指定する。実験時にはdampingと呼ばれていた計算を再現するための機能。

```text
0.0:
  現行TeaCacheと同じprevious residualを使う。

0.5:
  予測式の出力へ半分だけ寄せる。初期推奨値。

1.0:
  予測式の出力をそのまま使う。
```

実験結果で評価した `linear_step_a*` は、この値を `a` として再現する。
実験結果で評価した `taylor2_step_curve_b*` は、`Prediction strength = 1.0` とし、`Taylor2 curve strength = b` にすることで再現する。

---

### 3.8 Taylor2 curve strength

```text
Taylor2 curve strength:
  slider 0.00 - 1.00
  default: 0.25
```

#### 使用対象

以下のprediction formulaで使用する。

```text
Taylor2 curve
```

#### 意味

2次差分項の強さを指定する。

```text
0.0:
  2次項を無効化する。

0.25:
  弱めに曲率を反映する。初期推奨値。

0.50:
  中程度に曲率を反映する。

1.0:
  2次項を完全に使う。
```

---

### 3.9 Fallback

```text
Fallback:
  previous residual
```

これはユーザー操作可能な設定ではなく、固定仕様とする。

以下の場合は必ずprevious residualへfallbackする。

```text
1. Prediction formulaがTeaCache (residual only)
2. 必要なresidual履歴が不足している
3. residual shapeが一致しない
4. dtype変換に失敗した
5. NaNまたはInfが出た
6. 予測計算で例外が発生した
7. UjiCacheが無効
```

fallback時:

```text
r_used = r_last
```

ただし、`r_last` 自体が存在しない場合はprevious residualへfallbackできない。
この場合はskipを許可せずfull calculationへ戻すか、UjiCacheをunavailable / degradedとして扱う。

---

## 4. 用語定義

### 4.1 actual forward

TeaCacheがskipせず、通常通りDiT / transformer blocksを計算するstep。

このstepで新しいresidualを得る。

```text
r_actual = output_hidden - input_hidden
```

または、既存TeaCache実装が保存しているresidual定義に従う。

UjiCacheはTeaCache側のresidual定義を変更しない。

---

### 4.2 skip step

TeaCacheが通常計算を省略すると判断したstep。

本来のTeaCacheでは、このstepでprevious residualを使う。

UjiCacheは、このskip stepで使用するresidualをprediction formulaにより置き換える。

---

### 4.3 previous residual

直近のactual forwardで得たresidual。

```text
r_last
```

TeaCache標準では、skip stepでこれをそのまま使用する。

---

### 4.4 skip streak

最後のactual forward以降、連続してskipされた回数。

現在のskip stepを含む。

例:

```text
step 20: actual forward
step 21: skip  -> skip_streak = 1
step 22: skip  -> skip_streak = 2
step 23: actual forward -> skip_streak reset
step 24: skip  -> skip_streak = 1
```

---

### 4.5 late phase

sampling progressが `Use prediction after progress` を超えた後半区間。

```text
late_phase = progress > use_prediction_after_progress
```

late phaseでは、skip streak条件に関係なく、skip 1回目からprediction formulaを使用する。

---

## 5. コアロジック

UjiCacheは、TeaCacheがskip stepだと判断した場合のみ実行される。

actual forward stepでは、UjiCacheは新しいresidual履歴を保存するだけで、出力は変更しない。

---

### 5.1 actual forward時の処理

TeaCacheが通常計算を行ったstepで、UjiCacheはresidual履歴を更新する。

疑似コード:

```python
def on_actual_forward(step_index, input_hidden, output_hidden, branch_id):
    residual = output_hidden - input_hidden
    state[branch_id].residual_history.append({
        "step_index": step_index,
        "residual": residual.detach()
    })

    # 履歴は必要最小限だけ保持する
    # linear: 2点
    # Taylor2: 3点
    # 将来拡張を見越すなら最大5点程度
    state[branch_id].trim_history(max_items=5)

    state[branch_id].skip_streak = 0
```

`branch_id` はcond/uncondを分けるために使う。
CFGでcond/uncondが同一batch化されている場合でも、可能ならbranchごとに履歴を分ける。

---

### 5.2 skip step時の処理

TeaCacheがskipすると決めたstepで、UjiCacheが使用residualを決める。

疑似コード:

```python
def on_skip_step(step_index, total_steps, branch_id, progress):
    s = state[branch_id]

    if not s.residual_history:
        return request_full_calculation_or_mark_unavailable("missing_previous_residual")

    s.skip_streak += 1

    # progress はTeaCache側のprogress値をそのまま渡す
    late_phase = progress > use_prediction_after_progress

    if prediction_formula == "TeaCache (residual only)":
        return fallback_previous_residual(s)

    # late phaseではskip streak条件を無視してpredictionを許可する
    if late_phase:
        prediction_allowed = True
    else:
        prediction_allowed = s.skip_streak >= apply_prediction_from_skip

    if not prediction_allowed:
        return fallback_previous_residual(s)

    try:
        if prediction_formula == "Linear extrapolation":
            return predict_linear(s, step_index)

        if prediction_formula == "Taylor2 curve":
            return predict_taylor2(s, step_index)

    except Exception:
        return fallback_previous_residual(s)

    return fallback_previous_residual(s)
```

---

## 6. 優先順位ルール

UjiCacheの挙動は以下の優先順位で決める。

```text
1. TeaCacheがactual forwardすると決めたstep
   → UjiCacheは出力を変更しない。
   → residual履歴だけ更新する。

2. TeaCacheがskipすると決めたstep
   → UjiCacheの対象になる。

3. previous residualが存在しない
   → previous residualへfallbackできない。
   → full calculationへ戻すか、UjiCacheをunavailable / degradedとして扱う。

4. Prediction formulaが TeaCache (residual only)
   → 常にprevious residualを使う。

5. progress > Use prediction after progress
   → late phase。
   → skip streakに関係なくpredictionを使う。

6. progress <= Use prediction after progress
   → 前半〜中盤。
   → skip_streak >= Apply prediction from skip # の場合だけpredictionを使う。

7. predictionに必要な履歴が足りない
   → previous residualへfallback。

8. prediction計算で異常が出る
   → previous residualへfallback。
```

---

## 7. 挙動例

### 7.1 設定例

```text
total_steps = 30
Use prediction after progress = 0.70
Apply prediction from skip # = 2
Prediction formula = Linear extrapolation
```

late phase判定:

```text
step 20:
  progress = 20 / 29 = 0.689
  late_phase = False

step 21:
  progress = 21 / 29 = 0.724
  late_phase = True
```

---

### 7.2 前半〜中盤で 5〜7 がskipされた場合

```text
step 4: actual forward
step 5: skip
step 6: skip
step 7: skip
```

挙動:

```text
step 5:
  progress < 0.70
  skip_streak = 1
  apply_prediction_from_skip = 2
  → previous residual

step 6:
  progress < 0.70
  skip_streak = 2
  → prediction

step 7:
  progress < 0.70
  skip_streak = 3
  → prediction
```

---

### 7.3 前半〜中盤で 9〜10 がskipされた場合

```text
step 8: actual forward
step 9: skip
step 10: skip
```

挙動:

```text
step 9:
  progress < 0.70
  skip_streak = 1
  → previous residual

step 10:
  progress < 0.70
  skip_streak = 2
  → prediction
```

---

### 7.4 後半で 23〜25 がskipされた場合

```text
step 22: actual forward
step 23: skip
step 24: skip
step 25: skip
```

挙動:

```text
step 23:
  progress > 0.70
  late_phase = True
  skip_streak = 1
  → prediction

step 24:
  progress > 0.70
  late_phase = True
  skip_streak = 2
  → prediction

step 25:
  progress > 0.70
  late_phase = True
  skip_streak = 3
  → prediction
```

late phaseでは、`Apply prediction from skip #` は無視される。

---

## 8. branch管理

CFGが有効な場合、slotごとにresidualの変化が異なる可能性がある。
したがって、既存TeaCache実装が使っているslot単位に合わせて履歴を分ける。

```text
branch_id = slot_0
branch_id = slot_1
```

cond/uncondとの対応が分かる場合のみ、その意味をログに出す。

履歴はslotごとに独立させる。

```python
state = {
    branch_id: {
        "residual_history": [],
        "skip_streak": 0
    }
}
```

slotの識別ができない場合は、batch単位で一括管理してもよいが、ログに警告を出す。

---

## 9. dtype / device方針

prediction計算は安全のため内部的にfloat32で行ってよい。

ただし、最終的に返すresidualは元のresidual dtypeへ戻す。

```python
orig_dtype = r_last.dtype
orig_device = r_last.device

r_pred = r_pred.to(device=orig_device, dtype=orig_dtype)
```

deviceは原則としてresidualが存在するdeviceに合わせる。

CPUへ移動しない。
不要なGPU-CPU転送を発生させない。

---

## 10. 数値安全性

prediction結果に以下が含まれる場合はfallbackする。

```text
NaN
Inf
shape mismatch
device mismatch
dtype conversion failure
extreme norm explosion
```

任意でnorm guardを入れる。

例:

```python
if norm(r_pred) > norm(r_last) * max_norm_ratio:
    fallback
```

初期実装では `max_norm_ratio` はUIに出さなくてよい。
内部定数として以下を使う。

```text
max_norm_ratio = 3.0
```

---

## 11. ログ

debug modeまたはverbose modeでは、skip stepごとに以下を出力できるようにする。

```text
[UjiCache] step=23/30 progress=0.767 late=True streak=1 formula=linear action=prediction
[UjiCache] step=9/30 progress=0.300 late=False streak=1 formula=linear action=fallback reason=streak
[UjiCache] step=10/30 progress=0.333 late=False streak=2 formula=linear action=prediction
[UjiCache] step=24/30 progress=0.800 late=True streak=2 formula=taylor2 action=fallback reason=insufficient_history
```

集計ログ:

```text
[UjiCache] total_skip_steps=9
[UjiCache] prediction_used=5
[UjiCache] fallback_used=4
[UjiCache] fallback_reasons={streak:2, insufficient_history:1, numeric_error:1}
```

---

## 12. Infotext / metadata

生成画像のmetadataには以下を保存する。

```text
UjiCache enabled: true
UjiCache formula: Linear extrapolation
UjiCache use_prediction_after_progress: 0.00
UjiCache apply_prediction_from_skip: 2
UjiCache prediction_strength: 0.50
UjiCache taylor2_curve_strength: 0.25
```

`Prediction formula = TeaCache (residual only)` の場合も保存する。
比較実験で条件を再現するため。

---

## 13. 初期推奨値

```text
Enable UjiCache:
  false

Preset mode:
  Custom

Prediction formula:
  TeaCache (residual only)

Use prediction after progress:
  0.00

Apply prediction from skip #:
  2

Prediction strength:
  0.50

Taylor2 curve strength:
  0.25

Fallback:
  previous residual
```

実験時の推奨開始点:

```text
Prediction formula:
  Linear extrapolation

Use prediction after progress:
  0.70

Apply prediction from skip #:
  2

Prediction strength:
  0.50
```

---

## 14. 実装上の重要事項

UjiCacheは、土台となるTeaCacheのskip判定を置き換えない。

UjiCacheはTeaCache判定ロジックを内包するため、通常TeaCache patchとは同時に有効化しない。

以下は行わない。

```text
- skipするstepの再判定
- UjiCache予測式によるTeaCache thresholdの実行中の動的変更
- schedulerの変更
- model weightの変更
- attention kernelの変更
```

UIで指定された `Rel L1 Threshold` は、土台となるTeaCache判定に渡す設定値として扱ってよい。
UjiCacheの予測式は、生成中にこのthresholdを自動変更しない。

UjiCacheが行うのは、TeaCacheがskipすると決めたstepで、

```text
r_last
```

の代わりに、

```text
r_pred
```

を返すことだけである。

---

## 15. 最小実装の成功条件

MVPでは以下を満たせばよい。

```text
1. UIが表示される。
2. Enable UjiCacheでON/OFFできる。
3. TeaCache (residual only) がTeaCache標準と同じ結果になる。
4. Linear extrapolationが動く。
5. Taylor2 curveが動く。
6. progress > Use prediction after progress ではskip streak 1からpredictionを使う。
7. progress <= Use prediction after progress ではskip streak条件を満たした場合のみpredictionを使う。
8. 履歴不足時にprevious residualへfallbackする。
9. NaN/Inf発生時にprevious residualへfallbackする。
10. previous residualが存在しない場合はfull calculationへ戻すか、UjiCache unavailable / degradedとして扱う。
11. debug logでprediction/fallbackの理由が確認できる。
```
