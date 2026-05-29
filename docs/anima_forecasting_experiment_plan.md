# Anima base v1.0 推論高速化：Feature Forecasting 実験計画書 v2

作成日: 2026-05-29  
対象: Anima base v1.0 / Forge Neo / Nz-Anima-PredLab  
目的: DiT 推論本体の 1 step あたり計算を、キャッシュ・外挿・軽量近似によって高速化できるかを検証する。

---

## 0. v2での主な更新点

前回案では、主に以下の3候補を中心にしていた。

1. block output forecasting
2. cross-attention output forecasting
3. MLP output forecasting

その後、`Nz-Anima-PredLab` の TeaCache 実装を確認した結果、TeaCache が保持している再利用本体は、単なる判定信号ではなく、Anima block stack 前後の residual であることが分かった。

現在の実装では概念的に以下を保存している。

```text
ori_x = block stack input
for block in model.blocks:
    x = block(x, ...)
residual = block stack output - ori_x
```

skip時には、以下のように使われる。

```text
x = x + previous_residual
```

したがって、v2では **TeaCache residual forecasting** を最重要候補として追加する。

ただし、これは **cross-attention output forecasting** と **MLP output forecasting** を候補から外すという意味ではない。両者は「最初に実装する高速化方式」としては優先度を下げるが、観測データは取る。

---

## 1. 問題設定

Anima base v1.0 は、SDXL と比較すると推論が遅い。実測では解像度を上げたときの生成時間がほぼ画素数に比例して増えており、Attention 単体というより、DiT block 全体、すなわち projection / attention / MLP / activation 読み書きが広く効いている可能性が高い。

既に SageAttention2 が利用されており、Attention backend 単体の高速化余地は大きくない可能性がある。

そこで、次の方針を採る。

```text
GPUにより多く計算させるのではなく、
DiTに計算させる回数・範囲そのものを減らす。
```

Spectrum は DiT denoiser の最終出力を過去stepから予測し、そのstepの DiT forward 全体をスキップする。コミュニティ報告では、TeaCacheより画風崩れが小さい場合がある。

このことから、Animaの denoising trajectory には、毎step厳密な DiT forward を要しない、時間方向に滑らかな領域が存在する可能性がある。

本実験の目的は、Spectrum final output よりもさらに安全・軽量に近似できる予測対象を探すことである。

---

## 2. 基本仮説

### 仮説A: DiT内部または周辺に、Spectrum target より予測しやすい特徴量が存在する

Spectrum は最終 denoiser output を予測する。だが、Anima の内部表現または residual の中に、より低曲率で、軽い数式で近似しやすいものがあるかもしれない。

候補として以下を見る。

- TeaCache residual
- block output
- cross-attention output
- MLP output

### 仮説B: ただし「滑らか」だけでは足りない

候補となるtensorは、以下3条件を満たす必要がある。

```text
1. timestep方向に滑らかである
2. 軽量なスクリプトベースの計算で予測できる
3. その近似誤差が後段で増幅されにくい
```

特に cross-attention output や MLP output は、単体では滑らかでも、block内部の他の計算との整合性が崩れる可能性がある。

---

## 3. 予測対象の整理

### 3.1 比較基準: Spectrum final output

Spectrum が予測しているもの。

```text
z_t, timestep, text conditioning
  ↓
Anima denoiser forward 全体
  ↓
final denoiser output
```

これは今回の主な比較基準であり、必ず保存・解析する。

### 3.2 最優先候補: TeaCache residual

`Nz-Anima-PredLab` の TeaCache が保存している residual。

```text
TeaCache residual = block_stack_output - block_stack_input
```

Animaの処理でいえば、概念的には以下である。

```text
latent
  ↓
prepare_embedded_sequence
  ↓
block_stack_input
  ↓
Anima blocks
  ↓
block_stack_output
  ↓
final_layer
  ↓
unpatchify
  ↓
final denoiser output
```

TeaCache residual は `block_stack_output - block_stack_input` であり、final_layer / unpatchify より前にある。

これは Spectrum final output より内側だが、cross-attention や MLP 単体よりは外側であり、内部整合性を比較的保ちやすい可能性がある。

最初に `previous_residual` の単純再利用ではなく、以下のような予測値に置き換えられるかを検証する。

```text
x = x + predicted_residual
```

### 3.3 高優先候補: block output

各 Anima block の出力。

```text
h_i_input
  ↓
Block i
  ↓
h_i_output
```

block単位の出力は、AttentionやMLP単体より内部整合性を保ちやすい一方で、TeaCache residualより細かく層ごとの性質を調べられる。

### 3.4 観測対象として残す: cross-attention output

各block内の cross-attention branch 出力。

テキスト条件側のK/Vは固定であるため、比較的滑らかな可能性がある。しかし、画像token側のQueryはstepごとに変化する。

予想されるリスク:

- プロンプト追従の低下
- キャラクター属性・衣装・小物の曖昧化
- self-attention / MLP との内部不整合

したがって実装優先度は下げるが、観測データは取る。

### 3.5 観測対象として残す: MLP output

各block内の MLP / FeedForward branch 出力。

MLPは計算量が大きいため、もし予測可能なら高速化効果は大きい。一方で、線画・色・質感・局所ディテールに影響しやすい可能性がある。

予想されるリスク:

- 線が眠くなる
- 顔や髪の微細な崩れ
- 服飾や小物のディテール劣化
- 高周波成分の不足

したがって実装優先度は下げるが、観測データは取る。

### 3.6 後続候補: token-wise / component-wise variants

ToCaやSVD-Cache系の先行研究を踏まえると、tensor全体ではなく、token単位・主成分単位で予測可能性が異なる可能性がある。

ただし初期段階では実装を複雑化させすぎない。まずは上記tensorを保存し、オフライン解析でtoken-wise / PCA / SVD の有望性を見る。

---

## 4. 実装候補の優先順位と観測対象の優先順位を分ける

### 4.1 実装候補としての優先順位

最初に高速化実装へ進めるなら、以下の順に考える。

```text
1. TeaCache residual forecasting
2. block output forecasting
3. cross-attention output forecasting
4. MLP output forecasting
5. token-wise / component-wise forecasting
```

理由:

- TeaCache residual は既存TeaCache実装の延長で差し込みやすい
- block stack全体の効果をresidualとして扱うため、内部部品単体より整合性が高そう
- block output は層ごとの性質を利用できる
- cross-attention / MLP は部分出力なので、不整合リスクが高い

### 4.2 観測データ取得対象としての優先順位

観測データはより広く取る。

```text
必ず保存:
  - Spectrum final output
  - TeaCache residual
  - selected block outputs

軽量統計は全blockで保存:
  - block output
  - cross-attention output
  - MLP output

代表blockのみ生tensor保存:
  - cross-attention output
  - MLP output
```

つまり、cross-attention output と MLP output は、実装候補としては後ろに回すが、観測対象から外さない。

---

## 5. 保存形式

### 5.1 推奨構成

```text
生tensor:
  Zarr

軽量統計:
  Parquet または CSV

実験条件:
  meta.json

小規模共有用:
  safetensors
```

### 5.2 推奨ディレクトリ構成

```text
anima_probe_run_001/
  meta.json
  stats.parquet
  tensors.zarr/
    spectrum_final_output/
      all_steps
    teacache_residual/
      all_steps
    block_output/
      block_00
      block_07
      block_14
      block_21
      block_27
    cross_attn_output/
      block_07
      block_14
      block_21
    mlp_output/
      block_07
      block_14
      block_21
```

### 5.3 Zarrを推す理由

- chunk単位で読み書きしやすい
- 巨大tensorを部分的に読める
- step / block / tensor種別ごとに階層化しやすい
- 圧縮をかけられる

### 5.4 safetensorsの位置づけ

`safetensors` は安全で高速なtensor保存形式だが、追記型ログや階層的な実験データベースにはZarrのほうが扱いやすい。

`safetensors` は以下の用途に向く。

```text
- 有望候補の小規模dump
- 再現用サンプル
- 外部共有用
```

---

## 6. 保存すべきメタデータ

`meta.json` には最低限以下を保存する。

```json
{
  "model": "Anima base v1.0",
  "extension": "Nz-Anima-PredLab",
  "resolution": "1024x1024",
  "steps": 32,
  "scheduler": "er_sde or actual scheduler name",
  "cfg": 4.0,
  "seed": 123456,
  "prompt_id": "prompt_001",
  "prompt_hash": "...",
  "negative_prompt_hash": "...",
  "attention_backend": "sageattention2",
  "teacache_enabled": false,
  "spectrum_enabled": false,
  "dtype": "fp16 or bf16",
  "notes": "..."
}
```

---

## 7. 保存すべき軽量統計

全対象tensorについて、可能なら以下を毎step保存する。

```text
run_id
prompt_id
seed
step_index
progress
tensor_type
block_index
shape
dtype
mean
std
norm_l2
mean_abs
max_abs
relative_step_diff
relative_curvature
cos_prev
cos_delta
```

### 7.1 一次差分

```text
relative_step_diff = ||x_t - x_{t-1}|| / ||x_t||
```

step間変化の大きさ。

### 7.2 二次差分 / 曲率

```text
relative_curvature = ||x_t - 2x_{t-1} + x_{t-2}|| / ||x_t||
```

外挿しやすさの目安。小さいほどSpectrum的予測に向く。

### 7.3 cosine similarity

```text
cos_prev = cos(x_t, x_{t-1})
```

方向がどれだけ保たれているか。

### 7.4 差分方向のcosine

```text
cos_delta = cos(x_t - x_{t-1}, x_{t-1} - x_{t-2})
```

変化方向が安定しているか。

---

## 8. 予測しやすさの解析方法

保存したtensorに対して、まず軽量な外挿器を試す。

### 8.1 Previous value baseline

```text
x_t ≈ x_{t-1}
```

TeaCache的な単純再利用に近い。

### 8.2 Linear extrapolation

```text
x_t ≈ 2x_{t-1} - x_{t-2}
```

これで当たるなら、時間方向の軌道はかなり滑らか。

### 8.3 Adams-Bashforth風外挿

```text
x_t ≈ x_{t-1} + 1.5Δ_{t-1} - 0.5Δ_{t-2}
```

Flow Matching的な軌道に向く可能性がある。

### 8.4 Chebyshev + ridge regression

Spectrumに近い方式。Spectrum final output と同じ予測器を、TeaCache residual / block output / cross-attention output / MLP output に対しても適用し、誤差を比較する。

### 8.5 PCA / SVD後の予測

tensor全体ではなく、主成分係数だけを予測する。

```text
tensor
  ↓
PCA / SVD
  ↓
上位主成分係数の時系列を予測
  ↓
復元
```

SVD-Cache系の先行研究と接続する重要な解析である。

---

## 9. 評価指標

### 9.1 tensor予測誤差

```text
relative_error = ||x_true - x_pred|| / ||x_true||
```

### 9.2 cosine error

```text
cosine_error = 1 - cos(x_true, x_pred)
```

### 9.3 Spectrum final outputとの比較

各候補について、Spectrum final output の予測誤差と比較する。

```text
candidate_error < spectrum_output_error
```

なら、その候補はSpectrumより予測しやすい可能性がある。

### 9.4 downstream impact

最終的には、候補tensorだけを予測値に置き換えたとき、最終 denoiser output がどれだけズレるかを見る。

```text
candidate tensor を予測値に置換
  ↓
残り計算を通常実行
  ↓
final denoiser output の差分を測る
```

これが最重要である。

### 9.5 画像評価

数値誤差だけでなく、最終画像で以下を確認する。

```text
- 絵柄が変わるか
- 顔が変わるか
- 線が眠くなるか
- 手指が崩れるか
- 髪・目・装飾の細部が劣化するか
- 色や陰影が古いstepのまま残るか
- プロンプト追従が落ちるか
- 構図がズレるか
```

---

## 10. 実験フェーズ

### Phase 1: 軽量統計ログの取得

目的:
生tensorを大量保存する前に、全block・全候補の概要を掴む。

対象:

```text
- Spectrum final output
- TeaCache residual
- block output 全block
- cross-attention output 全block
- MLP output 全block
```

保存:

```text
stats.parquet または stats.csv
```

この時点では、必要に応じて生tensor保存を最小限にする。

### Phase 2: 代表blockの生tensor保存

目的:
実際に予測器を当てるためのtensor列を保存する。

推奨対象:

```text
必須:
  - Spectrum final output
  - TeaCache residual
  - block output: block 0, 7, 14, 21, 27

代表保存:
  - cross-attention output: block 7, 14, 21
  - MLP output: block 7, 14, 21
```

block数が28でない場合は、浅・中・深から均等に選ぶ。

### Phase 3: オフライン予測実験

各候補に対して以下を比較する。

```text
- previous value
- linear extrapolation
- Adams-Bashforth風
- Chebyshev + ridge
- PCA/SVD + linear
- PCA/SVD + Chebyshev
```

出力:

```text
candidate_ranking.csv
prediction_error_plots/
```

### Phase 4: 置換実験

オフラインで有望だった候補のみ、実際の推論中に予測値へ置き換える。

最初の置換候補:

```text
1. TeaCache residual
2. block output selected layers
3. cross-attention output selected layers, if unexpectedly smooth
4. MLP output selected layers, if unexpectedly smooth
```

### Phase 5: 画像評価

同一seed・同一promptで以下を比較する。

```text
- baseline OFF
- Spectrum
- TeaCache reuse
- TeaCache residual forecast
- block output forecast
- cross-attention output forecast
- MLP output forecast
```

ただし、最初から全組み合わせを試さない。Phase 3で絞った候補だけでよい。

---

## 11. 最初の実験条件

初期実験はメモリ管理の副作用を避けるため、1536x1536ではなく1024x1024を主戦場にする。

推奨条件:

```text
resolution: 1024x1024
batch size: 1
steps: 32
CFG: 固定
scheduler: 固定
attention backend: SageAttention2
TeaCache: OFF, unless collecting TeaCache internal values in dry-run style
Spectrum: OFF, unless collecting Spectrum baseline
prompt: 3〜5種類
seed: 各prompt 2〜3個
```

1536x1536は、1024で有望候補が出てから確認する。理由は、1536では専用VRAMに余裕があるように見えても共有GPUメモリへ漏れる現象が観測されており、Forge Neo側のOOM回避やメモリ管理の副作用が混ざる可能性があるため。

---

## 12. データ量対策

生tensorは巨大化しやすい。

方針:

```text
1. 軽量統計は広く取る
2. 生tensorは代表blockに絞る
3. 保存dtypeは fp16 / bf16 を基本にする
4. 解析時に float32 へ変換する
5. 予測器評価後、有望候補だけ追加dumpする
```

推奨:

```text
Raw dump full coverage は避ける。
Stats full coverage + raw selected coverage にする。
```

---

## 13. 使うライブラリ

保存:

```text
zarr
safetensors
json
pandas / polars
pyarrow
```

解析:

```text
numpy
torch
scipy
scikit-learn
matplotlib
```

インストール例:

```bash
pip install zarr safetensors numpy scipy scikit-learn pandas pyarrow matplotlib
```

---

## 14. 成功判定

ある候補を次段階へ進める条件は以下。

```text
1. Spectrum final output より予測誤差が小さい、または同程度
2. 予測器がDiT forwardより十分軽い
3. 置換時の downstream impact が小さい
4. 画像上の崩れがSpectrum以下
5. 実測で有意な高速化がある
```

特に重要なのは、tensor自体の予測誤差ではなく、**置換したときの最終画像劣化**である。

---

## 15. 現時点の暫定優先順位

実装候補としては:

```text
1. TeaCache residual forecasting
2. block output forecasting
3. cross-attention output forecasting
4. MLP output forecasting
5. token-wise / component-wise variants
```

観測対象としては:

```text
0. Spectrum final output, as baseline
1. TeaCache residual
2. block output
3. cross-attention output
4. MLP output
5. token-wise / PCA / SVD derived features
```

重要:

```text
cross-attention output と MLP output は候補から外さない。
ただし最初に高速化実装する対象としては後ろに回す。
観測データは取る。
```

---

## 16. 次に実装すべきプローブ

最初に実装するべきプローブは以下。

### 16.1 TeaCache residual dump

保存対象:

```text
residual = block_stack_output - block_stack_input
```

TeaCache skipを行わないdry-run状態でも、このresidual列を保存できるようにする。

### 16.2 Spectrum final output dump

保存対象:

```text
model_function output / final denoiser output
```

Spectrum有効時だけでなく、baseline実計算時にも保存する。

### 16.3 block output probe

各blockのforward出力をhookする。最初は代表blockのみ生tensor保存。全blockについては軽量統計を保存。

### 16.4 cross-attention / MLP output probe

全blockについて軽量統計を保存。代表blockのみ生tensor保存。

---

## 17. 最終メッセージ

今回のv2計画では、TeaCache residual を新たに最重要候補として追加する。

ただし、cross-attention output forecasting と MLP output forecasting を捨てるわけではない。これらは実装優先度こそ下げるが、滑らかさや予測可能性は実測しなければ分からない。したがって観測対象には残す。

この実験の核心は、以下を比較することである。

```text
Spectrum final output
vs
TeaCache residual
vs
block output
vs
cross-attention output
vs
MLP output
```

そして、最終的にはこう判断する。

```text
どのtensorが、
DiT forwardより圧倒的に軽い計算で予測でき、
かつ画像崩れを最小限に抑えられるか。
```

---

## 18. 先行研究URL一覧

本実験計画に関連する先行研究として、以下のarXiv論文を参照する。

1. https://arxiv.org/abs/2406.01733
2. https://arxiv.org/abs/2410.05317
3. https://arxiv.org/abs/2506.03275
4. https://arxiv.org/abs/2409.18523
5. https://arxiv.org/abs/2411.10510
6. https://arxiv.org/abs/2601.07396
7. https://arxiv.org/abs/2505.20353
8. https://arxiv.org/abs/2505.05829
9. https://arxiv.org/abs/2503.05156
10. https://arxiv.org/abs/2503.07120
11. https://arxiv.org/abs/2509.13789
12. https://arxiv.org/abs/2512.17298
13. https://arxiv.org/abs/2504.03140
