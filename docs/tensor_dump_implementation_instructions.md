# Nz-Anima-PredLab: Tensor Dump 実装作業指示書

## 0. 目的

`Nz-Anima-PredLab` に、Anima 推論中の中間テンソルを研究用ログとして保存する機能を追加する。

保存対象は以下の5種類とする。

1. TeaCache residual
2. block output
3. cross-attention output
4. MLP output
5. Spectrum final output

目的は、各テンソルが timestep 方向にどれくらい滑らかに変化するか、また DiT 本体より軽いスクリプトベースの計算でどれくらい近似できるかを、後段のオフライン解析で評価できるようにすることである。

本作業は、推論高速化そのものの実装ではない。まずは **データ収集基盤** を作る。

---

## 1. 前提

### 1.1 現在のUI構成

現在の `script.py` では、トップレベルの `Nz-Anima-PredLab` アコーディオン内に `Debug log mode` サブアコーディオンが存在する。

この中に、以下の6つのチェックボックスを追加する。

```text
Dump TeaCache residual
Dump block output
Dump cross-attention output
Dump MLP output
Dump Spectrum final output
Dump baseline final output
```

既存の `Debug log mode` は、従来のコンソールログ設定を移設した場所である。今回の tensor dump も、実験・調査用ログなので、このサブアコーディオンに置く。

### 1.2 現在のTeaCache実装

現在の TeaCache 実装では、Anima の block stack 入力を `ori_x` として保存し、全 block 通過後の `x_B_T_H_W_D` との差分を residual として計算している。

概念的には以下である。

```python
ori_x = x_B_T_H_W_D.to(cache_device)
for block in model.blocks:
    x_B_T_H_W_D = block(...)
residual = x_B_T_H_W_D.to(cache_device) - ori_x
```

skip 時には、保存済みの `previous_residual` を現在の hidden state に加算している。

```python
x[start:end] = x[start:end] + residual.to(x.device)
```

したがって、今回保存する `TeaCache residual` は、以下の量である。

```text
TeaCache residual = block_stack_output - block_stack_input
```

これは Spectrum final output とは異なる。Spectrum は denoiser final output を予測対象にする。一方、TeaCache residual は final_layer / unpatchify より前の hidden token 空間にある block stack residual である。

### 1.3 現在のSpectrum実装

Spectrum は `model_function_wrapper` 経由で denoiser forward 全体を包み、実計算 step では `model_function(...)` の出力 `out` を取得している。

今回保存する `Spectrum final output` は、実計算 step における `out` とする。

初期実装では、forecast step の予測値は保存対象に含めない。必要になった場合、後続タスクとして `Spectrum forecast output` を別チェックボックスで追加する。

---

## 2. 保存形式

保存形式は、既存の `anima_forecasting_experimental_plan.md` の方針を維持する。

```text
生tensor:
  Zarr

軽量統計:
  Parquet

実験条件:
  meta.json
```

### 2.1 なぜ Zarr か

Zarr は、chunked / compressed なN次元配列を保存できる。今回のように、step、block、tensor type ごとに巨大な配列を保存し、後で一部だけ読み出す用途に向いている。

`safetensors` は安全で高速だが、基本的に「まとまったtensorを1ファイルに保存する」用途に向いている。追記型の実験ログや部分読み出しには Zarr のほうが扱いやすい。

### 2.2 なぜ Parquet か

Parquet は、軽量統計を表形式で保存するのに向いている。

たとえば、各 step / block / tensor_type ごとに、以下のような統計を保存する。

```text
mean
std
norm_l2
norm_l1
max_abs
shape
dtype
device_before_dump
```

後段で pandas / polars / pyarrow から読みやすい。

### 2.3 meta.json

`meta.json` には、実験条件とファイル構造を保存する。

最低限、以下を含める。

```json
{
  "schema_version": 1,
  "extension": "Nz-Anima-PredLab",
  "model": "Anima base v1.0",
  "generation_index": 12,
  "created_at": "2026-05-29T12:34:56+09:00",
  "steps": 32,
  "width": 1024,
  "height": 1024,
  "batch_size": 1,
  "sampler": "unknown_or_detected_value",
  "scheduler": "unknown_or_detected_value",
  "cfg_scale": "unknown_or_detected_value",
  "dump_flags": {
    "teacache_residual": true,
    "block_output": true,
    "cross_attention_output": true,
    "mlp_output": true,
    "spectrum_final_output": true,
    "baseline_final_output": true
  }
}
```

取得できない項目は `null` または `"unknown"` でよい。

---

## 3. 保存先ディレクトリ

ユーザー指定の保存先は以下である。

```text
\StabilityMatrix\Images\logs\YYYY-MM-DD
```

ただし、実装では絶対パスを決め打ちしない。画像出力ディレクトリから `Images` 相当の場所を推定し、その配下に `logs/YYYY-MM-DD` を作る。

推奨方針は以下。

1. `modules.shared.opts.outdir_samples`、`outdir_txt2img_samples`、または生成画像の保存先に近い値を参照する。
2. その出力先から `Images` 相当の親ディレクトリを推定する。
3. 推定できない場合は、現在の出力ディレクトリ直下に `logs/YYYY-MM-DD` を作る。
4. それも失敗した場合は、`Path.cwd() / "logs" / YYYY-MM-DD` にフォールバックする。

作成時は必ず以下を使う。

```python
log_dir.mkdir(parents=True, exist_ok=True)
```

### 3.1 runディレクトリ

日付ディレクトリ直下に、1生成ごとの run ディレクトリを作る。

例:

```text
StabilityMatrix/Images/logs/2026-05-29/
  run_20260529_123456_gen0012/
    meta.json
    stats.parquet
    tensors.zarr/
```

run名には時刻と `STATE.generation_index` を含める。

---

## 4. 出力ディレクトリ構造

推奨構造は以下。

```text
run_20260529_123456_gen0012/
  meta.json
  stats.parquet
  tensors.zarr/
    spectrum_final_output/
      actual
    baseline_final_output/
      actual
    teacache_residual/
      slot_0
      slot_1
    block_output/
      block_00
      block_01
      ...
    cross_attention_output/
      block_00
      block_01
      ...
    mlp_output/
      block_00
      block_01
      ...
```

各 Zarr array は、原則として以下の第1軸を持つ。

```text
[record_index, ...tensor_shape]
```

`record_index` と実際の step / block / slot の対応は `stats.parquet` に保存する。

Zarr array 名に step を含める方式は避ける。stepごとにarrayを作ると、後段解析が面倒になる。

---

## 5. 追加するファイル

### 5.1 `nz_anima_predlab/tensor_dump.py`

tensor dump 関連の処理は、`patcher.py` に直書きしない。

新規ファイルとして `tensor_dump.py` を追加する。

役割:

```text
- runディレクトリ作成
- meta.json 保存
- Zarr group / array 作成
- tensor append
- 軽量統計の蓄積
- stats.parquet 書き出し
- dump有効/無効判定
- 例外時の安全なフォールバック
```

### 5.2 `state.py`

以下のフラグを追加する。

```python
dump_teacache_residual: bool = False
dump_block_output: bool = False
dump_cross_attention_output: bool = False
dump_mlp_output: bool = False
dump_spectrum_final_output: bool = False
dump_baseline_final_output: bool = False
```

また、tensor dump 用の runtime 情報を保持する。

```python
tensor_dump_run_dir: str | None = None
tensor_dump_initialized: bool = False
tensor_dump_records: int = 0
tensor_dump_errors: int = 0
```

必要なら `RuntimeState.reset_runtime()` 相当の関数に、これらのリセット処理を追加する。

### 5.3 `script.py`

`Debug log mode` サブアコーディオン内に、6つのチェックボックスを追加する。

```python
with gr.Accordion("Debug log mode", open=False, elem_id="nzap-debug-panel"):
    ...
    dump_teacache_residual = gr.Checkbox(
        label="Dump TeaCache residual",
        value=False,
        elem_id="nzap-dump-teacache-residual",
    )
    dump_block_output = gr.Checkbox(
        label="Dump block output",
        value=False,
        elem_id="nzap-dump-block-output",
    )
    dump_cross_attention_output = gr.Checkbox(
        label="Dump cross-attention output",
        value=False,
        elem_id="nzap-dump-cross-attention-output",
    )
    dump_mlp_output = gr.Checkbox(
        label="Dump MLP output",
        value=False,
        elem_id="nzap-dump-mlp-output",
    )
    dump_spectrum_final_output = gr.Checkbox(
        label="Dump Spectrum final output",
        value=False,
        elem_id="nzap-dump-spectrum-final-output",
    )
    dump_baseline_final_output = gr.Checkbox(
        label="Dump baseline final output",
        value=False,
        elem_id="nzap-dump-baseline-final-output",
    )
```

各チェックボックスは、ONにしたら親の `Enable Nz-Anima-PredLab` もONになるようにする。

```python
for control in (...):
    control.change(
        fn=_enable_parent_if_child_enabled,
        inputs=[control],
        outputs=[enabled],
    )
```

`return [...]` の末尾に6つを追加する。

`_apply_ui_args()` の分岐も更新する。

例:

```python
if len(script_args) >= 54:
    STATE.apply_options(*script_args[:54])
    return
```

既存の `>=48`, `>=46`, `>=35`, `>=23`, `>=4` の互換分岐は残す。

### 5.4 `patcher.py`

以下を追加・変更する。

```text
- TeaCache residual dump: TeaCache full calculation 時に residual を保存
- Spectrum final output dump: Spectrum actual forward 時に out を保存
- block output dump: Block.forward wrapper で output を保存
- cross-attention output dump: SelfCrossAttention.compute_attention wrapper で cross-attn branch output を保存
- MLP output dump: MLP submodule forward wrapper で output を保存
```

---

## 6. tensor_dump.py の設計

### 6.1 依存ライブラリ

必須:

```text
zarr
numpy
pandas
pyarrow
```

PyTorch tensor を扱うので `torch` は既存環境にある前提。

依存追加が問題になる場合は、Parquetのみ後回しにして、初期実装では `stats.jsonl` にフォールバックできるようにする。

推奨:

```python
try:
    import zarr
    import pandas as pd
except Exception as exc:
    # dump unavailable として安全に無効化
```

### 6.2 API案

`tensor_dump.py` に以下の関数を用意する。

```python
def any_tensor_dump_enabled() -> bool:
    ...


def ensure_run_dir(p=None) -> Path | None:
    ...


def initialize_run_if_needed(p=None) -> None:
    ...


def dump_tensor(
    tensor_type: str,
    tensor,
    *,
    step_index: int | None = None,
    call_index: int | None = None,
    block_index: int | None = None,
    slot: int | None = None,
    decision: str | None = None,
    attn_type: str | None = None,
    extra: dict | None = None,
) -> None:
    ...


def flush_stats() -> None:
    ...
```

### 6.3 dump_tensor の挙動

`dump_tensor()` は、以下の処理を行う。

1. dump機能が有効か確認する。
2. tensor が `torch.Tensor` でなければ何もしない。
3. `tensor.detach()` する。
4. GPU tensor の場合、CPUへ移す。
5. 保存 dtype は原則 `float16` または元 dtype のままにする。
6. Zarr に append する。
7. 軽量統計をメモリ上の list に追加する。
8. 例外が起きても推論を止めない。

重要:

```python
try:
    ...
except Exception as exc:
    STATE.tensor_dump_errors += 1
    warning(f"tensor_dump_failed type={tensor_type} reason={exc}")
```

推論中の実験ログ保存で、生成そのものを落としてはならない。

### 6.4 軽量統計

各保存ごとに、最低限以下を記録する。

```text
tensor_type
record_index
generation_index
step_index
call_index
block_index
slot
attn_type
decision
shape
dtype
numel
mean
std
abs_mean
l1_norm
l2_norm
max_abs
zarr_path
```

`mean/std/norm` は CPU tensor に対して float32 で計算する。

```python
x = tensor_cpu.float()
mean = float(x.mean().item())
std = float(x.std().item())
abs_mean = float(x.abs().mean().item())
l2_norm = float(torch.linalg.vector_norm(x).item())
max_abs = float(x.abs().max().item())
```

巨大tensorでは統計計算も重い。必要なら `STATE.dump_compute_stats` のような追加フラグで無効化できる設計にしてもよい。

---

## 7. 各テンソルのhook位置

## 7.1 TeaCache residual

### 条件

```text
STATE.dump_teacache_residual == True
かつ
STATE.teacache_enabled == True
```

TeaCache が disable の場合は何もしない。

### hook位置

`_teacache_forward_body()` の full calculation 分岐内。

現在の概念コード:

```python
if should_calc:
    ori_x = x_B_T_H_W_D.to(cache_device)
    for block in model.blocks:
        x_B_T_H_W_D = block(...)
    residual = x_B_T_H_W_D.to(cache_device) - ori_x
    ...
```

`residual` 計算直後に保存する。

```python
if STATE.dump_teacache_residual:
    dump_tensor(
        "teacache_residual",
        residual,
        step_index=step_index,
        decision="full",
        extra={
            "cond_or_uncond": list(cond_or_uncond),
            "cache_device": str(cache_device),
        },
    )
```

slotごとに分割した residual を保存するほうが後段解析しやすい。

推奨:

```python
for slot_index, key in enumerate(cond_or_uncond):
    residual_slice = residual[slot_index * batch_per_slot : (slot_index + 1) * batch_per_slot]
    dump_tensor(
        "teacache_residual",
        residual_slice,
        step_index=step_index,
        slot=int(key),
        decision="full",
    )
```

skip時には新規 residual は存在しないため保存しない。

---

## 7.2 Spectrum final output

### 条件

```text
STATE.dump_spectrum_final_output == True
かつ
STATE.spectrum_enabled == True
```

Spectrum が disable の場合は何もしない。

### hook位置

`_spectrum_model_function_wrapper_body()` の actual forward 分岐。

実計算時:

```python
out = _spectrum_actual_forward(model_function, args)
```

この直後に保存する。

```python
if STATE.dump_spectrum_final_output:
    dump_tensor(
        "spectrum_final_output",
        out,
        step_index=step_index,
        call_index=cnt,
        decision="actual",
        extra={"reason": reason},
    )
```

`spectrum_dry_run` 時も実計算outが取れるため、保存してよい。

forecast値は初期実装では保存しない。必要なら `decision="forecast"` として保存する拡張を後で追加する。

---

## 7.3 block output

### 条件

```text
STATE.dump_block_output == True
```

### hook位置

`backend.nn.anima.Block.forward` をwrapする。

既存の `block_structure_trace` / `identity_patch` と同様に `Block.forward` を差し替える。

ただし、patch衝突に注意する。

推奨:

```text
新規 patch kind: "tensor_dump"
```

この patch の中で、Block.forward と SelfCrossAttention.compute_attention と MLP forward をまとめてwrapする。

### block index

既存コードと同様に、call index と block数から block index を推定する。

```python
block_index = call_index % num_blocks
```

ただし、cond/uncondやbatch処理により呼び出し回数が増える可能性がある。`STATE.denoiser_calls` や `step_index` と合わせて記録する。

### 保存位置

```python
def dumped_block_forward(self, x_B_T_H_W_D, *args, **kwargs):
    output = original_block_forward(self, x_B_T_H_W_D, *args, **kwargs)
    if STATE.dump_block_output:
        dump_tensor(
            "block_output",
            output,
            step_index=max(0, STATE.denoiser_calls - 1),
            block_index=block_index,
            call_index=call_index,
        )
    return output
```

---

## 7.4 cross-attention output

### 条件

```text
STATE.dump_cross_attention_output == True
```

### hook位置

`backend.nn.anima.SelfCrossAttention.compute_attention` をwrapする。

既存実装では、`is_SelfAttn` で self / cross を判定できる。

### 保存対象

保存対象は、attention kernel直後の raw result ではなく、blockへ返る branch output とする。

つまり、概念的には以下。

```python
result = original_attention_kernel(...)
branch_output = self.output_dropout(self.output_proj(result))
return branch_output
```

既存の attention override 実装でもこの形で返している。

### 注意

`compute_attention` をwrapすると、既存の attention backend override / sparse attention patch と衝突する可能性がある。

最初の実装では、以下のどちらかにする。

#### 案A: tensor_dump patch は attention override / sparse attention と排他

安全。実装が簡単。

#### 案B: 既存 patch の内側で dump する

柔軟だが複雑。

初期実装では案Aを推奨する。

### 保存例

```python
if STATE.dump_cross_attention_output and not getattr(self, "is_SelfAttn", False):
    dump_tensor(
        "cross_attention_output",
        branch_output,
        step_index=max(0, STATE.denoiser_calls - 1),
        block_index=current_block_index,
        attn_type="cross",
    )
```

---

## 7.5 MLP output

### 条件

```text
STATE.dump_mlp_output == True
```

### 注意点

5種類の中で、MLP output がもっとも実装調査を要する。

理由:

```text
- Anima Block 内部の MLP submodule 名が、現時点の拡張コード上で明示されていない
- Block.forward を写経してMLP直後にhookを入れると、Forge Neo本体の更新に弱くなる
```

### 推奨実装方針

まず、diagnose用に `Block` の属性一覧をログ出力して、MLPらしい submodule 名を確認する。

候補名:

```text
mlp
ffn
ff
feed_forward
feedforward
```

該当 module が見つかったら、その `forward` をwrapする。

```python
original_mlp_forward = mlp_module.forward

def dumped_mlp_forward(*args, **kwargs):
    output = original_mlp_forward(*args, **kwargs)
    if STATE.dump_mlp_output:
        dump_tensor(
            "mlp_output",
            output,
            step_index=max(0, STATE.denoiser_calls - 1),
            block_index=current_block_index,
        )
    return output
```

### フォールバック

MLP module が見つからない場合は、警告を1回だけ出して、何も保存しない。

```text
mlp_dump_unavailable reason=mlp_module_not_found
```

この失敗で生成を止めない。

---

## 8. patch適用ロジック

### 8.1 tensor_dump patchの新設

`patcher.py` に以下を追加する。

```python
if kind == "tensor_dump":
    return _apply_tensor_dump_patch()
```

`remove_patch()` は既存の仕組みを使える。

### 8.2 _configure_generation_patches の変更

現在は TeaCache / Spectrum / Sparse / Attention override が排他的に適用されている。

Tensor dump は、原則として他の実験patchに追加で乗る補助patchである。

ただし、cross-attention / MLP / block output dump は `Block.forward` や `compute_attention` をwrapするため、patch衝突に注意する。

初期実装では以下を推奨する。

```text
- TeaCache residual dump は TeaCache patch 内で実装
- Spectrum final output dump は Spectrum patch 内で実装
- block/cross-attn/MLP dump は tensor_dump patch で実装
- tensor_dump patch は sparse_attention / attention_kernel override と同時使用しない
```

`_configure_generation_patches()` の最後に以下を追加する。

```python
if STATE.tensor_dump_block_level_active():
    if STATE.sparse_enabled or STATE.attention_override_active():
        remove_patch("tensor_dump")
        warning("tensor_dump_unavailable reason=conflicts_with_attention_or_sparse_patch")
    else:
        apply_patch("tensor_dump")
else:
    remove_patch("tensor_dump")
```

`tensor_dump_block_level_active()` は以下のような意味。

```python
return (
    self.dump_block_output
    or self.dump_cross_attention_output
    or self.dump_mlp_output
)
```

TeaCache residual と Spectrum final output だけなら `tensor_dump` patch は不要。

---

## 9. fallback仕様

### 9.1 TeaCache residual

```text
TeaCacheがdisabled:
  何も出力しない

TeaCacheがenabledだがpatch失敗:
  何も出力しない
  warningを出す

TeaCacheがskip decision:
  新規residualは存在しないため保存しない
```

### 9.2 Spectrum final output

```text
Spectrumがdisabled:
  何も出力しない

Spectrumがenabledだがpatch失敗:
  何も出力しない
  warningを出す

Spectrum forecast step:
  初期実装では保存しない

Spectrum actual step:
  outを保存する
```

### 9.3 block / cross-attn / MLP

```text
Anima Block / SelfCrossAttention が見つからない:
  何も出力しない
  warningを出す

MLP module が見つからない:
  MLP outputだけ無効化
  他のdumpは継続

保存中に例外:
  そのrecordだけ捨てる
  推論は継続
```

---

## 10. データ量対策

全step・全block・全tensorを保存すると巨大になる。

初期実装では、以下の制限を内部定数として入れる。
UIに出すのは後続タスクでよい。

```python
DUMP_BLOCK_START = 0
DUMP_BLOCK_END = 27
DUMP_MAX_RECORDS_PER_TYPE = 100000
DUMP_SAVE_DTYPE = "float16"
```

ただし、研究実験では全block統計が欲しいため、軽量統計は多めに保存してよい。

### 10.1 生tensor保存の推奨初期設定

```text
TeaCache residual:
  保存する

Spectrum final output:
  保存する

block output:
  block 0, 7, 14, 21, 27 だけ保存、または全block保存はユーザー判断

cross-attention output:
  block 7, 14, 21 のみ保存

MLP output:
  block 7, 14, 21 のみ保存
```

初期UIではblock範囲指定を省略してもよいが、コード上は定数で範囲制御できるようにする。

---

## 11. stats.parquet のスキーマ案

```text
schema_version: int
run_id: str
generation_index: int
tensor_type: str
record_index: int
step_index: int | null
call_index: int | null
block_index: int | null
slot: int | null
attn_type: str | null
decision: str | null
shape: str
dtype: str
saved_dtype: str
numel: int
mean: float
std: float
abs_mean: float
l1_norm: float
l2_norm: float
max_abs: float
zarr_path: str
extra_json: str
```

`extra_json` には、可変情報をJSON文字列で保存する。

例:

```json
{"reason": "warmup", "cache_device": "cuda:0"}
```

---

## 12. Zarr dataset設計

### 12.1 array作成

各 tensor_type / block / slot ごとに array を作る。

例:

```text
tensors.zarr/teacache_residual/slot_0
tensors.zarr/block_output/block_07
tensors.zarr/cross_attention_output/block_14
tensors.zarr/mlp_output/block_21
tensors.zarr/spectrum_final_output/actual
tensors.zarr/baseline_final_output/actual
```

### 12.2 append方式

Zarr array は `record_index` 軸に append する。

初回保存時に shape を確定する。

```python
array = group.create_dataset(
    name,
    shape=(0, *tensor_shape),
    chunks=(1, *chunk_shape),
    dtype=saved_dtype,
    compressor=...,  # 依存問題があれば省略
    maxshape=(None, *tensor_shape),
)
```

Zarr v3 / v2 のAPI差異に注意する。
プロジェクト環境で使う zarr のバージョンを固定できない場合は、薄いwrapper関数を作る。

---

## 13. 後段解析を見据えた注意

後段解析では、以下を比較する予定である。

```text
- Spectrum final output
- TeaCache residual
- block output
- cross-attention output
- MLP output
```

そのため、最低限、すべての保存recordに以下が必要。

```text
step_index
block_index または slot
record_index
shape
zarr_path
```

`step_index` が欠けると、時間方向の滑らかさを評価できない。

---

## 14. テスト項目

### 14.1 UIテスト

- `Debug log mode` 内に5チェックボックスが表示される。
- 各チェックボックスをONにすると、親の `Enable Nz-Anima-PredLab` がONになる。
- 既存の Debug log ドロップダウン、Print timing log、Verbose diagnose log が従来通り動作する。
- 古い引数数でも `STATE.apply_options` がエラーにならない。

### 14.2 TeaCache residual dump

条件:

```text
TeaCache ON
Dump TeaCache residual ON
```

期待:

```text
logs/YYYY-MM-DD/run_xxx/tensors.zarr/teacache_residual/... が作成される
stats.parquet に teacache_residual record が記録される
```

TeaCache OFFの場合:

```text
Dump TeaCache residual ON でも何も保存されない
生成は正常終了する
```

### 14.3 Spectrum final output dump

条件:

```text
Spectrum ON
Dump Spectrum final output ON
```

期待:

```text
spectrum_final_output/actual が作成される
actual forward step のみ保存される
```

Spectrum OFFの場合:

```text
Dump Spectrum final output ON でも何も保存されない
生成は正常終了する
```

Baseline final output:

```text
Spectrum OFF
Dump baseline final output ON
```

期待:

```text
baseline_final_output/actual が作成される
通常 forward output のみ保存される
```

Spectrum ONの場合:

```text
Dump baseline final output ON でも何も保存されない
生成は正常終了する
```

### 14.4 block output dump

条件:

```text
Dump block output ON
TeaCache OFF
Spectrum OFF
Sparse OFF
Attention override OFF
```

期待:

```text
block_output/block_xx が保存される
block_index が stats.parquet に記録される
```

### 14.5 cross-attention output dump

条件:

```text
Dump cross-attention output ON
```

期待:

```text
cross_attention_output/block_xx が保存される
attn_type = cross が記録される
self-attention は保存されない
```

### 14.6 MLP output dump

条件:

```text
Dump MLP output ON
```

期待:

```text
MLP module が見つかる場合: mlp_output/block_xx が保存される
MLP module が見つからない場合: warningのみ、生成は継続
```

---

## 15. 実装時の禁止事項

- tensor dump失敗で生成処理を停止しない。
- GPU tensor を参照保持したままにしない。必ず `detach()` し、保存用にはCPUへ移す。
- `patcher.py` に保存処理の詳細を大量に直書きしない。`tensor_dump.py` に分離する。
- TeaCache / Spectrum の排他仕様を壊さない。
- 既存の旧UI引数互換を壊さない。
- 初期実装で forecast値の保存まで広げすぎない。

---

## 16. 最初のPRでの完了条件

最初のPRでは、以下を満たせば完了とする。

```text
1. Debug log mode に6つのdumpチェックボックスが追加されている
2. STATEに6つのdumpフラグが追加されている
3. logs/YYYY-MM-DD/run_xxx/ が自動作成される
4. meta.json が保存される
5. stats.parquet が保存される
6. tensors.zarr が保存される
7. TeaCache residual が保存できる
8. Spectrum final output が保存できる
9. block output が保存できる
10. cross-attention output が保存できる
11. MLP output は、可能なら保存。不可能なら安全にwarning fallbackする
12. いずれのdump失敗でも生成は落ちない
```

MLP output がBlock内部構造の都合で初回PRに間に合わない場合は、以下の条件で許容する。

```text
- UIとSTATEにはフラグを用意する
- ONにしても生成は落ちない
- warningで mlp_dump_unavailable を出す
- READMEまたはログで未対応理由を示す
```

---

## 17. 後続タスク

初回実装後、以下を別タスクとして検討する。

```text
- dump対象block範囲のUI追加
- 保存dtypeのUI追加
- statsのみ保存モード
- Spectrum forecast outputの保存
- TeaCache skip時の reused residual record 保存
- Zarr圧縮設定の調整
- ParquetではなくJSONLへのfallback
- オフライン解析スクリプトの追加
```

---

## 18. まとめ

この改修は現実的である。

特に以下2つは、既存実装の差し込み位置が明確であり、優先して実装できる。

```text
- TeaCache residual
- Spectrum final output
```

次に、既存の `Block.forward` / `SelfCrossAttention.compute_attention` patch機構を流用して、以下を実装する。

```text
- block output
- cross-attention output
```

最後に、Anima Block内部のMLP submodule名を確認したうえで、以下を実装する。

```text
- MLP output
```

保存形式は、実験計画書通り以下を採用する。

```text
生tensor:
  Zarr

軽量統計:
  Parquet

実験条件:
  meta.json
```
