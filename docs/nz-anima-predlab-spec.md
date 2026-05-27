# Nz-Anima-PredLab 仕様書 v0.1.1

## 1. 目的

`Nz-Anima-PredLab` は、Forge Neo 上で Anima / Cosmos-Predict2 系 T2I モデルの推論パイプラインを観測し、将来的な高速化実験を安全に切り替えられるようにする拡張機能である。

初期実装では高速化 patch を行わず、Forge Neo の既存実装を壊さない診断・計測機能を優先する。高速化実験を追加する場合も、対象モデル、対象モード、復元処理、fallback 条件を明示して実装する。

関連文書:

- `Nz-Anima-PredLab 要件定義 v0.1.1.txt`
- `docs/forge-neo-research.md`

参考にした Forge Neo 拡張:

- `abzaloff/sd-dynamic-prompts`
- `eduardoabreu81/sd-webui-tagcomplete-neo`

## 2. 対象範囲

対象:

- StabilityMatrix 版 Forge Neo / SD WebUI Forge Neo
- Windows 11
- NVIDIA GPU
- Anima / Cosmos-Predict2 系 T2I モデル
- txt2img
- ER SDE / Euler a を主な確認対象 sampler とする

初期対象外:

- img2img
- Hires.fix
- ControlNet
- IP-Adapter
- 参照画像系拡張
- 複数 GPU
- ComfyUI
- Forge Neo 本体の直接改造

## 3. 基本方針

- 拡張は `extensions/Nz-Anima-PredLab/` として実装する。
- Forge Neo 本体のファイルは変更しない。
- `Off` の場合は通常推論に影響を与えない。
- 診断モードでは画像生成結果を意図的に変更しない。
- 高速化実験モードでは推論パイプラインを変更してよいが、画像が生成されること、かつ baseline から視覚的に大きく逸脱しないことを必須条件とする。
- patch を行う場合は、元関数を保存し、OFF / unload / 例外時に復元できるようにする。
- 拡張 import 時には重い処理をしない。torch / Forge backend の深い参照、モデル検査、GPU 処理は callback または生成時へ遅延する。
- 設定キー、API path、ログ prefix、HTML element id はすべて `nzap` / `Nz-Anima-PredLab` 名前空間に閉じる。
- 外部ネットワークアクセスは行わない。将来必要になった場合も明示的な opt-in 設定を必須とする。

## 4. 参考拡張から採用する作法

### 4.1 scripts entrypoint

`scripts/` 直下のファイルは Forge Neo が拡張を発見するための入口に留める。

想定:

```python
from nz_anima_predlab.script import Script

__all__ = ["Script"]
```

本体実装、callback 登録、状態管理は `nz_anima_predlab/` package 側へ置く。

### 4.2 起動と reload

- callback 登録は多重登録を避ける。
- WebUI の reload / txt2img と img2img の二重ロードで同じ callback が重複しても、診断ログや patch が二重実行されないようにする。
- optional import は `try/except` で隔離し、失敗時は拡張全体を落とさず unsupported / degraded 状態へ移行する。
- 起動時に存在しない runtime file / cache file があっても UI が壊れないよう、必要な空ファイルまたは空 state を初期化する。

### 4.3 UI / settings

- Settings タブには `("nz_anima_predlab", "Nz-Anima-PredLab")` section を作る。
- `shared.opts.add_option()` と `shared.OptionInfo` を使う。
- UI 内の element id は `nzap-*` で統一する。
- 生成タブの UI は `scripts.AlwaysVisible` の折りたたみ panel とし、初期状態では邪魔にならないよう閉じる。
- 生成ごとに変える項目は AlwaysVisible UI、永続項目は Settings タブへ置く。

### 4.4 ドキュメント

- README には Forge Neo 専用であること、想定 Python / Gradio / Forge Neo 系統、非対応 UI、インストール手順、トラブルシュートを明記する。
- `CHANGELOG.md` を追加し、調査版、診断版、実験版の変更点を追えるようにする。
- 先行実装や Forge Neo 本体の調査に基づく仕様変更は `docs/` に残す。

## 5. ディレクトリ構成

想定構成:

```text
Nz-Anima-PredLab/
├── scripts/
│   └── nz_anima_predlab.py
├── nz_anima_predlab/
│   ├── __init__.py
│   ├── script.py
│   ├── callbacks.py
│   ├── settings.py
│   ├── state.py
│   ├── logging.py
│   ├── model_detect.py
│   ├── diagnostics.py
│   ├── forge_introspection.py
│   ├── timing.py
│   ├── patcher.py
│   ├── attention.py
│   └── lowbit.py
├── docs/
│   ├── forge-neo-research.md
│   └── nz-anima-predlab-spec.md
├── README.md
├── CHANGELOG.md
└── LICENSE
```

`AGENTS.md` は Codex 向けの開発者運用メモとして、必要になった時点で別途作成する。拡張機能の実行に必須の構成物とはしない。

## 6. Forge Neo 連携

使用する Forge Neo API / hook:

- `modules.script_callbacks.on_ui_settings`
- `modules.script_callbacks.on_model_loaded`
- `modules.script_callbacks.on_cfg_denoiser`
- `modules.script_callbacks.on_cfg_after_cfg`
- `modules.script_callbacks.on_script_unloaded`
- `modules.scripts.Script`
- `modules.scripts.AlwaysVisible`

初期実装では、settings は `on_ui_settings` で追加する。生成ごとの状態初期化が必要な場合は、AlwaysVisible script の `process_before_every_sampling()` を使う。

callback 登録:

- `callbacks.py` に `register_callbacks()` を置く。
- `register_callbacks()` は idempotent にする。
- `on_script_unloaded` では patch 解除と runtime state の掃除を行う。

将来 UI から詳細 status を取得する必要が出た場合のみ、FastAPI endpoint を追加する。追加する場合の path は `/nzapapi/v1/...` とし、初期診断版では必須にしない。

## 7. 設定項目

すべての永続設定キーは `nzap_*` prefix を使う。

### 7.1 Enable

設定名:

```text
Enable Nz-Anima-PredLab
```

永続 key:

```text
nzap_enable
```

型:

- checkbox / bool

既定値:

- `False`

動作:

- `False` の場合、すべての診断、patch、実験処理を無効にする。
- ただし、拡張ロード時の最小限の初期化と settings 登録は行ってよい。

### 7.2 Debug log mode

設定名:

```text
Debug log mode
```

永続 key:

```text
nzap_mode
```

初期選択肢:

- `Off`
- `Diagnose only`
- `Identity patch test`

将来追加する実験モード:

- `Trace attention`
- `Trace cond/uncond`
- `Trace low-bit / compile`
- `Experimental 2D sparse attention`
- `Compile / low-bit experiment`
- `Fast attention kernel`
- `Cond/uncond optimization`

動作:

- `Off`: 明示的に無効。`Enable` が true でも処理しない。
- `Diagnose only`: 基本情報、timing、attention、cond/uncond、low-bit / compile 関連情報を一括出力する。
- `Identity patch test`: `backend.nn.anima.Block.forward` を Nz-Anima-PredLab の wrapper 経由に切り替え、元の Forge Neo 実装をそのまま呼び戻す。画像内容を変えず、推論パイプラインの一部を拡張側で捕捉できるかを実機検証する。

### 7.2.1 高速化実験 UI 方針

高速化実験は、debug log mode dropdown だけで多数の専用 mode を増やすのではなく、AlwaysVisible panel 内にカテゴリ別の操作群として配置する。

基本原則:

- UI は top-level の `Nz-Anima-PredLab` Accordion を1つだけ持つ。その配下に `Attention` / `2D Sparse` / `Cond / Uncond` / `Low-bit / Compile` / `Diagnostics` のサブ Accordion を置く。
- 他拡張と同じ階層に Nz-Anima-PredLab 用の top-level Accordion を複数作らない。
- すべての項目の初期状態は Forge Neo 本体の挙動と一致させる。
- Forge Neo 本体に既に存在する選択肢は、Nz-Anima-PredLab 側でも本体の現在値を初期値として表示する。
- Forge Neo 本体に存在しない実験機能は、必ず `Enable ...` checkbox を持つ。
- 実験機能の `Enable` は初期値 `False` とする。
- すべての experimental checkbox が `False` で、既存機能の値が Forge current/default のままなら、推論結果と推論経路は Forge Neo baseline と同等でなければならない。
- UI の選択値は生成開始時に snapshot し、生成中に UI を変更しても進行中の batch へは反映しない。
- 画像生成に影響する実験が有効な場合は、生成ログへ設定 snapshot を出す。

#### Attention kernel controls

目的:

- Forge Neo が Anima で使っている attention backend を確認し、既存 backend 間の差し替えによる速度差を比較する。

UI:

| Control | UI type | Default | Notes |
| --- | --- | --- | --- |
| `Attention backend` | radio / dropdown | Forge current backend | StabilityMatrix版 Forge Neo 実測では `attention_sage`。起動ログでも `Using SageAttention 2` が確認されている。 |
| `Attention target` | radio | `self + cross` | Forge baseline と一致。実験時のみ `self only` / `cross only` を選べる。 |
| `Attention block range` | range slider | `0..27` | Forge baseline は全 block 同一 backend。range を絞る場合は実験扱い。 |

候補値:

- `attention_sage`
- `attention_flash`
- `attention_xformers`
- `attention_pytorch`
- `Forge current/default`

`Forge current/default` は実行環境で検出された本体設定をそのまま使う値である。対象環境では現時点で `attention_sage` と解釈する。

`Attention backend` が `Forge current/default` 以外の場合は、`Debug log mode=Off` でも attention kernel 実験として有効になる。実験時は `attention_kernel_call` と `attention_kernel_summary` をログに出す。ログには `requested_backend`、Forge 内部で観測できた `actual_backend`、`internal_fallback`、`actual_backends` を含め、指定 backend 関数が内部で `pytorch_sdpa` に fallback していないかを確認できるようにする。

実機検証では `attention_sage` は `actual_backends=sage:1792`、`internal_fallbacks=0` で最速だった。`attention_flash` は `actual_backends=flash:1792` で動作したが、総生成時間は `attention_sage` より長かった。`attention_xformers` は対象環境で `xformers` 実体が import されておらず、内部で `pytorch_sdpa_fallback` へ落ち、cross attention で shape mismatch を起こしたため unsafe と扱う。`xformers` が実体として利用できない場合は、Nz-Anima-PredLab は `attention_xformers` の実行を避けて元の attention 経路へ戻す。

#### 2D sparse attention / NATTEN controls

目的:

- `backend.nn.anima.Block.forward` で見えている `T/H/W` 情報を使い、self-attention を局所 attention に置換して高速化を狙う。

UI:

| Control | UI type | Default | Notes |
| --- | --- | --- | --- |
| `Enable 2D sparse attention` | checkbox | `False` | Forge Neo 本体に存在しない Nz-Anima-PredLab 実験機能。 |
| `Sparse backend` | radio | `NATTEN (optional)` | `NATTEN` が利用可能なら既定で使う。利用不可の場合は選択不可または degraded 表示にし、`Torch prototype` を検証用 fallback として選べるようにする。 |
| `Sparse target` | radio | `self attention only` | cross-attention は text/context sequence を使うため初期実験では変更しない。 |
| `Block start` | slider | `14` | 28 blocks の後半から適用する初期 preset。ユーザーが `0..27` の範囲で調整できる。Enable off では無効。 |
| `Block end` | slider | `27` | 初期実験 preset。ユーザーが `0..27` の範囲で調整できる。Enable off では無効。 |
| `Step start` | slider | `0` | 初期値は全 step 対象。 |
| `Step end` | slider | `last step` | 生成 steps に合わせて解釈する。 |
| `Local attention window` | slider | `15` | 各 latent token が参照する局所近傍の幅。奇数値を基本とする。画像品質と速度の主要 tradeoff。 |
| `Dilation` | slider | `1` | 初期値は通常近傍。 |
| `Full attention interval` | slider / number | `0` | `0` は full attention 挿入なし。 |

注意:

- `Enable 2D sparse attention=False` が Forge baseline。
- `Block start=14` / `Block end=27` は固定仕様ではなく初期値である。後半 block 限定は、全 block 適用より破綻リスクを抑えた初期 preset として採用する。
- `Local attention window` は full attention の代わりに各 token が見る近傍範囲である。値が大きいほど full attention に近く安全寄り、小さいほど高速化余地が増えるが破綻リスクも上がる。
- `Sparse backend=NATTEN` は optional dependency とする。NATTEN が import できない場合でも拡張全体は動作し、NATTEN backend だけ degraded / unavailable にする。
- `Torch prototype` は高速化本命ではなく、NATTEN 由来の問題か sparse algorithm 自体の問題かを切り分けるための検証 backend とする。
- 最初の algorithmic patch は self-attention のみを対象にする。
- `Block.forward` は `x_B_T_H_W_D` を受け取るため、H/W/T を保持できる最も実用的な patch point である。

#### Cond/uncond optimization controls

目的:

- Forge Neo 既存の CFG batching を崩さず、未検証条件で追加の高速化余地があるか調べる。

UI:

| Control | UI type | Default | Notes |
| --- | --- | --- | --- |
| `Enable cond/uncond optimization` | checkbox | `False` | 通常 CFG>1 は既に同一 model call に batch されているため、低優先度の実験機能。 |
| `Cond/uncond mode` | radio | `Forge default` | baseline は Forge Neo の `calc_cond_uncond_batch()` に従う。 |
| `Skip uncond when CFG=1` | checkbox | `False` | Forge 側挙動を確認してから有効化する。 |
| `Guidance step schedule` | checkbox | `False` | 一部 step だけ CFG 処理を変える実験。 |
| `Guidance interval` | slider / number | `1` | schedule 有効時のみ使用。 |

既知事実:

- 2026-05-26 の実測では CFG>1 の通常生成で `cond_or_uncond=[1, 0]`、`input_shape=2x16x1x192x192` が得られ、cond/uncond は同一 model call に batch されていた。
- このため cond/uncond 最適化は高速化の本命ではないが、検証項目からは外さない。

#### Low-bit / compile controls

目的:

- Forge Neo の dtype / ops / compile 関連機能を Anima に適用した場合の速度、VRAM、画質を比較する。

UI:

| Control | UI type | Default | Notes |
| --- | --- | --- | --- |
| `Precision / ops mode` | dropdown | Forge current ops | 実測では `ForgeOperations`、storage/computation とも `torch.bfloat16`。 |
| `Enable Nz low-bit experiment` | checkbox | `False` | Forge current から追加で低bit化する場合のみ有効。 |
| `Low-bit target` | checkbox group | none | 候補: attention / MLP / all linear / selected blocks。 |
| `Low-bit format` | radio | 未確定 | 候補: fp8 / int8 / nf4。実装可否確認が必要。 |
| `Enable torch.compile experiment` | checkbox | `False` | Forge current で compile 未使用なら off が baseline。 |
| `Compile target` | radio | `none` | 候補: block / self-attention / MLP / full diffusion model。 |
| `Compile mode` | dropdown | Forge/PyTorch default | 候補: default / reduce-overhead / max-autotune。 |
| `Warmup runs` | slider / number | `1` | compile 初回コストと2回目以降を分けて測る。 |

注意:

- model reload が必要な設定と runtime patch で足りる設定を UI 上で区別する。
- reload 必須項目を変更した場合、生成直前に silent reload しない。UI またはログで「設定変更時にはモデルをリロードしてください」と明示する。
- 初期実装では自動 model reload 機能を実装しない。
- compile は初回生成を遅くする可能性があるため、benchmark では warmup と measured run を分ける。

高速化実験の優先順位:

1. `Experimental 2D sparse attention`
2. `Compile / low-bit experiment`
3. `Fast attention kernel`
4. `Cond/uncond optimization`

理由:

- 実測では Anima T2I latent は `1x16x1x192x192` で、T=1、H/W=192 の5D latentとして扱える。
- 実測では attention backend は `attention_sage` で、Anima attention path も有効だった。
- 実測では CFG>1 の cond/uncond は `cond_or_uncond=[1, 0]`、`input_shape=2x16x1x192x192` として同一 model call に batch されていた。
- そのため、cond/uncond 最適化は高速化の本命ではない。ただし検証項目として維持する。
- CFG=1.0、negative prompt空、batch size > 1、複数prompt、特殊拡張併用時の挙動は未検証として残す。

### 7.3 Logging

設定名:

```text
Print timing log
Verbose diagnose log
```

永続 key:

```text
nzap_print_timing_log
nzap_verbose_diagnose_log
```

動作:

- `Print timing log`: 生成ごとの total sampling time と average step time を出力する。
- `Verbose diagnose log`: model detection evidence、attention backend、cond/uncond 情報、dtype 情報などを追加出力する。

### 7.4 Runtime status

AlwaysVisible UI には、可能なら短い status 表示を置く。

状態:

- `disabled`
- `unsupported`
- `ready`
- `diagnosing`
- `trace`
- `patch-active`
- `error`

この status はログと同じ state object から読む。UI 更新のためだけに推論経路へ余分な処理を入れない。

## 8. 状態管理

拡張は実行時状態をグローバルな小さい state object に保持する。

保持する状態:

- 現在の enabled / mode / logging 設定
- 現在ロードされている model detection result
- 生成開始時刻
- step 開始時刻
- step duration list
- sampling step count
- 最後に出力した診断情報
- patch 適用状態
- patch 元関数の参照
- runtime status
- 直近 error message
- 同一モデルで警告を出したかどうか

状態の初期化タイミング:

- 拡張 import 時
- `on_model_loaded`
- `process_before_every_sampling`
- `on_script_unloaded`

生成ごとの timing state は `process_before_every_sampling()` で初期化する。`on_cfg_denoiser()` で step 開始、`on_cfg_after_cfg()` で step 終了として記録する。

## 9. モデル検出仕様

`model_detect.py` は、ロード済みモデルが Anima / Cosmos-Predict2 系かを判定する。

入力:

- `shared.sd_model`
- `sd_model.sd_checkpoint_info`
- `sd_model.filename`
- `sd_model.model_config`
- `sd_model.forge_objects`

判定 evidence:

- `type(sd_model).__name__`
- `sd_model.model_config.__class__.__name__`
- `getattr(sd_model.model_config, "huggingface_repo", "")`
- `sd_model.filename`
- `sd_model.sd_checkpoint_info.name`
- `sd_model.forge_objects.unet.model.diffusion_model.__class__.__name__`
- `backend.nn.anima.Anima` 由来と判断できる class name

判定結果:

```text
supported: bool
confidence: "strong" | "weak" | "none"
family: "anima" | "cosmos_predict2" | "unknown"
evidence: dict
reason: str
```

強い判定:

- diffusion model class が `Anima`
- model config / repo 名に `Anima` または `Cosmos` / `Predict2` 系の明確な情報がある
- Forge Neo loader が `CosmosTransformer3DModel` を `backend.nn.anima.Anima` としてロードしたと判断できる

弱い判定:

- checkpoint filename や title のみに `anima`, `cosmos`, `predict2` が含まれる

非対応:

- 判定 evidence が不足している
- SDXL / Flux / Z-Image / Wan / Qwen / Chroma など別モデルと判断できる

非対応モデルでは、警告ログのみを出し、処理は変更しない。同一モデル・同一 session では警告を出しすぎないようにする。

## 10. 診断仕様

### 10.1 共通ログ

出力 prefix:

```text
[Nz-Anima-PredLab]
```

必須項目:

- version
- enabled
- mode
- model detection result
- sampler
- scheduler
- resolution
- steps
- CFG scale
- total sampling time
- average step time
- runtime status

可能なら出力:

- peak VRAM
- attention backend
- cond/uncond presence
- cond/uncond batch structure
- model forward count
- model storage dtype
- model computation dtype
- Forge operation family
- VAE decode time
- step timing min / max / p50 / p95

### 10.2 Timing

計測対象:

- sampling pass 全体
- denoiser callback 間の step duration

平均 step time:

```text
avg_step_time = sum(step_durations) / len(step_durations)
```

注意:

- sampler によって denoiser call 数が UI steps と一致しない可能性がある。
- そのためログでは `steps` と `denoiser_calls` を分けて出す。

### 10.3 Attention trace

出力項目:

- `backend.attention.attention_function.__name__`
- SageAttention availability
- FlashAttention availability
- xFormers availability
- PyTorch attention availability
- Anima `SelfCrossAttention` 経路の検出可否

診断目的:

- Forge Neo の高速 attention 設定が Anima に効いているかを確認する。
- 将来の `Fast attention kernel` mode の patch point を確認する。

### 10.4 Cond/uncond trace

出力項目:

- `params.text_uncond is None`
- CFG scale
- `params.sampling_step`
- `params.total_sampling_steps`
- `params.denoiser.step`
- `params.denoiser.total_steps`
- latent / sigma shape
- text cond / uncond type
- `transformer_options.cond_or_uncond` が通常 callback 時点で見えるかどうか
- `transformer_options.cond_indices` が通常 callback 時点で見えるかどうか
- `transformer_options.uncond_indices` が通常 callback 時点で見えるかどうか
- 必要時のみ有効にする診断 wrapper では、`model.apply_model()` 直前の `cond_or_uncond` / `cond_indices` / `uncond_indices`
- cond/uncond が同一 model call に batch されている可能性

診断目的:

- Forge Neo の `calc_cond_uncond_batch()` が通常ケースで十分に働いているかを確認する。
- `CFG=1.0` の uncond 省略が Anima でも働くかを確認する。

実測での注意:

- Forge Neo `neo` の `CFGDenoiserParams` では、sampling step は `sampling_step` / `total_sampling_steps` として渡される。
- `transformer_options.cond_or_uncond`、`cond_indices`、`uncond_indices` は `cfg_denoiser_callback` の後、`backend.sampling.sampling_function.calc_cond_uncond_batch()` 内で作られるため、通常 callback だけでは直接観測できない。
- これらを正確に観測するには、`calc_cond_uncond_batch()` または `model.apply_model()` 直前の軽量診断 wrapper が必要になる。
- 2026-05-26の実測では、CFG>1の通常生成で `cond_or_uncond=[1, 0]`、`input_shape=2x16x1x192x192` が得られ、cond/uncondは同一 model call に batch されていた。
- この結果により、cond/uncond最適化は低優先度とする。ただし検証項目からは外さない。
- 未検証条件として、CFG=1.0、negative prompt空、batch size > 1、複数prompt、特殊拡張併用時のbatch構造を残す。
- `calc_cond_uncond_batch()` の軽量診断 wrapper は必要時のみ使い、標準の `Diagnose only` では自動適用しない。

### 10.5 Low-bit / compile trace

出力項目:

- `backend.args.dynamic_args.ops`
- model `storage_dtype`
- model `computation_dtype`
- unet dtype
- text encoder dtype 可能なら
- command-line flags から推測できる low-bit / attention 関連状態

診断目的:

- Forge Neo の model load 時 operation 選択を確認する。
- runtime patch で低bit化すべきか、model reload が必要かを判断する。

## 11. Runtime artifacts

初期診断版ではファイル出力を必須にしない。

高速化実験版でも、初期実装では JSON / CSV benchmark log を保存しない。実験結果はコンソール summary を主な確認手段にする。

重要な summary:

- `total_sampling_time`
- `avg_step_time`
- `denoiser_calls`
- 実験機能が有効な場合の設定 snapshot

画質比較は初期実装では目視確認とする。baseline / patched pair の自動保存は行わない。

将来ログ、cache、比較結果、profiling 結果をファイル保存する場合:

- `logs/`, `cache/`, `tmp/` のように用途別 directory を分ける。
- 生成物は git 管理しない。
- cache key には Forge Neo version、Nz-Anima-PredLab version、checkpoint hash / filename、mode を含める。
- cache が壊れている場合は破棄して再生成し、生成処理を止めない。
- 個人環境の絶対 path を共有用レポートへそのまま出さない。

## 12. Patch 仕様

初期診断では高速化 patch を行わない。ただし実機検証用に、画像内容を変えない identity patch を許可する。

`Identity patch test` では `backend.nn.anima.Block.forward` を Nz-Anima-PredLab の wrapper に差し替え、wrapper 内で元の `Block.forward` をそのまま呼ぶ。これは高速化ではなく、Forge Neo 本体の推論パイプラインの一部を拡張側から安全に迂回・復帰できるかを確認するための検証である。

検証ログ:

- patch 適用対象と挙動: `target=backend.nn.anima.Block.forward behavior=call_original`
- 各 call の一部: `identity_patch_call=... route=Nz-Anima-PredLab->original_Block.forward`
- 生成後 summary: `identity_patch_summary=calls=... shape_mismatches=... errors=... active=True`

2026-05-26 の StabilityMatrix版 Forge Neo 実機検証では、32 steps / 28 blocks の生成で
`identity_patch_summary=calls=896 num_blocks=28 shape_mismatches=0 errors=0` が得られた。
これは `32 * 28 = 896` と一致し、Anima block-level の推論経路を Nz-Anima-PredLab wrapper
経由に切り替えられることを確認した結果である。

将来 patch を行う場合、すべての patch は `patcher.py` で管理する。

必須 API:

```text
apply_patch(kind, context) -> PatchResult
remove_patch(kind) -> PatchResult
remove_all_patches() -> PatchResult
is_patched(kind) -> bool
```

必須条件:

- 元関数を保存する。
- 多重 patch を防ぐ。
- 例外時は元関数へ fallback する。
- OFF / unsupported model / unload で必ず復元する。
- patch 適用・解除をログに出す。

patch 候補:

- `backend.nn.anima.Block.forward`
- `backend.nn.anima.SelfCrossAttention.compute_attention`
- `backend.nn.anima.SelfCrossAttention.torch_attention_op`
- `backend.attention.attention_function`

2D sparse attention は、flatten 後の generic attention だけでは H/W 情報が失われるため、`Block.forward` または `SelfCrossAttention` 付近で形状情報を扱う方針とする。

patch 優先順位:

1. H/W/T 情報を保持できる `Block.forward` / `SelfCrossAttention` 付近での2D sparse attention実験。
2. Forge Neoの低bit・compile機能をAnimaへ適用するためのmodel load / operation選択調査。
3. attention backend差し替え。実測ではSageAttentionが既に使われているため優先度は中から低。
4. cond/uncond最適化。実測で通常CFG>1は同一forward batch化済みのため優先度は低いが、未検証条件の確認項目として維持する。

## 13. Safety

以下の場合は処理を変更しない:

- `Enable Nz-Anima-PredLab` が false
- `Debug log mode` が `Off` で、すべての実験機能が baseline / disabled
- model detection が unsupported
- txt2img 以外
- img2img / Hires.fix / ControlNet / IP-Adapter / 参照画像系拡張が有効と判断できる
- patch 対象関数が見つからない
- patch 対象関数の signature が想定と異なる

以下の場合は patch を解除して fallback する:

- 例外発生
- 出力 tensor shape が想定と異なる
- NaN / Inf を検出した場合
- ユーザーが `Debug log mode` を Off にし、該当する実験機能も baseline / disabled にした場合
- script unload

例外処理:

- optional dependency / optional Forge API が見つからない場合は degraded status にする。
- callback 内で例外が出た場合は `[Nz-Anima-PredLab]` prefix 付きで要点をログに出し、可能なら以後の patch を無効化する。
- 例外を握りつぶして silent failure にしない。ただし画像生成を止める例外は最小化する。

## 14. 出力例

Diagnose only:

```text
[Nz-Anima-PredLab] version=0.1.1 enabled=True mode=Diagnose only status=diagnosing
[Nz-Anima-PredLab] model_supported=True confidence=strong family=anima
[Nz-Anima-PredLab] sampler=ER SDE scheduler=Normal steps=30 cfg=5.0 resolution=1536x1536
[Nz-Anima-PredLab] denoiser_calls=30 avg_step_time=1.234s total_sampling_time=37.020s
```

Identity patch test:

```text
[Nz-Anima-PredLab] applied identity patch kind=block_forward_identity target=backend.nn.anima.Block.forward behavior=call_original
[Nz-Anima-PredLab] version=0.1.1 enabled=True mode=Identity patch test status=identity-patch
[Nz-Anima-PredLab] identity_patch_call=call=0 block_index=0 input_shape=2x1x96x96x2048 output_shape=2x1x96x96x2048 same_shape=True route=Nz-Anima-PredLab->original_Block.forward
[Nz-Anima-PredLab] identity_patch_summary=calls=896 num_blocks=28 logged_calls=17 shape_mismatches=0 errors=0 active=True target=backend.nn.anima.Block.forward behavior=call_original
```

Trace attention:

```text
[Nz-Anima-PredLab] attention_backend=attention_sage
[Nz-Anima-PredLab] sage_enabled=True flash_enabled=False xformers_enabled=False pytorch_attention_enabled=True
[Nz-Anima-PredLab] anima_attention_path=SelfCrossAttention -> backend.attention.attention_function
```

Trace cond/uncond:

```text
[Nz-Anima-PredLab] cfg=5.0 uncond_present=True denoiser_step=4/30
[Nz-Anima-PredLab] cond_or_uncond=[0, 1] cond_indices=[0] uncond_indices=[1]
```

Trace low-bit / compile:

```text
[Nz-Anima-PredLab] forge_ops=ForgeOperationsInt8 storage_dtype=torch.int8 computation_dtype=torch.bfloat16
```

## 15. Packaging / compatibility

対象:

- Forge Neo 専用 extension として扱う。
- A1111 本家、Forge classic、ComfyUI との互換性は保証しない。

README に明記する項目:

- Forge Neo 専用であること。
- Python / Gradio / Forge Neo の想定系統。
- インストール手順。
- 初期版は高速化 patch ではなく診断・計測が目的であること。
- トラブルシュート: 拡張が表示されない、unsupported model になる、ログが出ない、生成が遅くなった場合。

## 16. Acceptance Criteria

初期診断版の完了条件:

- Forge Neo 拡張として読み込まれる。
- settings に `Enable Nz-Anima-PredLab` が表示される。
- settings key が `nzap_*` に統一されている。
- mode を選択できる。
- `scripts/nz_anima_predlab.py` が薄い entrypoint になっている。
- callback 登録が多重実行されない。
- import 時にモデル検査や GPU 処理を行わない。
- unsupported model で処理変更が起きない。
- supported model で model detection evidence を出力できる。
- `Diagnose only` で total sampling time と average step time を出力できる。
- `Diagnose only` で attention backend、uncond presence、CFG 関連情報、dtype / Forge ops 関連情報を一括出力できる。
- `Identity patch test` で `backend.nn.anima.Block.forward` を wrapper 経由に切り替え、元の `Block.forward` を呼び戻せる。
- `Identity patch test` の summary で `steps * num_blocks` と一致する call count、`shape_mismatches=0`、`errors=0` を確認できる。
- `Off` ではログ出力と処理変更が止まる。
- 例外時に WebUI 起動と画像生成を可能な限り止めず、status を `error` または degraded 状態へ移せる。

高速化実験版の完了条件:

- patch 対象が明示されている。
- patch 適用前後で復元できる。
- baseline と同一条件で比較できる。
- すべての実験項目を off にした場合、Forge Neo baseline と同等の挙動になる。
- Forge Neo 本体にない機能は `Enable ...` checkbox が off の状態を default とする。
- Forge Neo 本体にある機能は、本体の current/default 値を UI 初期値として表示する。
- 画像が生成される。
- 品質劣化が視覚的に許容範囲内である。
- 1step 平均時間が 5% 以上短縮する。

## 17. Open Issues

- Forge Neo の `torch.compile` 実装箇所を特定する。
- `on_cfg_denoiser()` の呼び出し回数が target sampler ごとに UI steps と一致するか確認する。
- Anima / Cosmos-Predict2 派生 checkpoint の検出条件を実機で確認する。
- 2D sparse attention 実験で `Block.forward` から self-attention 実装へ H/W/T 情報を渡す方法を決める。
- low-bit / compile 設定ごとに、runtime patch で足りるか model reload が必要かを判定して UI に表示する。
- NATTEN が対象環境で import / 実行できるか確認する。

## 18. 確定した仕様判断

2026-05-26 時点で確定した実験 UI 方針:

- 2D sparse attention の初期 preset は後半 block 限定とする。`Block start=14`、`Block end=27` を default にし、ユーザーは slider で `0..27` の範囲を自由に変更できる。
- `Window size` という名称は使わず、UI 表示名は `Local attention window` とする。
- `Local attention window` は slider で指定する。初期値は `15` とする。値は基本的に奇数のみを扱う。
- `Sparse backend` の default は `NATTEN (optional)` とする。NATTEN が利用できない場合は degraded / unavailable とし、`Torch prototype` を検証用 fallback として選べるようにする。
- `Torch prototype` は高速化本命ではなく、NATTEN なしでも破綻するかを切り分けるための backend とする。
- 実験結果はコンソール summary のみで確認する。JSON / CSV 保存は初期実装では行わない。
- 画質比較は目視確認のみとする。baseline / patched pair の自動保存は初期実装では行わない。
- UI は top-level の `Nz-Anima-PredLab` Accordion 1つの配下にカテゴリ別サブ Accordion を置く。他拡張と同じ階層に Nz-Anima-PredLab 用 Accordion を複数作らない。
- low-bit / compile で model reload が必要な設定がある場合、自動 reload は行わない。UI またはログで「設定変更時にはモデルをリロードしてください」と知らせる。
