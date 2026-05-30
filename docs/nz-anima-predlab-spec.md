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
- `Debug log mode` が空欄、かつすべての個別 experimental control が baseline / disabled の場合は通常推論に影響を与えない。空欄でも個別 experimental control が有効な場合は、該当実験を明示 opt-in として実行してよい。
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

- ``
- `Diagnose only`
- `Identity Patch test`

現行実装では、実験機能ごとに mode を増やさず、AlwaysVisible panel 内の個別 control で切り替える。`Trace attention` / `Trace cond/uncond` / `Trace low-bit / compile` / `Experimental 2D sparse attention` / `Compile / low-bit experiment` / `Fast attention kernel` / `Cond/uncond optimization` は旧案または将来検討名であり、`nz_anima_predlab.state.MODES` には含めない。

動作:

- 空欄: debug mode としては明示的に無効。ただし、`Enable attention backend override=True` かつ `Attention backend != Forge current/default`、`Enable 2D sparse attention=True` などの個別 experimental control が有効な場合は、debug mode が空欄のままでも実験 patch は動作する。
- `Diagnose only`: 基本情報、timing、attention、low-bit / compile 関連情報を一括出力する。`Verbose diagnose log=True` かつ他の experimental patch が無効な場合は、`block_structure_trace` を適用して Anima block / qkv 構造も観測する。
- `Identity Patch test`: `backend.nn.anima.Block.forward` を Nz-Anima-PredLab の wrapper 経由に切り替え、元の Forge Neo 実装をそのまま呼び戻す。画像内容を変えず、推論パイプラインの一部を拡張側で捕捉できるかを実機検証する。

### 7.2.1 高速化実験 UI 方針

高速化実験は、debug log mode dropdown だけで多数の専用 mode を増やすのではなく、AlwaysVisible panel 内にカテゴリ別の操作群として配置する。

基本原則:

- UI は top-level の `Nz-Anima-PredLab` Accordion を1つだけ持つ。現行実装ではその配下に `Debug log mode` / `Attention` / `TeaCache` / `Spectrum` / `2D Sparse` / `Cond / Uncond` / `Low-bit / Compile` のサブ Accordion を置く。`Debug log mode` 内の `Enable debug log mode` は debug dropdown の有効化だけを制御し、親の `Enable Nz-Anima-PredLab` とは別物として扱う。
- 他拡張と同じ階層に Nz-Anima-PredLab 用の top-level Accordion を複数作らない。
- すべての項目の初期状態は Forge Neo 本体の挙動と一致させる。
- Forge Neo 本体に既に存在する選択肢は、Nz-Anima-PredLab 側でも本体の現在値を初期値として表示する。
- Forge Neo 本体に存在しない実験機能は、必ず `Enable ...` checkbox を持つ。
- 実験機能の `Enable` は初期値 `False` とする。
- すべての experimental checkbox が `False` で、`Enable attention backend override=False` または `Attention backend=Forge current/default` のままなら、推論結果と推論経路は Forge Neo baseline と同等でなければならない。
- サブ Accordion 内の実験用 `Enable ...` checkbox を `True` にした場合、親の `Enable Nz-Anima-PredLab` が `False` なら UI callback で `True` にする。
- 親の `Enable Nz-Anima-PredLab` を `False` にした場合、サブ Accordion 内の checkbox は変更しない。ユーザーが一時的に親だけを off にしても、子項目の調整値を保持する。
- UI の選択値は生成開始時に snapshot し、生成中に UI を変更しても進行中の batch へは反映しない。
- 画像生成に影響する実験が有効な場合は、生成ログへ設定 snapshot を出す。

機能追加時の相互排他ルール:

- 新しい機能を実装するときは、それが既存の機能と相互排他かどうかを必ず検討する。
- 同時に `Enable` にした場合に server crash、CUDA error、tensor shape corruption、unrecoverable patch conflict などの深刻な問題を起こす可能性がある組み合わせは、UI または生成開始時の設定処理で同時に `Enable` にできないようにする。
- 生成画像が大きく崩れる、品質劣化が増える、速度が落ちる、結果が比較しにくくなる程度の「非推奨」組み合わせは、原理的に実行可能であれば同時に `Enable` にできてもよい。その場合はログやUI表示で実験的・非推奨であることが分かるようにする。
- `TeaCache` と `Spectrum` はどちらも step 単位で denoiser 計算を省略する実験であり、同時有効化は結果の原因切り分けを困難にするため UI 上で相互排他にする。`Enable TeaCache experiment=True` にしたら `Enable Spectrum experiment=False` にし、`Enable Spectrum experiment=True` にしたら `Enable TeaCache experiment=False` にする。

#### Attention kernel controls

目的:

- Forge Neo が Anima で使っている attention backend を確認し、既存 backend 間の差し替えによる速度差を比較する。

UI:

| Control | UI type | Default | Notes |
| --- | --- | --- | --- |
| `Enable attention backend override` | checkbox | `False` | Attention kernel patch を有効化する。親の `Enable Nz-Anima-PredLab` が `False` なら UI callback で `True` にする。 |
| `Attention backend` | dropdown | `Forge current/default` | StabilityMatrix版 Forge Neo 実測では本体の現在値が `attention_sage`。`Enable attention backend override=True` かつ `Forge current/default` 以外を選ぶと attention kernel patch が有効になる。 |
| `Attention target` | radio | `self + cross` | Forge baseline と一致。実験時のみ `self only` / `cross only` を選べる。 |
| `Attention block start` | slider | `0` | 現行UIは range slider ではなく start/end の2本の slider。 |
| `Attention block end` | slider | `27` | Forge baseline は全 block 同一 backend。range を絞る場合は実験扱い。 |

候補値:

- `attention_sage`
- `attention_flash`
- `attention_xformers`
- `attention_pytorch`
- `Forge current/default`

`Forge current/default` は実行環境で検出された本体設定をそのまま使う値である。対象環境では現時点で `attention_sage` と解釈する。

`Enable attention backend override=True` かつ `Attention backend` が `Forge current/default` 以外の場合は、`Debug log mode` が空欄でも attention kernel 実験として有効になる。実験時は `attention_kernel_call` と `attention_kernel_summary` をログに出す。ログには `requested_backend`、Forge 内部で観測できた `actual_backend`、`internal_fallback`、`actual_backends` を含め、指定 backend 関数が内部で `pytorch_sdpa` に fallback していないかを確認できるようにする。

実機検証では `attention_sage` は `actual_backends=sage:1792`、`internal_fallbacks=0` で最速だった。`attention_flash` は `actual_backends=flash:1792` で動作したが、総生成時間は `attention_sage` より長かった。`attention_xformers` は対象環境で `xformers` 実体が import されておらず、内部で `pytorch_sdpa_fallback` へ落ち、cross attention で shape mismatch を起こしたため unsafe と扱う。`xformers` が実体として利用できない場合は、Nz-Anima-PredLab は `attention_xformers` の実行を避けて元の attention 経路へ戻す。

#### Tensor dump controls

目的:

- Anima feature forecasting 実験のため、推論中の中間 tensor と軽量統計を研究用ログとして保存する。
- この機能は高速化実装ではなく、後段のオフライン解析用データ収集基盤である。

UI:

| Control | UI type | Default | Notes |
| --- | --- | --- | --- |
| `Dump TeaCache residual` | checkbox | `False` | `TeaCache=True` の full calculation 時のみ `block_stack_output - block_stack_input` を保存する。 |
| `Dump block output` | checkbox | `False` | 全 block の軽量統計を保存し、生 tensor は代表 block `0,7,14,21,27` に限定する。 |
| `Dump cross-attention output` | checkbox | `False` | cross-attention branch output を保存する。self-attention は対象外。 |
| `Dump MLP output` | checkbox | `False` | MLP module が特定できる場合のみ保存する。見つからない場合は warning fallback とする。 |
| `Dump Spectrum final output` | checkbox | `False` | Spectrum ON の actual forward output を保存する。forecast output は初期対象外。 |
| `Dump baseline final output` | checkbox | `False` | Spectrum OFF の通常 forward output を `baseline_final_output` として保存する。 |

動作:

- tensor dump は `Enable Nz-Anima-PredLab=True` かつ `Enable debug log mode=True` の場合だけ有効になる。
- 保存先は画像出力ディレクトリから `Images/logs/YYYY-MM-DD/run_.../` を推定し、推定できない場合は `logs/YYYY-MM-DD/run_.../` へ fallback する。
- 保存構成は `meta.json`、`stats.parquet`、`tensors.zarr/` とする。
- `zarr` / `pandas` / `pyarrow` が import できない場合はインストールを試みる。失敗した場合は `tensor_dump_unavailable` warning を出し、生成は継続する。
- block / cross-attention / MLP dump は `TeaCache=False`、`Spectrum=False`、`Enable 2D sparse attention=False`、`Enable attention backend override=False` の場合だけ有効にする。
- 各 record には `logical_step_index`、`local_call_index`、`block_call_index`、`block_index`、`timestep_value`、`teacache_model_call`、`spectrum_cnt` を可能な範囲で保存する。

#### TeaCache / residual cache controls

目的:

- `Anima` の transformer block 列を一部 sampling step で skip し、前回 full calculation 時に保存した residual を再利用することで、30〜35 step の通常生成を維持したまま 1生成あたりの推論時間短縮を狙う。
- attention kernel そのものを高速化する機能ではなく、DiT block stack の計算頻度を減らす cache management 実験として扱う。

参考実装:

- `daraskme/comfy_anima_tea_cache`
- `welltop-cn/ComfyUI-TeaCache`
- `ali-vilab/TeaCache`

UI:

| Control | UI type | Default | Notes |
| --- | --- | --- | --- |
| `Enable TeaCache experiment` | checkbox | `False` | Forge Neo 本体に存在しない Nz-Anima-PredLab 実験機能。 |
| `TeaCache preset` | dropdown / segmented radio | `Balanced` | 候補: `Safe` / `Balanced` / `Aggressive` / `Custom`。preset は下記パラメータの初期値をまとめて選ぶ。 |
| `Rel L1 threshold` | slider / number | `0.070` | 主要 tradeoff。Anima 30〜32 step では `0.06..0.07` を安全寄りの初期検証範囲とし、`0.08` 以上は品質劣化リスクが高い実験域として扱う。破壊的な動作確認用に UI 上限は `1.0` とする。 |
| `Start percent` | slider | `0.05` | TeaCache 判定を開始する sampling 進行率。32 step ではおおむね step 1〜2 以降に相当する。 |
| `End percent` | slider | `0.95` | TeaCache 判定を終了する sampling 進行率。終盤の細部を守るため、最後の数 step は full calculation に戻せるようにする。 |
| `Cache device` | radio | `cuda` | 候補: `cuda` / `cpu`。`cuda` は高速寄りで VRAM を少し使う。`cpu` はVRAM節約用だが転送で遅くなる可能性がある。 |
| `Modulated source` | dropdown | `first_block_shift` | 候補: `first_block_shift` / `timestep_embedding`。Anima向け既存検証では `first_block_shift` が安定寄り。 |
| `Coefficient profile` | dropdown | `Anima 2B 30step first_block_shift` | 使用中の多項式係数を明示する。係数未校正の `Identity / uncalibrated` は診断用途のみ。 |
| `Max skip streak` | slider / number | `0` | `0` は制限なし。初期実装では安全装置として `2` または `3` を選べるようにしてよい。 |
| `Force full calc interval` | slider / number | `0` | `0` は無効。`N > 0` の場合、N step ごとに必ず full calculation を行う。 |
| `Dry-run TeaCache decisions only` | checkbox | `False` | 実際には block skip せず、skip/run 判定だけログに出す。初期実装・閾値調整・安全確認用。 |
| `Verbose TeaCache trace` | checkbox | `False` | step ごとの rel_l1 / accumulated / should_calc / skip/run を出す。通常は summary のみ。 |

Preset:

| Preset | Rel L1 threshold | Start percent | End percent | Notes |
| --- | ---: | ---: | ---: | --- |
| `Safe` | `0.060` | `0.05` | `0.95` | 品質優先。速度向上は小さめ。 |
| `Balanced` | `0.070` | `0.05` | `0.95` | Anima 30 step の既存検証で LPIPS 0.05 付近の境界。初期 default。 |
| `Aggressive` | `0.080` | `0.05` | `0.95` | 速度優先。品質劣化が急増し得るため明示的な実験扱い。 |
| `Custom` | UI値を保持 | UI値を保持 | UI値を保持 | 個別調整用。 |

UI behavior:

- `TeaCache preset` で `Safe` / `Balanced` / `Aggressive` を選ぶと、`Rel L1 threshold` / `Start progress` / `End progress` はその preset の値へ自動更新する。
- `Rel L1 threshold` / `Start progress` / `End progress` のいずれかを手動で変更した場合、`TeaCache preset` は自動的に `Custom` へ切り替わる。
- `Custom` を選んだ場合は既存の slider 値を維持する。

必須動作:

- 生成または batch の最初の model call / sampling step では、利用可能な `previous_residual` が存在しないため、必ず full calculation を行う。
- `previous_residual` が未初期化、shape mismatch、dtype/device mismatch、NaN/Inf 検出、cond/uncond state 不整合のいずれかを検出した場合も full calculation へ戻す。
- `Start percent` の条件を満たしていても、初回 step だけは TeaCache skip を許可しない。
- `Dry-run TeaCache decisions only=True` の場合は、skip 条件を満たしても必ず full calculation を行い、判定結果だけをログに残す。
- CFG の cond/uncond は原則として別 state に分ける。`cond_or_uncond` が取得できない場合は TeaCache を無効化または full calculation 固定にする。
- TeaCache state は generation 開始時、model reload、unsupported model、OFF、unload で破棄する。

初期実装で使う係数:

```text
profile=Anima 2B 30step first_block_shift
coefficients=[5954.035087553969, -2410.0426539290293, 349.24023850217395, -17.264742642375417, 0.31229336331906893]
source=first_block_shift
steps=30
rmse=0.03206983196972507
```

32 step で使う場合も初期実験では同 profile を使ってよいが、ログには `profile_steps=30 runtime_steps=32` を出して、厳密には32 step用に再校正されていないことを分かるようにする。

#### Spectrum / spectral feature forecasting controls

目的:

- Anima base v1.0 の通常 step 数を維持したまま、Spectrum による denoiser 出力予測で推論時間短縮を狙う。
- SDXL 向けの一般設定ではなく、Anima base v1.0 で出力変化を抑えることを優先した実験機能として扱う。
- Spectrum は training-free の spectral feature forecasting であり、Chebyshev 多項式と ridge regression によって過去 step の特徴推移から将来 step を予測する。

参考実装:

- `hanjq17/Spectrum`
- `AdamNizol/ComfyUI-Anima-Enhancer`
- Forge Neo built-in `sd_forge_spectrum`

適用方針:

- 初期ターゲットは Anima base v1.0 とする。
- ただし、Anima base v1.0 以外であることだけを理由に Spectrum を禁止したり、警告ログを出したりしない。
- Anima 向けに調整した preset を提供するが、他モデルで動かすかどうかはユーザーの実験範囲とする。
- patch point、tensor shape、dtype/device、既存 wrapper などの実行条件が合わない場合のみ、通常の安全 fallback として Forge baseline へ戻す。

UI:

| Control | UI type | Default | Notes |
| --- | --- | --- | --- |
| `Enable Spectrum experiment` | checkbox | `False` | Forge Neo 本体の `Spectrum Integrated` とは別の、Nz-Anima-PredLab 管理下の実験機能。 |
| `Spectrum preset` | dropdown / segmented radio | `Balanced` | 候補: `Safe` / `Balanced` / `Aggressive` / `Custom`。preset は下記パラメータの初期値をまとめて選ぶ。 |
| `Prediction weighting` (`w`) | slider / number | `0.20` | Chebyshev 予測と短期 Taylor 補間の blend。Anima 向け初期値は ComfyUI-Anima-Enhancer の推奨範囲 `0.2..0.3` を優先する。 |
| `Polynomial degree` (`m`) | slider / number | `16` | Chebyshev basis の次数。Anima 向け先行事例の `8..16` を重視し、UI 範囲は `1..32` とする。 |
| `Ridge lambda` | slider / number | `0.50` | ridge regression の正則化。初期 preset では `0.50` に固定する。 |
| `Warmup steps` | slider / number | `6` | Spectrum 予測を開始する前に full denoiser を実行する step 数。 |
| `Window size` | slider / number | `2` | actual forward と forecast の間隔。初期 preset では `2` に固定する。 |
| `Flex window` | slider / number | `0.00` | actual forward 後に window を広げる量。Anima base v1.0 では出力変化抑制を優先し、初期 preset では `0.00` に固定する。 |
| `Stop progress` | slider | `0.80` | 進行率がこの値を超えたら forecast を止め、終盤を full denoiser に戻す。 |
| `Dry-run Spectrum decisions only` | checkbox | `False` | 実際には forecast せず、actual/forecast 判定だけをログに出す。初期実装・安全確認用。 |
| `Verbose Spectrum trace` | checkbox | `False` | call ごとの decision / reason / window / history を出す。通常は summary のみ。 |

Preset:

| Preset | w | m | lambda | warmup | window | flex | stop | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `Safe` | `0.20` | `8` | `0.50` | `8` | `2` | `0.00` | `0.80` | 出力変化を最小化する検証開始値。 |
| `Balanced` | `0.20` | `16` | `0.50` | `6` | `2` | `0.00` | `0.80` | ComfyUI-Anima-Enhancer 寄りの初期 default。 |
| `Aggressive` | `0.30` | `16` | `0.50` | `6` | `2` | `0.00` | `0.90` | 速度寄り。終盤 guard を遅らせるため、出力変化は増え得る。 |
| `Custom` | UI値を保持 | UI値を保持 | UI値を保持 | UI値を保持 | UI値を保持 | UI値を保持 | UI値を保持 | 個別調整用。 |

UI behavior:

- `Spectrum preset` で `Safe` / `Balanced` / `Aggressive` を選ぶと、`w` / `m` / `lambda` / `warmup` / `window` / `flex` / `stop` はその preset の値へ自動更新する。
- `w` / `m` / `lambda` / `warmup` / `window` / `flex` / `stop` のいずれかを手動で変更した場合、`Spectrum preset` は自動的に `Custom` へ切り替わる。
- `Custom` を選んだ場合は既存の slider 値を維持する。
- `Enable Spectrum experiment=True` にした場合は、UI callback で `Enable TeaCache experiment=False` にする。
- `Enable TeaCache experiment=True` にした場合は、UI callback で `Enable Spectrum experiment=False` にする。
- 古い `ui-config.json` や外部 API 操作などで生成開始時 snapshot が両方 `True` になっていた場合は、実行時の保険としてどちらか片方だけを有効に正規化する。初期実装では既存の TeaCache 優先に合わせ、`TeaCache=True` / `Spectrum=False` として扱う。

必須動作:

- generation の最初の model call は必ず actual forward にする。
- `Warmup steps` 内は必ず actual forward にする。
- `Stop progress` 以降は必ず actual forward にする。
- forecaster history が空、または予測に必要な履歴が不足している場合は actual forward にする。
- shape / dtype / device が前回履歴と一致しない場合は forecaster state を reset し、actual forward に戻す。
- forecast 結果に NaN / Inf を検出した場合は forecast を破棄し、actual forward に戻す。
- `Dry-run Spectrum decisions only=True` の場合は、forecast 条件を満たしても actual forward を行い、decision counter だけを更新する。
- Spectrum state は generation 開始時、model reload、OFF、unload、timestep 巻き戻り検出時に破棄する。
- 初期実装では cond/uncond を複雑に分割せず、Forge Neo が渡す batched model output をそのまま forecaster の対象にする。batch shape や cond/uncond 構造が変化した場合は shape mismatch として reset / actual forward に戻す。

#### 2D sparse attention / NATTEN controls

目的:

- `backend.nn.anima.Block.forward` で見えている `T/H/W` 情報を使い、self-attention を局所 attention に置換して高速化を狙う。

UI:

| Control | UI type | Default | Notes |
| --- | --- | --- | --- |
| `Enable 2D sparse attention` | checkbox | `False` | Forge Neo 本体に存在しない Nz-Anima-PredLab 実験機能。 |
| `Sparse backend` | radio | `NATTEN (optional)` | `NATTEN` が利用可能なら既定で使う。利用不可の場合は選択不可または degraded 表示にし、`Torch prototype` を検証用 fallback として選べるようにする。 |
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
- 現行実装では sparse patch は self-attention のみを対象にする。cross-attention は text/context sequence を使うため変更しない。UI上の `Sparse target` control は未実装で、初期仕様から削除する。
- `Block.forward` は `x_B_T_H_W_D` を受け取るため、H/W/T を保持できる最も実用的な patch point である。

#### Cond/uncond optimization controls

目的:

- Forge Neo 既存の CFG batching を崩さず、未検証条件で追加の高速化余地があるか調べる。

UI:

| Control | UI type | Default | Notes |
| --- | --- | --- | --- |
| `Enable cond/uncond optimization` | checkbox | `False` | 通常 CFG>1 は既に同一 model call に batch されているため、低優先度の実験機能。 |
| `Skip uncond when CFG=1` | checkbox | `False` | Forge 側挙動を確認してから有効化する。 |
| `Guidance step schedule` | checkbox | `False` | 一部 step だけ CFG 処理を変える実験。 |
| `Guidance interval` | slider / number | `1` | schedule 有効時のみ使用。 |

既知事実:

- 2026-05-26 の実測では CFG>1 の通常生成で `cond_or_uncond=[1, 0]`、`input_shape=2x16x1x192x192` が得られ、cond/uncond は同一 model call に batch されていた。
- このため cond/uncond 最適化は高速化の本命ではないが、検証項目からは外さない。
- 現行実装では UI 値の snapshot と設定ログのみ対応しており、`cond_uncond_enabled` に対応する高速化 patch はまだ適用していない。必要時の診断 patch として `cond_batch_trace` は `patcher.py` に存在するが、現行UIフローからは自動適用しない。

#### Low-bit / compile controls

目的:

- Forge Neo の dtype / ops / compile 関連機能を Anima に適用した場合の速度、VRAM、画質を比較する。

UI:

| Control | UI type | Default | Notes |
| --- | --- | --- | --- |
| `Enable Nz low-bit experiment` | checkbox | `False` | 現行実装では設定 snapshot とログのみ。実際の低bit化 patch は未実装。 |
| `Enable torch.compile experiment` | checkbox | `False` | 現行実装では設定 snapshot とログのみ。実際の compile 適用は未実装。 |
| Reload note | markdown | - | `Reload the model after changing settings that require model reload.` を表示する。 |

注意:

- 現行UIには `Precision / ops mode`、`Low-bit target`、`Low-bit format`、`Compile target`、`Compile mode`、`Warmup runs` は存在しない。これらは将来候補として扱う。
- model reload が必要な設定と runtime patch で足りる設定を将来UI上で区別する。
- reload 必須項目を変更した場合、生成直前に silent reload しない。現行実装では markdown と `lowbit_compile_config` ログで「設定変更時にはモデルをリロードしてください」と明示する。
- 初期実装では自動 model reload 機能を実装しない。
- compile 実装を追加する場合、初回生成を遅くする可能性があるため、benchmark では warmup と measured run を分ける。

高速化実験の優先順位:

1. `TeaCache / residual cache experiment`
2. `Spectrum / spectral feature forecasting experiment`
3. `Experimental 2D sparse attention`
4. `Compile / low-bit experiment`
5. `Fast attention kernel`
6. `Cond/uncond optimization`

理由:

- 実測では Anima T2I latent は `1x16x1x192x192` で、T=1、H/W=192 の5D latentとして扱える。
- 実測では attention backend は `attention_sage` で、Anima attention path も有効だった。
- `attention_sage` / `attention_flash` の差し替え実験では拡張側 wrapper が全 attention call を捕捉できたが、対象環境では `attention_sage` が既に有効で、backend 差し替えによる追加高速化余地は限定的だった。
- TeaCache は attention kernel の置換ではなく、Anima block 列を step 単位で skip して residual を再利用するため、既存 attention backend と併用できる可能性がある。
- `daraskme/comfy_anima_tea_cache` の Anima 30 step 検証では、`rel_l1_thresh=0.06..0.07` 付近で品質劣化を抑えつつ約 7〜14% の高速化が観測されている。
- Spectrum は Forge Neo の汎用 model wrapper hook で denoiser output を予測できるため、Anima base v1.0 向け高速化の有力候補として TeaCache の次に優先する。ただし TeaCache とは同時に有効化しない。
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

非対応モデルでは、Anima 固有 patch point を必要とする実験は適用しない。通常生成ごとに警告ログだけを出すことは避け、診断モードまたは明示的な実験適用時に必要最小限の理由を記録する。

Spectrum のように Forge Neo の汎用 model wrapper hook で動作し得る実験は、Anima base v1.0 向け preset を持っていても、モデル family が Anima ではないことだけを理由に禁止しない。実行に必要な hook や tensor 条件が合わない場合のみ fallback する。

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

### 10.6 TeaCache trace

出力項目:

- TeaCache enabled / dry-run / preset
- coefficient profile
- coefficient source
- runtime steps
- profile steps
- start_percent / end_percent
- cache_device
- modulated_source
- rel_l1_thresh
- total model calls
- full calculation count
- skip count
- skip rate
- first full calculation count
- forced full calculation count
- fallback count
- cond/uncond state count

Verbose trace でのみ出す項目:

- step index
- current_percent
- cond_or_uncond
- rel_l1
- estimated distance
- accumulated distance
- should_calc
- reason: `first_step` / `no_previous_residual` / `threshold` / `below_threshold` / `dry_run` / `outside_range` / `force_interval` / `max_skip_streak` / `fallback`

診断目的:

- 初回 step が必ず full calculation になっているか確認する。
- TeaCache が実際に何 step skip したか確認する。
- `Dry-run TeaCache decisions only` で画像を変えずに閾値の挙動を確認する。
- `rel_l1_thresh` の差による skip rate と品質劣化の関係を手動比較できるようにする。

### 10.7 Spectrum trace

出力項目:

- Spectrum enabled / dry-run / preset
- `w`
- `m`
- `lambda`
- warmup steps
- window size
- flex window
- stop progress
- total model calls
- actual forward count
- forecast count
- forecast rate
- fallback count
- error count

Verbose trace でのみ出す項目:

- call index
- step index
- current progress
- decision: `actual` / `forecast`
- reason: `first_call` / `warmup` / `tail_guard` / `window` / `no_history` / `dry_run` / `shape_mismatch` / `nan_inf` / `fallback`
- history length
- current window

診断目的:

- 初回 model call と warmup 範囲が必ず actual forward になっているか確認する。
- forecast が何 call 発生したか確認する。
- `Dry-run Spectrum decisions only` で画像を変えずに予測スケジュールを確認する。
- `Safe` / `Balanced` / `Aggressive` preset の速度差と出力変化を手動比較できるようにする。

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

現行実装は診断機能に加えて、実機検証用の `Identity Patch test`、attention backend 差し替え、2D sparse attention 実験 patch、TeaCache / residual cache 実験 patch、Spectrum / spectral feature forecasting 実験 patch を持つ。

`Identity Patch test` では `backend.nn.anima.Block.forward` を Nz-Anima-PredLab の wrapper に差し替え、wrapper 内で元の `Block.forward` をそのまま呼ぶ。これは高速化ではなく、Forge Neo 本体の推論パイプラインの一部を拡張側から安全に迂回・復帰できるかを確認するための検証である。

検証ログ:

- patch 適用対象と挙動: `target=backend.nn.anima.Block.forward behavior=call_original`
- 各 call の一部: `identity_patch_call=... route=Nz-Anima-PredLab->original_Block.forward`
- 生成後 summary: `identity_patch_summary=calls=... shape_mismatches=... errors=... active=True`

2026-05-26 の StabilityMatrix版 Forge Neo 実機検証では、32 steps / 28 blocks の生成で
`identity_patch_summary=calls=896 num_blocks=28 shape_mismatches=0 errors=0` が得られた。
これは `32 * 28 = 896` と一致し、Anima block-level の推論経路を Nz-Anima-PredLab wrapper
経由に切り替えられることを確認した結果である。

すべての patch は `patcher.py` で管理する。

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
- `backend.nn.anima.Anima._forward`
- `backend.nn.anima.Anima.forward`
- `backend.nn.anima.SelfCrossAttention.compute_attention`
- `backend.nn.anima.SelfCrossAttention.torch_attention_op`
- `backend.attention.attention_function`
- Forge Neo `ModelPatcher.model_options["model_function_wrapper"]`

現行実装済み patch:

- `cond_batch_trace`: `backend.sampling.sampling_function.calc_cond_uncond_batch` の診断 wrapper。現行UIフローからは自動適用しない。
- `block_structure_trace`: `backend.nn.anima.Block.forward` と `SelfCrossAttention.compute_qkv` の診断 wrapper。`Diagnose only` + `Verbose diagnose log` + 他実験無効時に適用する。
- `block_forward_identity`: `Identity Patch test` 用。`Block.forward` を wrapper 経由にして元実装を呼ぶ。
- `attention_kernel`: `Block.forward` と `SelfCrossAttention.torch_attention_op` を wrapper し、選択した Forge attention backend を明示実行する。
- `sparse_attention`: `Block.forward` と `SelfCrossAttention.torch_attention_op` を wrapper し、条件に合う self-attention を 2D sparse attention に置換する。

未実装 / 足場のみ patch:

- `cond_uncond_optimization`
- `lowbit`
- `compile`

2D sparse attention は、flatten 後の generic attention だけでは H/W 情報が失われるため、`Block.forward` または `SelfCrossAttention` 付近で形状情報を扱う方針とする。

TeaCache は `Block.forward` 単体ではなく、Anima diffusion model の block 列全体を囲む patch point を優先する。現行実装では `backend.nn.anima.Anima._forward` が存在すればそれを patch point とし、Forge Neo のように `_forward` が存在しない環境では `backend.nn.anima.Anima.forward` を patch point とする。`cond_or_uncond` が取得できない、または signature が想定外の場合は元の Anima forward 経路へ fallback し、diagnostic log に `teacache_unavailable_reason` を出す。

TeaCache patch の基本挙動:

- full calculation 時は、元の Anima block 列を通常通り実行し、`previous_residual = hidden_after_blocks - hidden_before_blocks` を保存する。
- skip 時は block 列を実行せず、`hidden_before_blocks + previous_residual` を次段へ渡す。
- final layer / unpatchify / VAE decode は skip しない。
- cond/uncond が同一 model call に batch される場合でも、cache state と previous_residual は cond/uncond ごとに分離する。
- generation の最初の model call は必ず full calculation とし、TeaCache skip の候補にしない。
- `Start percent` 範囲内でも、`previous_residual` がない state では必ず full calculation とする。
- `Dry-run TeaCache decisions only` では、skip 判定と summary counter は計算するが、実際の block skip は行わない。
- patch 適用・解除は `patcher.py` の通常 patch 管理に統合し、OFF / unload / unsupported model で必ず復元する。

Spectrum patch の基本挙動:

- 初期実装では Forge Neo の `ModelPatcher.set_model_unet_function_wrapper()` 相当の wrapper を使い、model output tensor を forecaster の対象にする。
- actual forward 時は元の `model_function` を通常通り実行し、その出力を `FastChebyshevForecaster` へ保存する。
- forecast 時は元の `model_function` を呼ばず、Chebyshev ridge regression と短期 Taylor 補間の blend で model output を予測する。
- `w` は Chebyshev 予測の重みとして扱い、`1-w` 側を短期 Taylor 補間として扱う。
- forecast 結果は元出力と同じ shape / dtype / device に戻す。
- `Dry-run Spectrum decisions only` では、forecast 判定と summary counter は計算するが、実際の forecast 出力は使わない。
- Forge Neo built-in `Spectrum Integrated` など、既に別の Spectrum 系 `model_function_wrapper` が存在すると判断できる場合、Nz-Anima-PredLab 側の Spectrum は重ね掛けしない。
- 既存 wrapper が存在する場合の扱いは実装時に慎重に決める。初期実装では、安全に chain できない wrapper を検出した場合は Nz Spectrum を適用せず baseline へ戻す。
- patch 適用・解除は `patcher.py` の通常 patch 管理に統合し、OFF / unload / 例外時に必ず復元する。

patch 優先順位:

1. `Identity Patch test`。診断用であり、他の experimental patch と同時に使わない。
2. `TeaCache / residual cache experiment` または `Spectrum / spectral feature forecasting experiment`。両者は UI 相互排他により同時適用しない。
3. H/W/T 情報を保持できる `Block.forward` / `SelfCrossAttention` 付近での2D sparse attention実験。
4. Forge Neoの低bit・compile機能をAnimaへ適用するためのmodel load / operation選択調査。
5. attention backend差し替え。実測ではSageAttentionが既に使われているため優先度は中から低。
6. cond/uncond最適化。実測で通常CFG>1は同一forward batch化済みのため優先度は低いが、未検証条件の確認項目として維持する。

## 13. Safety

以下の場合は処理を変更しない:

- `Enable Nz-Anima-PredLab` が false
- `Debug log mode` が空欄で、すべての実験機能が baseline / disabled
- model detection が unsupported で、かつ対象 experimental patch が Anima 固有 patch point を必要とする
- txt2img 以外
- img2img / Hires.fix / ControlNet / IP-Adapter / 参照画像系拡張が有効と判断できる
- patch 対象関数が見つからない
- patch 対象関数の signature が想定と異なる
- TeaCache で必要な `cond_or_uncond` / sampling step / sigmas / transformer_options が取得できない
- Spectrum で必要な model function wrapper hook、sampling step、出力 tensor が取得できない

以下の場合は patch を解除して fallback する:

- 例外発生
- 出力 tensor shape が想定と異なる
- NaN / Inf を検出した場合
- TeaCache residual の shape / dtype / device が現在の hidden state と一致しない場合
- Spectrum forecast の shape / dtype / device が現在の model output と一致しない場合
- ユーザーが `Debug log mode` を空欄にし、該当する実験機能も baseline / disabled にした場合
- script unload

TeaCache 固有の安全条件:

- 初回 step / 初回 state は必ず full calculation にする。利用可能な cache がない状態で skip してはならない。
- `previous_residual is None` の状態では、`rel_l1_thresh` や `start_percent` に関係なく full calculation にする。
- `Start percent` より前、または `End percent` より後では full calculation にする。
- `Max skip streak` が設定されている場合、連続 skip が上限に達した次の候補 step は full calculation にする。
- `Force full calc interval` が設定されている場合、該当 interval の step は full calculation にする。
- coefficient profile と runtime steps が一致しない場合はログへ警告を出す。ただし初期実験では `30step` profile を `32step` 実験に使うことは許可する。
- TeaCache と Spectrum の同時有効化は UI で防止する。生成開始時 snapshot が両方 `True` の場合は実行時の保険として片方だけを有効に正規化する。
- TeaCache と 2D sparse attention の同時有効化は初期実装では禁止または degraded 扱いとする。両方が有効な場合は TeaCache を無効化するか、明示的な優先順位に従って片方だけ適用する。

Spectrum 固有の安全条件:

- 初回 model call は必ず actual forward にする。
- `Warmup steps` 内は必ず actual forward にする。
- `Stop progress` 以降は必ず actual forward にする。
- forecaster history が空または不足している状態では forecast しない。
- shape / dtype / device mismatch、NaN / Inf、timestep 巻き戻り、batch shape 変化を検出した場合は forecaster state を reset し、actual forward に戻す。
- Anima base v1.0 以外であることだけを理由に、Spectrum を禁止したり警告ログを出したりしない。実行に必要な hook や tensor 条件が合わない場合のみ fallback する。
- Forge Neo built-in Spectrum など別の Spectrum 系 wrapper との二重適用は避ける。

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

Identity Patch test:

```text
[Nz-Anima-PredLab] applied identity patch kind=block_forward_identity target=backend.nn.anima.Block.forward behavior=call_original
[Nz-Anima-PredLab] version=0.1.1 enabled=True mode=Identity Patch test status=identity-patch
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

TeaCache experiment:

```text
[Nz-Anima-PredLab] teacache_config=enabled=True preset=Balanced threshold=0.0700 progress=0.05..0.95 cache_device=cuda source=first_block_shift coefficient_profile=Anima 2B 30step first_block_shift max_skip_streak=0 force_full_interval=0 dry_run=False
[Nz-Anima-PredLab] teacache_call=call=1 step=0 progress=0.000 decision=full reason=first_call rel_l1=0:None,1:None threshold=0.0700 dry_run=False
[Nz-Anima-PredLab] teacache_summary=model_calls=32 full_calcs=28 skips=4 dry_run_skips=0 skip_rate=0.125 first_full_calcs=1 forced_full_calcs=0 fallbacks=0 errors=0 num_blocks=28 active=True dry_run=False unavailable_reason=None
```

Spectrum experiment:

```text
[Nz-Anima-PredLab] spectrum_config=enabled=True preset=Balanced w=0.20 m=16 lambda=0.50 warmup=6 window=2 flex=0.00 stop_progress=0.80 dry_run=False
[Nz-Anima-PredLab] spectrum_call=call=1 step=0 progress=0.000 decision=actual reason=first_call history=0 window=2 dry_run=False
[Nz-Anima-PredLab] spectrum_summary=model_calls=32 actual_forwards=24 forecasts=8 forecast_rate=0.250 fallbacks=0 errors=0 active=True dry_run=False unavailable_reason=None
```

## 15. Packaging / compatibility

対象:

- Forge Neo 専用 extension として扱う。
- A1111 本家、Forge classic、ComfyUI との互換性は保証しない。

README に明記する項目:

- Forge Neo 専用であること。
- Python / Gradio / Forge Neo の想定系統。
- インストール手順。
- 現行版は診断・計測を主目的としつつ、実験機能として identity patch、attention backend差し替え、2D sparse attention の patch 足場を含むこと。
- TeaCache は実験機能として実装済みだが、品質・速度・skip率は環境ごとの実機検証が必要であること。
- Spectrum は Anima base v1.0 向けの実験機能として実装済みだが、品質・速度・出力変化は環境ごとの実機検証が必要であること。
- トラブルシュート: 拡張が表示されない、unsupported model になる、ログが出ない、生成が遅くなった場合。

## 16. Acceptance Criteria

現行コア機能の完了条件:

- Forge Neo 拡張として読み込まれる。
- settings に `Enable Nz-Anima-PredLab` が表示される。
- settings key が `nzap_*` に統一されている。
- mode を選択できる。
- `scripts/nz_anima_predlab.py` が薄い entrypoint になっている。
- callback 登録が多重実行されない。
- import 時にモデル検査や GPU 処理を行わない。
- unsupported model では Anima 固有 patch による処理変更が起きない。
- supported model で model detection evidence を出力できる。
- `Diagnose only` で total sampling time と average step time を出力できる。
- `Diagnose only` で attention backend、uncond presence、CFG 関連情報、dtype / Forge ops 関連情報を一括出力できる。
- `Identity Patch test` で `backend.nn.anima.Block.forward` を wrapper 経由に切り替え、元の `Block.forward` を呼び戻せる。
- `Identity Patch test` の summary で `steps * num_blocks` と一致する call count、`shape_mismatches=0`、`errors=0` を確認できる。
- `Debug log mode` が空欄、かつすべての個別 experimental control が baseline / disabled の場合、ログ出力と処理変更が止まる。
- `Debug log mode` が空欄でも `Enable attention backend override=True` かつ `Attention backend != Forge current/default`、または `Enable 2D sparse attention=True` の場合は、該当 experimental patch が動作し、設定 snapshot と summary を出力できる。
- 例外時に WebUI 起動と画像生成を可能な限り止めず、status を `error` または degraded 状態へ移せる。

高速化実験版の完了条件:

- patch 対象が明示されている。
- patch 適用前後で復元できる。
- baseline と同一条件で比較できる。
- すべての実験項目を off にした場合、Forge Neo baseline と同等の挙動になる。
- Forge Neo 本体にない機能は `Enable ...` checkbox が off の状態を default とする。
- Forge Neo 本体にある機能は、本体の current/default 値を UI 初期値として表示する。
- 同時に有効化すべきでない実験は UI で相互排他にし、ユーザーがログを読まなくても望ましくない状態にならないようにする。
- 画像が生成される。
- 品質劣化が視覚的に許容範囲内である。
- 1step 平均時間が 5% 以上短縮する。

TeaCache 実験版の完了条件:

- `Enable TeaCache experiment=False` で Forge Neo baseline と同等の推論経路になる。
- `Dry-run TeaCache decisions only=True` で画像内容を変更せず、skip/run 判定だけを summary / verbose trace へ出力できる。
- generation の最初の model call / sampling step が必ず full calculation として記録される。
- `previous_residual is None` の state で skip が発生しない。
- cond/uncond が batch されている場合でも、cache state と previous_residual が cond/uncond ごとに分離される。
- `teacache_summary` で `full_calcs`、`skips`、`skip_rate`、`fallbacks`、`errors`、`active` を確認できる。
- `rel_l1_thresh=0.060..0.070`、`start_percent=0.05`、`end_percent=0.95` の範囲で、32 step Anima生成がエラーなく完走する。
- TeaCache で例外または不整合を検出した場合、full calculation または Forge baseline へfallbackし、WebUIの生成処理を止めない。

Spectrum 実験版の完了条件:

- `Enable Spectrum experiment=False` で Forge Neo baseline と同等の推論経路になる。
- `Spectrum preset=Safe/Balanced/Aggressive` で、定義済みの `w` / `m` / `lambda` / `warmup` / `window` / `flex` / `stop` が UI に反映される。
- Spectrum の各 numeric control を手動変更すると、`Spectrum preset` が `Custom` へ切り替わる。
- `Enable Spectrum experiment=True` にすると `Enable TeaCache experiment=False` になり、`Enable TeaCache experiment=True` にすると `Enable Spectrum experiment=False` になる。
- `Dry-run Spectrum decisions only=True` で画像内容を変更せず、actual/forecast 判定だけを summary / verbose trace へ出力できる。
- generation の最初の model call と warmup 範囲が必ず actual forward として記録される。
- `Stop progress` 以降が actual forward として記録される。
- `spectrum_summary` で `actual_forwards`、`forecasts`、`forecast_rate`、`fallbacks`、`errors`、`active` を確認できる。
- Anima base v1.0 / 30〜32 step の通常生成で `Balanced` preset がエラーなく完走する。
- Anima base v1.0 以外であることだけを理由に、Spectrum を禁止したり警告ログを出したりしない。
- Spectrum で例外または不整合を検出した場合、actual forward または Forge baseline へ fallback し、WebUI の生成処理を止めない。

## 17. Open Issues

- Forge Neo の `torch.compile` 実装箇所を特定する。
- `on_cfg_denoiser()` の呼び出し回数が target sampler ごとに UI steps と一致するか確認する。
- Anima / Cosmos-Predict2 派生 checkpoint の検出条件を実機で確認する。
- 2D sparse attention 実験で、NATTEN / Torch prototype の品質・速度・fallback条件を実機で確認する。
- low-bit / compile 設定ごとに、runtime patch で足りるか model reload が必要かを判定して UI に表示する。
- NATTEN が対象環境で import / 実行できるか確認する。
- Forge Neo の Anima 実装で `Anima.forward` TeaCache patch が32 step生成をエラーなく完走し、期待どおり skip できるか確認する。
- 30 step 用 TeaCache 係数を 32 step 実験へ使った場合の品質・skip率・速度を実機で確認し、必要なら32 step用係数を再校正する。
- TeaCache と attention backend差し替え、2D sparse attention、low-bit / compile を同時に有効化した場合の優先順位を実機で確認する。
- Spectrum の Forge Neo `model_function_wrapper` 実装で、Anima base v1.0 / 30〜32 step 生成がエラーなく完走し、ComfyUI-Anima-Enhancer 寄りの出力変化に収まるか確認する。
- Spectrum の `m=8` と `m=16`、`stop_progress=0.80` と `0.90` の速度・出力変化を Anima base v1.0 で比較する。
- Forge Neo built-in `Spectrum Integrated` が同時に有効な場合の検出方法と、Nz Spectrum を重ね掛けしないための fallback 条件を確認する。

## 18. 確定した仕様判断

2026-05-26 時点で確定した実験 UI 方針:

- 2D sparse attention の初期 preset は後半 block 限定とする。`Block start=14`、`Block end=27` を default にし、ユーザーは slider で `0..27` の範囲を自由に変更できる。
- `Window size` という名称は使わず、UI 表示名は `Local attention window` とする。
- `Local attention window` は slider で指定する。初期値は `15` とする。値は基本的に奇数のみを扱う。
- `Sparse backend` の default は `NATTEN (optional)` とする。NATTEN が利用できない場合は degraded / unavailable とし、`Torch prototype` を検証用 fallback として選べるようにする。
- `Torch prototype` は高速化本命ではなく、NATTEN なしでも破綻するかを切り分けるための backend とする。
- 実験結果はコンソール summary のみで確認する。JSON / CSV 保存は初期実装では行わない。
- 画質比較は目視確認のみとする。baseline / patched pair の自動保存は初期実装では行わない。
- 現行UIは top-level の `Nz-Anima-PredLab` Accordion 1つの配下に `Attention` / `TeaCache` / `Spectrum` / `2D Sparse` / `Cond / Uncond` / `Low-bit / Compile` のカテゴリ別サブ Accordion を置く。他拡張と同じ階層に Nz-Anima-PredLab 用 Accordion を複数作らない。
- low-bit / compile で model reload が必要な設定がある場合、自動 reload は行わない。UI またはログで「設定変更時にはモデルをリロードしてください」と知らせる。

2026-05-27 時点で確定した TeaCache 仕様判断:

- TeaCache は attention kernel 高速化ではなく、Anima block 列を step 単位で skip し、前回 full calculation の residual を再利用する cache management 実験として扱う。
- TeaCache 実装では `TeaCache` サブ Accordion を追加する。
- 初期 default preset は `Balanced` とし、`rel_l1_thresh=0.070`、`start_percent=0.05`、`end_percent=0.95`、`cache_device=cuda`、`modulated_source=first_block_shift` とする。
- 32 step 実験でも `start_percent=0.05` を default としてよい。ただし最初の model call / sampling step は必ず full calculation とし、cache 未初期化状態で skip しない。
- `Dry-run TeaCache decisions only` を用意し、実装初期は画像を変えずにskip判定を観測できるようにする。
- 初期実装では TeaCache summary はコンソールログのみとし、JSON / CSV保存は行わない。

2026-05-28 時点で確定した Spectrum 仕様判断:

- Spectrum は `Spectrum` サブ Accordion として追加する。Nz-Anima-PredLab 用の top-level Accordion は増やさない。
- 初期ターゲットは Anima base v1.0 の推論高速化とする。
- ただし Anima base v1.0 以外であることだけを理由に、Spectrum を禁止したり警告ログを出したりしない。
- 実装方針と preset は Forge Neo built-in Spectrum の汎用 SDXL 寄り設定ではなく、ComfyUI-Anima-Enhancer の Anima 向け先行事例を優先する。
- Spectrum preset は `Safe` / `Balanced` / `Aggressive` / `Custom` とする。
- `Safe`: `w=0.20`、`m=8`、`lambda=0.50`、`warmup=8`、`window=2`、`flex=0.00`、`stop=0.80`。
- `Balanced`: `w=0.20`、`m=16`、`lambda=0.50`、`warmup=6`、`window=2`、`flex=0.00`、`stop=0.80`。
- `Aggressive`: `w=0.30`、`m=16`、`lambda=0.50`、`warmup=6`、`window=2`、`flex=0.00`、`stop=0.90`。
- TeaCache と Spectrum は UI 上で相互排他にする。TeaCache を Enable にしたら Spectrum を disable にし、Spectrum を Enable にしたら TeaCache を disable にする。
- 初期実装では `flex=0.25` のような window 拡張は preset に採用しない。Anima base v1.0 では速度より出力変化の少なさを優先する。
- 初期実装では Spectrum summary はコンソールログのみとし、JSON / CSV保存は行わない。
