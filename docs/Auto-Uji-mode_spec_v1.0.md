# Auto Uji mode 仕様書ドラフト

## 1. 目的

Auto Uji mode は、Forge Neo拡張 `Nz-Anima-PredLab` の UjiCache 実験を効率化するための補助機能である。

現在のUjiCacheは、Threshold、Formula、Prediction strength、EMA smoothing など、多数のパラメータを持つ。これらを1つずつ手動で変更し、そのたびにForge Neo本体のGenerateボタンを押す運用では、実験に時間がかかりすぎる。

Auto Uji mode は、CSV形式で複数のUjiCache条件を記述し、Generateを1回押すだけで、CSVの各行に対応した条件で順番に画像生成を行う。

主目的は、TeaCacheに対してUjiCacheの生成誤差を小さくするためのパラメータ探索を効率化することである。

---

## 2. 基本方針

Auto Uji mode は、UjiCache本体の計算ロジックには深く介入しない。

責務を以下のように分離する。

```text
UjiCache本体:
  residual再利用
  residual予測
  fallback判定
  UjiCache固有ログ
  UjiCache条件のPNG metadata記録

Auto Uji mode:
  CSVを読む
  CSV行数 × Forge本体のBatch countぶん生成をキュー化する
  各iteration直前に、該当CSV行のUjiCache設定をSTATEへ反映する
  seedセットを全CSV行で揃える
  Auto Uji行番号や条件名をログ出力する
```

Auto Uji modeは、あくまでも「外側の実験ランナー」とする。

---

## 3. UI仕様

`Nz-Anima-PredLab` の `UjiCache` サブアコーディオン内に、さらに下位アコーディオンとして `Auto Uji mode` を追加する。

Auto Uji modeは、UjiCacheと同階層のサブアコーディオンではなく、UjiCache配下のサブサブアコーディオンとする。

階層構造:

```text
Nz-Anima-PredLab
└─ UjiCache
   └─ Auto Uji mode
      ├─ Enable Auto Uji mode
      ├─ CSV input
      ├─ Optional: Preview parsed rows
      ├─ Optional: Row limit
      └─ Optional: Strict CSV validation
```

MVPでは、最低限以下を実装する。

```text
- Enable Auto Uji mode
- CSV input textbox
```

### 3.1 Enable Auto Uji mode

Auto Uji modeを有効にするcheckbox。

Auto Uji modeは、UjiCache操作パネルのパラメータ指定をCSVで代行する補助機能である。

Auto Uji modeから、親の `Enable Nz-Anima-PredLab` や `Enable UjiCache experiment` のON/OFFを自動操作しない。

Auto Uji modeを実行する場合、ユーザーは通常通り `Enable Nz-Anima-PredLab` と `Enable UjiCache experiment` を手動でONにする。

`Enable UjiCache experiment` がOFFの場合、UjiCache配下の機能はすべてOFFとして扱う。

そのため、Auto Uji modeのcheckboxがONであっても、`Enable UjiCache experiment` がOFFならAuto Uji modeは実行されない。

Auto Uji modeは、UjiCacheの操作パネルに存在するパラメータを行ごとに上書きするだけであり、他のサブアコーディオンの機能を操作しない。

### 3.2 CSV input

複数行のテキスト入力欄。

ユーザーはここにCSVを貼り付ける。

例:

```csv
name,threshold,formula,prediction_strength,slope_ema_smoothing,curve_ema_smoothing,taylor2_curve_strength,apply_prediction_from_skip,use_prediction_after_progress,max_skip_streak,force_full_interval
teacache_ref,0.07,teacache,0.00,0.00,0.00,0.25,2,0.00,0,0
linear_a,0.21,linear,0.50,0.20,0.00,0.25,2,0.00,0,0
taylor2_a,0.21,taylor2,0.50,0.20,0.10,0.25,2,0.00,0,0
```

---

## 4. CSV仕様

### 4.1 基本形式

1行目をヘッダー行とする。

2行目以降を実験条件行とする。

空行は無視する。

末尾カンマによる空列は無視してよい。

列名はsnake_caseを推奨する。

### 4.2 MVPで対応する列

MVPでは以下の列に対応する。

```text
name
threshold
formula
prediction_strength
slope_ema_smoothing
curve_ema_smoothing
taylor2_curve_strength
apply_prediction_from_skip
use_prediction_after_progress
max_skip_streak
force_full_interval
```

Auto Uji modeでは、以下の設定はCSVから操作しない。

```text
Enable UjiCache experiment
Cache device
Dry run
Verbose UjiCache trace
```

これらはGenerate時点のUI値、または既存の有効化ルールに従う。

### 4.3 formulaの短縮名

CSV内では短縮名を許容する。

```text
teacache / residual / residual_only
  -> TeaCache (residual only)

linear / linear_extrapolation
  -> Linear extrapolation

taylor2 / taylor / quadratic
  -> Taylor2 curve
```

不明なformulaが指定された場合は、その行をエラーとして扱う。

MVPでは、CSVの一部だけを実行しない。

parseに失敗した場合は、エラーをログ出力して生成を中断する。

### 4.4 未指定列の扱い

未指定の列は、現在のUjiCache UI値、またはUjiCacheの既定値を使う。

MVPでは、挙動を明確にするため、以下の方針を推奨する。

```text
CSVで指定された値:
  その行の値で上書きする

CSVで未指定の値:
  Generate時点のUjiCache UI値を使う
```

これにより、CSVには変更したい列だけを書ける。

---

## 5. 生成単位

Auto Uji modeでは、CSVの1行を1つのUjiCache条件として扱う。

```text
CSV 1行 = 1つのUjiCache条件
Forge本体のBatch count = 各CSV行で繰り返すiteration数
```

Auto Uji mode有効時は、Forge本体の `p.n_iter` をCSV行数で置き換えるのではなく、元の `p.n_iter` を各CSV行の繰り返し回数として尊重する。

実行時の総iteration数は以下になる。

```text
実行時 p.n_iter = CSV行数 × 元の p.n_iter
```

Forge本体の `batch_size` は尊重する。

`batch_size=1` は強制しない。

例:

```text
CSV行数: 5
Forge Batch count: 2
Forge batch_size: 4

生成総枚数:
5 × 2 × 4 = 40枚
```

この場合、

```text
CSV 1行目のUjiCache条件で 2 × 4 = 8枚生成
CSV 2行目のUjiCache条件で 2 × 4 = 8枚生成
CSV 3行目のUjiCache条件で 2 × 4 = 8枚生成
...
```

となる。

---

## 6. seed仕様

Auto Uji modeでは、CSV各行で同じseedセットを使う。

単一seedを全画像に固定するのではなく、Forgeの元の `p.n_iter` と `batch_size` に応じたseedセットを作り、そのseedセットをCSV全行に複製する。

### 6.1 固定seedの場合

Forge本体のseedが `12345`、Batch countが1、batch_sizeが4、CSVが3行の場合:

```text
CSV 1行目: 12345, 12346, 12347, 12348
CSV 2行目: 12345, 12346, 12347, 12348
CSV 3行目: 12345, 12346, 12347, 12348
```

Forge本体のseedが `12345`、Batch countが2、batch_sizeが4、CSVが3行の場合:

```text
CSV 1行目: 12345, 12346, 12347, 12348 / 12349, 12350, 12351, 12352
CSV 2行目: 12345, 12346, 12347, 12348 / 12349, 12350, 12351, 12352
CSV 3行目: 12345, 12346, 12347, 12348 / 12349, 12350, 12351, 12352
```

### 6.2 ランダムseedの場合

Forge本体のseedが `-1` の場合、ForgeがそのGenerate実行時に決定した基準seedを使う。

たとえば、Generate時に基準seedが `847362910` に解決され、Batch countが1、batch_sizeが4、CSVが3行の場合:

```text
CSV 1行目: 847362910, 847362911, 847362912, 847362913
CSV 2行目: 847362910, 847362911, 847362912, 847362913
CSV 3行目: 847362910, 847362911, 847362912, 847362913
```

つまり、1回のAuto Uji実験内では全CSV行で同じseedセットを使う。

次にGenerateを押した場合、Forge側が新しいランダムseedを決定するため、新しいseedセットが使われる。

### 6.3 seed mode UI

MVPではseed mode UIは作らない。

Auto Uji modeは常に「全CSV行で同じseedセットを共有する」挙動とする。

シード違いの影響だけを調べたい場合は、Auto Uji modeを使わず、Forge本体の通常Batch count / Batch sizeを使う。

---

## 7. residual / cache分離仕様

Auto Uji modeでは、CSVの前行で使われたUjiCache residual、residual_history、previous_residualなどが次行に混ざらないことを必須条件とする。

Auto Uji modeはUjiCache本体のcacheを直接操作しない。

既存の生成開始処理をCSV行ごとに走らせ、UjiCache本体側のgeneration単位リセットに任せる。

期待する挙動:

```text
CSV 1行目:
  generation_index = N
  UjiCache state A

CSV 2行目:
  generation_index = N + 1
  UjiCache state B

CSV 3行目:
  generation_index = N + 2
  UjiCache state C
```

各行のUjiCache stateは独立している必要がある。

---

## 8. ログ仕様

UjiCache本体の詳細ログは既存のまま利用する。

Auto Uji modeでは、最低限、各CSV行の開始ログだけを追加する。

例:

```text
[Nz-Anima-PredLab] auto_uji_row_start index=1/3 name=linear_a threshold=0.2100 formula=Linear extrapolation prediction_strength=0.50 batch_size=4 seeds=12345..12348
```

MVPでは、行ごとのsummary集約は必須としない。

将来的には、Auto Uji mode側で以下のような簡易summaryを追加してもよい。

```text
auto_uji_row_summary index=1 name=linear_a skips=... prediction_used=... fallback_used=... errors=...
```

ただし、このsummaryはUjiCache本体の内部ロジックに依存しすぎない範囲で実装する。

---

## 9. PNG metadata仕様

UjiCache条件のPNG metadata記録は、Auto Uji mode固有の機能ではなく、UjiCache本体側の改修項目とする。

理由:

```text
- Auto Uji mode使用時だけでなく、1枚ずつ生成した場合にもUjiCache条件をmetadataへ残したい
- UjiCacheの条件記録はUjiCache本体の責務である
- Auto Uji modeは実験ランナーであり、UjiCache本体のmetadata方針を持つべきではない
```

Auto Uji modeのMVPではPNG metadataの追加改修は行わない。

将来的にAuto Uji mode固有の情報として以下をmetadataに追加することは検討してよい。

```text
Auto Uji row index
Auto Uji row name
Auto Uji CSV hash
```

ただし、これはUjiCache本体のmetadata整備後に行う。

---

# 実装設計

## 10. 変更対象ファイル

主な変更対象は以下。

```text
nz_anima_predlab/script.py
nz_anima_predlab/state.py
nz_anima_predlab/logging.py または diagnostics.py
```

新規ファイルを切る場合:

```text
nz_anima_predlab/auto_ujicache.py
```

MVPでは、CSV parserとrow適用ロジックを `auto_ujicache.py` に分離することを推奨する。

---

## 11. Auto Uji用データ構造

新規dataclassを作る。

```python
@dataclass
class AutoUjiRow:
    index: int
    name: str
    threshold: float | None = None
    formula: str | None = None
    prediction_strength: float | None = None
    slope_ema_smoothing: float | None = None
    curve_ema_smoothing: float | None = None
    taylor2_curve_strength: float | None = None
    apply_prediction_from_skip: int | None = None
    use_prediction_after_progress: float | None = None
    max_skip_streak: int | None = None
    force_full_interval: int | None = None
```

`None` は「CSV未指定」を表す。

CSV未指定の値は、Generate時点のUjiCache UI値を保持する。

---

## 12. state.pyの追加項目

`RuntimeState` にAuto Uji用の状態を追加する。

```python
auto_ujicache_enabled: bool = False
auto_ujicache_csv: str = ""
auto_ujicache_rows: list[Any] = field(default_factory=list)
auto_ujicache_active: bool = False
auto_ujicache_row_index: int | None = None
auto_ujicache_row_name: str | None = None
auto_ujicache_parse_error: str | None = None
```

ただし、CSV row本体は `STATE` に長く保持しすぎず、`p._nzap_auto_ujicache_rows` に保存してもよい。

推奨:

```text
STATE:
  現在実行中のAuto Uji状態だけを持つ

p:
  今回のGenerateに紐づくrowsを持つ
```

---

## 13. UI引数の追加方針

`script.py` のUjiCacheサブアコーディオン内に `Auto Uji mode` を追加する。

UI componentは、既存の戻り値リストの末尾に追加するのが安全。

理由:

```text
- 既存UI引数の位置をなるべく崩さない
- _apply_ui_args の後方互換処理を壊しにくい
```

追加するUI component:

```python
auto_ujicache_enabled = gr.Checkbox(
    label="Enable Auto Uji mode",
    value=False,
    elem_id="nzap-auto-uji-enable",
)

auto_ujicache_csv = gr.Textbox(
    label="Auto Uji CSV",
    lines=6,
    elem_id="nzap-auto-uji-csv",
)
```

必要なら説明文:

```python
gr.Markdown(
    "Each CSV row is one UjiCache condition. Forge batch_size is preserved. The same seed set is reused for every CSV row."
)
```

---

## 14. hookごとの責務

### 14.1 before_process(p)

Auto Uji modeの準備を行う。

責務:

```text
- UI引数をSTATEへ反映する
- Auto UjiがOFFなら何もしない
- 親EnableまたはUjiCache EnableがOFFの場合、Auto UjiはOFFとして扱い、何もしない
- CSVをparseする
- rowsが0件ならエラーログを出して生成を中断する
- p._nzap_auto_ujicache_rows にrowsを保存する
- p._nzap_auto_ujicache_original_n_iter に元のn_iterを保存する
- p.n_iter = len(rows) × 元のp.n_iter に変更する
- p.batch_size は変更しない
```

疑似コード:

```python
def before_process(self, p, *script_args):
    _apply_ui_args(script_args)

    if not STATE.enabled:
        return
    if not STATE.ujicache_enabled:
        return
    if not STATE.auto_ujicache_enabled:
        return

    rows = parse_auto_ujicache_csv(STATE.auto_ujicache_csv)
    if not rows:
        raise RuntimeError("Auto Uji CSV has no valid rows")

    original_n_iter = int(getattr(p, "n_iter", 1) or 1)

    p._nzap_auto_ujicache_rows = rows
    p._nzap_auto_ujicache_original_n_iter = original_n_iter
    p._nzap_auto_ujicache_original_batch_size = p.batch_size

    p.n_iter = len(rows) * original_n_iter
```

### 14.2 process(p)

seedセットを全CSV行で揃える。

Forge Neoでは、この時点で `p.all_seeds` が生成済みである。

責務:

```text
- Auto Uji rowsがあるか確認
- 元のp.n_iter × batch_sizeぶんのseed templateを作る
- seed templateをCSV行数ぶん複製する
- subseedも同様に扱う
```

疑似コード:

```python
def process(self, p, *script_args):
    rows = getattr(p, "_nzap_auto_ujicache_rows", None)
    if not rows:
        return

    batch_size = int(getattr(p, "batch_size", 1) or 1)
    row_count = len(rows)
    original_n_iter = int(getattr(p, "_nzap_auto_ujicache_original_n_iter", 1) or 1)
    template_size = original_n_iter * batch_size

    seed_template = list(p.all_seeds[:template_size])
    if len(seed_template) < template_size:
        seed_template = [p.all_seeds[0] + i for i in range(template_size)]

    p.all_seeds = seed_template * row_count

    subseed_template = list(p.all_subseeds[:template_size])
    if len(subseed_template) < template_size:
        subseed_template = [p.all_subseeds[0] + i for i in range(template_size)]

    p.all_subseeds = subseed_template * row_count
```

注意:

```text
- subseed_strength > 0 の場合の挙動は要実機確認
- MVPでは p.all_subseeds も同じtemplate複製にする
```

### 14.3 process_before_every_sampling(p)

現在のiterationに対応するCSV行をSTATEへ反映する。

責務:

```text
- p.iteration と元のp.n_iterからrowを選ぶ
- rowの値でSTATE.ujicache_*を上書きする
- STATE.ujicache_enabled のON/OFFは変更しない
- 他のサブアコーディオンの機能ON/OFFは変更しない
- row start logを出す
- その後、既存の _begin_generation() を実行する
```

row選択は、元の `p.n_iter` を各CSV行の繰り返し回数として使う。

```python
original_n_iter = int(getattr(p, "_nzap_auto_ujicache_original_n_iter", 1) or 1)
row_index = int(getattr(p, "iteration", 0) or 0) // original_n_iter
row = rows[row_index]
```

重要:

```text
Auto Uji rowの適用は、_begin_generation() の中で STATE.reset_generation() が走る前後の順序に注意する。
```

推奨順:

```text
1. _apply_ui_args(script_args)
2. Auto Uji rowをSTATEへ反映
3. _begin_generation_core相当の処理を行う
```

現在の `_begin_generation()` が内部で `_apply_ui_args()` を呼ぶため、Auto Uji row適用後に `_begin_generation()` をそのまま呼ぶと、UI値で上書きされる可能性がある。

そのため、実装時には以下のどちらかを選ぶ。

### 案A: `_begin_generation()` を分割する

```python
def _begin_generation(p, script_args, source):
    _apply_ui_args(script_args)
    _apply_auto_ujicache_row_if_needed(p)
    _begin_generation_after_state_applied(p, source)
```

この案が最も安全。

### 案B: `_begin_generation()` の末尾近くでAuto Uji rowを再適用する

実装は簡単だが、設定反映順が分かりにくくなるため非推奨。

---

## 15. CSV row適用関数

`auto_ujicache.py` に以下の関数を作る。

```python
def apply_auto_ujicache_row_to_state(row: AutoUjiRow) -> None:
    if row.threshold is not None:
        STATE.ujicache_threshold = row.threshold
    if row.formula is not None:
        STATE.ujicache_formula = row.formula
    if row.prediction_strength is not None:
        STATE.ujicache_prediction_strength = row.prediction_strength
    ...
```

この関数は、`STATE.ujicache_enabled`、`STATE.teacache_enabled`、`STATE.spectrum_enabled` などの機能ON/OFFを変更しない。

Auto Uji modeは、UjiCache操作パネルに存在するパラメータの行単位上書きだけを担当する。

値のclampは、できれば既存の `STATE.apply_options()` と同じ基準に合わせる。

実装重複を避けるため、専用の小さなclamp関数を `auto_ujicache.py` 側に持ってもよい。

---

## 16. patch設定との関係

Auto Uji modeは、patchのON/OFFを直接操作しない。

Auto Uji modeは、ユーザーが手動で有効化したUjiCacheに対して、操作パネル上のパラメータ指定をCSVで代行する。

そのため、Auto Uji modeを使う場合は、通常通り `Enable UjiCache experiment` がONであり、既存の `_configure_generation_patches()` によって `ujicache` patchが適用される必要がある。

既存のpatch排他方針は維持するが、Auto Uji mode自身が他のサブアコーディオンの機能を自動OFFにすることはしない。

相互排他が必要な組み合わせは、Nz-Anima-PredLab本体側の既存ルールで防ぐ。

---

## 17. エラー処理

CSV parseに失敗した場合、MVPでは生成を中断する。

理由:

```text
- 誤った条件で大量生成されるほうが危険
- 実験ログの信頼性が落ちる
```

想定ログ:

```text
[Nz-Anima-PredLab] auto_uji_csv_error line=3 column=threshold reason=invalid_float value=abc
```

不明列は警告のみで無視してよい。

```text
[Nz-Anima-PredLab] auto_uji_csv_warning unknown_column=foo ignored=True
```

---

## 18. MVPスコープ外

MVPでは以下は扱わない。

```text
- promptをCSVで変える
- negative_promptをCSVで変える
- width / heightをCSVで変える
- sampler / schedulerをCSVで変える
- modelをCSVで変える
- Enable UjiCache experimentをCSVまたはAuto Uji modeから切り替える
- Cache deviceをCSVで変える
- Dry runをCSVで変える
- Verbose UjiCache traceをCSVで変える
- UjiCache本体の予測ロジックを変更する
- UjiCache本体のfallback条件を変更する
- PNG metadata完全化
- rowごとの詳細summary集約
- CSVファイルアップロード
```

CSVファイルアップロードは将来追加可能。

---

## 19. 受け入れ条件

### 19.1 基本生成

CSVが3行、Forge Batch countが1、batch_sizeが1の場合、Generate 1回で3枚生成される。

CSVが3行、Forge Batch countが2、batch_sizeが1の場合、Generate 1回で6枚生成される。

### 19.2 batch_size尊重

CSVが3行、Forge Batch countが1、batch_sizeが4の場合、Generate 1回で12枚生成される。

CSVが3行、Forge Batch countが2、batch_sizeが4の場合、Generate 1回で24枚生成される。

### 19.3 seedセット固定

Forge seedが12345、Batch countが1、batch_sizeが4、CSVが3行の場合、各行のseedセットが以下になる。

```text
12345, 12346, 12347, 12348
12345, 12346, 12347, 12348
12345, 12346, 12347, 12348
```

Forge seedが12345、Batch countが2、batch_sizeが4、CSVが3行の場合、各行のseedセットが以下になる。

```text
12345, 12346, 12347, 12348 / 12349, 12350, 12351, 12352
12345, 12346, 12347, 12348 / 12349, 12350, 12351, 12352
12345, 12346, 12347, 12348 / 12349, 12350, 12351, 12352
```

### 19.4 random seed対応

Forge seedが-1の場合、Generateごとに1つのseedセットが作られ、そのseedセットが全CSV行で共有される。

### 19.5 residual非共有

CSV 2行目以降で、前行のUjiCache residual_history / previous_residual が使い回されない。

UjiCacheのgeneration_indexによるstate分離が働いていることをログで確認できる。

### 19.6 UjiCache本体非侵襲

Auto Uji modeの追加によって、UjiCache本体のresidual予測ロジック、fallbackロジック、patch本体には変更を加えない。

### 19.7 Auto Uji OFF時のbaseline

Auto Uji modeがOFFの場合、既存のUjiCache UIおよび通常生成の挙動は変わらない。

### 19.8 Auto Ujiの操作範囲

Auto Uji modeは、`Enable UjiCache experiment`、`Cache device`、`Dry run`、`Verbose UjiCache trace` をCSVから変更しない。

Auto Uji modeは、他のサブアコーディオンの機能ON/OFFを変更しない。

### 19.9 UjiCache OFF時の扱い

`Enable UjiCache experiment` がOFFの場合、Auto Uji modeはOFFとして扱われる。

Auto Uji modeのcheckboxがONであっても、CSV parse、p.n_iter変更、seed template複製、row適用は行わない。

---

## 20. 推奨実装順

1. `auto_ujicache.py` を追加し、CSV parserと `AutoUjiRow` を実装する。
2. `state.py` にAuto Uji用の最小状態を追加する。
3. `script.py` のUjiCacheアコーディオン内に `Auto Uji mode` UIを追加する。
4. UI戻り値リスト末尾にAuto Uji用componentを追加する。
5. `_apply_ui_args()` と `STATE.apply_options()` をAuto Uji引数に対応させる。
6. `before_process()` を追加または既存hookを拡張し、CSV parseと `p.n_iter = len(rows) × 元のp.n_iter` を行う。
7. `process()` を追加または拡張し、元のp.n_iter × batch_sizeぶんの `p.all_seeds` / `p.all_subseeds` をseed template方式で上書きする。
8. `_begin_generation()` を分割し、Auto Uji row適用後に既存の生成開始処理が走るようにする。
9. `process_before_every_sampling()` で `p.iteration // 元のp.n_iter` に対応するrowをSTATEへ反映する。
10. `auto_uji_row_start` ログを追加する。
11. CSV 2行 × Batch count 1 × batch_size 1、CSV 2行 × Batch count 2 × batch_size 1、CSV 2行 × Batch count 2 × batch_size 4、seed固定、seed=-1 で実機テストする。
12. UjiCacheのresidualが行をまたいで混ざらないことをログで確認する。
