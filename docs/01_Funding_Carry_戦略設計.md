# 01 Funding Carry 戦略設計

## 1. 目的と位置づけ

Funding Carry は、`predicted funding`、`current funding`、`basis`、`open interest` の歪みを利用して、**方向性リスクをできるだけ抑えながら funding 収益を積み上げる**ことを目的とする戦略です。  
Hyperliquid 上では、最初のバックテスト対象として扱いやすく、**bar ベースの検証から開始できる**のが大きな利点です。

本ドキュメントでは、以下を実装直前の粒度で定義します。

- 戦略クラス設計
- 特徴量カラム定義
- バックテスト用ディレクトリ構成

---

## 2. 戦略の基本仕様

### 2.1 対象銘柄
- BTC
- ETH
- HYPE

### 2.2 推奨時間軸
- シグナル判定: `1h`
- 執行補助: `5m` または `1m`

### 2.3 戦略の基本思想
以下の状況で、**現物ロング + perp ショート**、または条件によっては逆方向を検討する。

- predicted funding が極端
- current funding が十分大きい
- mark price が oracle price に対してプレミアム方向に歪んでいる
- OI が増加しており crowding が強い
- spread がまだ許容範囲

### 2.4 バックテストの基本単位
- `1h` のシグナル判定をベースにする
- 執行価格は `next_open`、`5m TWAP`、`1m VWAP` など複数モデルで比較する
- PnL は必ず以下に分解する
  - `price_pnl`
  - `funding_pnl`
  - `fee`
  - `slippage`

---

## 3. 戦略クラス設計

### 3.1 クラス責務

```python
class FundingCarryStrategy(BaseStrategy):
    """
    Funding と basis の歪みを利用して carry を取りに行く戦略。
    方向性リスクを抑えた spot-perp carry と、
    perp-only bias carry の両方をサポートする。
    """
```

### 主な責務
- funding 系特徴量の生成
- basis/OI を用いた crowding 判定
- エントリー/イグジットのターゲットポジション生成
- funding 通過前後の保有制御
- PnL を funding と price に分解して出力

---

### 3.2 推奨コンストラクタ

```python
class FundingCarryStrategy(BaseStrategy):
    def __init__(
        self,
        symbols: list[str],
        signal_timeframe: str = "1h",
        execution_timeframe: str = "5m",
        mode: str = "spot_perp",  # or "perp_only"
        pred_funding_entry: float = 0.00015,
        current_funding_min: float = 0.00010,
        basis_entry: float = 0.0010,
        basis_exit: float = 0.0003,
        basis_stop: float = 0.0035,
        oi_change_1h_min: float = 0.0,
        spread_bps_max: float = 4.0,
        max_hold_hours: int = 2,
        max_notional_pct: float = 0.15,
        min_signal_interval_hours: int = 1,
    ):
        ...
```

---

### 3.3 想定 I/O

#### 入力 DataFrame
シグナル判定時点で最低限必要な列:

- `timestamp`
- `symbol`
- `close`
- `mark_price`
- `oracle_price`
- `current_funding`
- `pred_funding_1h`
- `open_interest`
- `spread_bps`
- `spot_price`（spot-perp モード時）

#### 出力 DataFrame
少なくとも以下の列を返す。

- `timestamp`
- `symbol`
- `signal`
- `target_position`
- `entry_side`
- `expected_hold_until`
- `reason_code`

##### `signal` の例
- `long_spot_short_perp`
- `short_spot_long_perp`
- `flat`
- `exit_basis_normalized`
- `exit_time_stop`
- `exit_funding_decay`
- `exit_basis_stop`

---

### 3.4 主要メソッド設計

#### `build_features(df: pd.DataFrame) -> pd.DataFrame`
責務:
- funding/basis/OI 系の特徴量を作る
- rolling z-score を付与する
- carry 可能性を表す中間列を作る

生成対象の中間列例:
- `basis`
- `basis_bps`
- `funding_z_24h`
- `oi_change_1h`
- `oi_change_4h`
- `basis_change_1h`
- `carry_score`

---

#### `generate_signal(df: pd.DataFrame) -> pd.DataFrame`
責務:
- シグナル条件を満たすか判定
- `target_position` を決定

##### ロング現物・ショート perp エントリー例
以下をすべて満たすとき:
- `pred_funding_1h > pred_funding_entry`
- `current_funding > current_funding_min`
- `basis > basis_entry`
- `oi_change_1h >= oi_change_1h_min`
- `spread_bps < spread_bps_max`

出力:
- `signal = "long_spot_short_perp"`
- `target_position = +1`（内部的には pair position として扱う）

---

#### `apply_exit_rules(df: pd.DataFrame, position_df: pd.DataFrame) -> pd.DataFrame`
責務:
- 保有ポジションに対する exit 条件判定

##### exit 条件
- `basis < basis_exit`
- `basis > entry_basis + basis_stop`
- `pred_funding_1h` が閾値割れ
- `holding_hours >= max_hold_hours`

---

#### `position_sizing(df: pd.DataFrame) -> pd.DataFrame`
責務:
- 最大 notional 制約
- symbol ごとの重みづけ
- OI や spread に応じた size 調整

##### 推奨
- BTC: 基準サイズ 1.0
- ETH: 0.8
- HYPE: 0.5

---

#### `decompose_pnl(trades_df, funding_df) -> pd.DataFrame`
責務:
- `price_pnl`
- `funding_pnl`
- `fee`
- `slippage`
- `net_pnl`
を時系列で分解出力する

---

### 3.5 設定オブジェクト例

```python
from dataclasses import dataclass

@dataclass
class FundingCarryConfig:
    symbols: list[str]
    signal_timeframe: str = "1h"
    execution_timeframe: str = "5m"
    mode: str = "spot_perp"
    pred_funding_entry: float = 0.00015
    current_funding_min: float = 0.00010
    basis_entry: float = 0.0010
    basis_exit: float = 0.0003
    basis_stop: float = 0.0035
    oi_change_1h_min: float = 0.0
    spread_bps_max: float = 4.0
    max_hold_hours: int = 2
    max_notional_pct: float = 0.15
```

---

## 4. 特徴量カラム定義

### 4.1 生データ列

| カラム名 | 型 | 説明 |
|---|---:|---|
| `timestamp` | datetime | レコード時刻 |
| `symbol` | str | 銘柄 |
| `close` | float | 対象足の終値 |
| `mark_price` | float | perp の mark |
| `oracle_price` | float | oracle 価格 |
| `current_funding` | float | 現在 funding |
| `pred_funding_1h` | float | 次 funding の予測値 |
| `open_interest` | float | OI |
| `spread_bps` | float | 板スプレッド |
| `spot_price` | float | 現物価格 |

---

### 4.2 派生特徴量

| カラム名 | 型 | 定義 | 用途 |
|---|---:|---|---|
| `basis` | float | `(mark_price - oracle_price) / oracle_price` | carry 歪み判定 |
| `basis_bps` | float | `basis * 10000` | 可読化 |
| `basis_change_1h` | float | `basis.diff(1)` on 1h | basis 縮小/拡大判定 |
| `oi_change_1h` | float | `open_interest.pct_change(1)` on 1h | crowding 判定 |
| `oi_change_4h` | float | `open_interest.pct_change(4)` on 1h | 中期 OI 増減 |
| `funding_z_24h` | float | 24h rolling z-score | 異常 funding 判定 |
| `spread_ok` | int | `spread_bps < threshold` | 執行可否判定 |
| `carry_score` | float | funding/basis/OI の合成スコア | 補助指標 |

#### `carry_score` の例
```python
carry_score = (
    0.5 * rank(pred_funding_1h)
    + 0.3 * rank(basis)
    + 0.2 * rank(oi_change_1h)
)
```

---

### 4.3 シグナル関連列

| カラム名 | 型 | 説明 |
|---|---:|---|
| `entry_long_spot_short_perp` | int | 条件成立で 1 |
| `entry_short_spot_long_perp` | int | 逆方向条件成立で 1 |
| `exit_basis_normalized` | int | basis 正常化 exit |
| `exit_basis_stop` | int | basis 拡大による stop |
| `exit_funding_decay` | int | predicted funding 鈍化 |
| `exit_time_stop` | int | 最大保有時間到達 |

---

### 4.4 PnL 分解列

| カラム名 | 型 | 説明 |
|---|---:|---|
| `position_spot` | float | 現物ポジション |
| `position_perp` | float | perp ポジション |
| `price_pnl_spot` | float | 現物価格損益 |
| `price_pnl_perp` | float | perp 価格損益 |
| `funding_pnl` | float | funding 損益 |
| `fee` | float | 手数料 |
| `slippage_cost` | float | スリッページ |
| `net_pnl` | float | 純損益 |

---

## 5. バックテスト用ディレクトリ構成

```text
strategies/
└── funding_carry/
    ├── README.md
    ├── config/
    │   ├── default.yaml
    │   ├── btc.yaml
    │   ├── eth.yaml
    │   └── hype.yaml
    ├── data_contract/
    │   ├── raw_schema.md
    │   └── feature_schema.md
    ├── src/
    │   ├── strategy.py
    │   ├── config.py
    │   ├── feature_builder.py
    │   ├── signal_rules.py
    │   ├── exit_rules.py
    │   ├── sizing.py
    │   └── pnl.py
    ├── tests/
    │   ├── test_features.py
    │   ├── test_signals.py
    │   ├── test_exit_rules.py
    │   └── test_pnl_decompose.py
    ├── notebooks/
    │   ├── 01_feature_check.ipynb
    │   ├── 02_signal_review.ipynb
    │   └── 03_backtest_summary.ipynb
    ├── reports/
    │   ├── stats/
    │   ├── charts/
    │   └── trades/
    └── artifacts/
        ├── features/
        ├── signals/
        └── backtests/
```

---

### 5.1 各フォルダの役割

#### `config/`
パラメータ管理。  
symbol 別設定を分離できるようにする。

#### `data_contract/`
入力スキーマと特徴量スキーマを文書化する。  
バックテスト時の破綻を防ぐため、列名・型・欠損許容を明示する。

#### `src/`
実装本体。

- `strategy.py`: クラス定義
- `feature_builder.py`: funding/basis/OI 特徴量生成
- `signal_rules.py`: entry 条件
- `exit_rules.py`: exit 条件
- `sizing.py`: ロット調整
- `pnl.py`: funding/price PnL 分解

#### `tests/`
最低限以下を担保する。
- basis 計算が正しい
- funding z-score に future leak がない
- exit の優先順位が一定
- funding と price の PnL が正しく分離される

---

### 5.2 推奨 raw data 配置

```text
data/
└── hyperliquid/
    ├── candles/
    │   ├── BTC_1m.parquet
    │   ├── BTC_5m.parquet
    │   ├── BTC_1h.parquet
    │   └── ...
    ├── asset_ctx/
    │   ├── BTC.parquet
    │   ├── ETH.parquet
    │   └── HYPE.parquet
    ├── funding/
    │   ├── funding_history_BTC.parquet
    │   ├── funding_history_ETH.parquet
    │   └── predicted_funding.parquet
    └── spot/
        ├── UBTC_USDC_1m.parquet
        ├── UETH_USDC_1m.parquet
        └── HYPE_USDC_1m.parquet
```

---

## 6. 実装時の注意点

1. **future leak を避ける**  
   predicted funding を参照する時点を厳密に管理する。

2. **funding 確定タイミングを別管理する**  
   price bar と funding event を混同しない。

3. **spot-perp と perp-only を分離評価する**  
   特に初期検証では別戦略として扱う方が良い。

4. **PnL 分解を標準機能にする**  
   funding で勝っているのか方向性で勝っているのかを必ず可視化する。

---

## 7. 最小実装順

1. `feature_builder.py`
2. `signal_rules.py`
3. `exit_rules.py`
4. `strategy.py`
5. `pnl.py`
6. `tests/`
7. notebook で可視化

---

## 8. この戦略で最初に確認すべきこと

- BTC / ETH で funding が本当に net positive か
- HYPE は spread/volatility を加味しても有効か
- basis 縮小の速度と holding time が一致しているか
- funding を取れても execution cost で消えていないか

以上を満たせば、Funding Carry は Hyperliquid 戦略群の中でも最も早く安定した backtest に到達しやすい。
