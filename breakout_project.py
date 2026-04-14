from __future__ import annotations

import argparse
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    import plotly.io as pio

    PLOTLY_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    PLOTLY_AVAILABLE = False

try:
    import shinybroker as sb

    SHINYBROKER_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    SHINYBROKER_AVAILABLE = False

try:
    import yfinance as yf

    YFINANCE_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    YFINANCE_AVAILABLE = False


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "project_outputs"

# -----------------------------
# Strategy configuration
# -----------------------------
HISTORICAL_DURATION = "5 Y"
BAR_SIZE = "1 day"
WHAT_TO_SHOW = "Trades"
USE_RTH = True

IB_HOST = "127.0.0.1"
IB_PORT = 7497
IB_CLIENT_ID = 9999
IB_TIMEOUT = 10

INITIAL_CAPITAL = 100_000.0
RISK_FREE_RATE = 0.02
SLIPPAGE_BPS = 5
COMMISSION_PER_TRADE = 1.00

LOOKBACK_CANDIDATES = [20, 40, 55]
ATR_WINDOW = 14
BREAKOUT_BUFFER = 0.00
STOP_LOSS_ATR_CANDIDATES = [1.5, 2.0]
PROFIT_TARGET_ATR_CANDIDATES = [2.0, 3.0]
TIMEOUT_DAYS = 10

TRAINING_WINDOW_DAYS = 252
TEST_WINDOW_DAYS = 63
MIN_TRAINING_TRADES = 2

DEFAULT_ASSET_UNIVERSE = [
    {"symbol": "MU", "secType": "STK", "exchange": "SMART", "currency": "USD"},
    {"symbol": "AAPL", "secType": "STK", "exchange": "SMART", "currency": "USD"},
    {"symbol": "NVDA", "secType": "STK", "exchange": "SMART", "currency": "USD"},
    {"symbol": "XOM", "secType": "STK", "exchange": "SMART", "currency": "USD"},
    {"symbol": "GLD", "secType": "STK", "exchange": "SMART", "currency": "USD"},
]
@dataclass(frozen=True)
class StrategyParameters:
    lookback: int
    atr_window: int
    breakout_buffer: float
    stop_atr_multiplier: float
    target_atr_multiplier: float
    timeout_days: int


def ensure_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def contract_symbol(contract_spec: dict[str, str]) -> str:
    return contract_spec["symbol"].upper()


def make_stock_contract(symbol: str) -> dict[str, str]:
    return {
        "symbol": symbol.upper(),
        "secType": "STK",
        "exchange": "SMART",
        "currency": "USD",
    }


def cache_path(contract_spec: dict[str, str]) -> Path:
    return DATA_DIR / f"{contract_symbol(contract_spec).lower()}_daily.csv"


def normalize_price_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        raise ValueError("No rows were returned from the data source.")

    renamed = raw_df.copy()
    renamed.columns = [str(col).strip().lower() for col in renamed.columns]

    timestamp_candidates = ["timestamp", "date", "datetime", "time"]
    timestamp_column = next((col for col in timestamp_candidates if col in renamed.columns), None)
    if timestamp_column is None:
        raise ValueError("Expected a timestamp column such as 'timestamp' or 'date'.")

    column_map = {timestamp_column: "timestamp"}
    for name in ["open", "high", "low", "close", "volume"]:
        if name in renamed.columns:
            column_map[name] = name
    renamed = renamed.rename(columns=column_map)

    required = {"timestamp", "open", "high", "low", "close"}
    missing = required.difference(renamed.columns)
    if missing:
        raise ValueError(f"Missing required price columns: {sorted(missing)}")

    normalized = renamed.loc[:, [col for col in ["timestamp", "open", "high", "low", "close", "volume"] if col in renamed.columns]].copy()
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"])
    normalized = normalized.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp")

    for col in ["open", "high", "low", "close", "volume"]:
        if col in normalized.columns:
            normalized[col] = pd.to_numeric(normalized[col], errors="coerce")

    normalized = normalized.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    normalized["source_index"] = np.arange(len(normalized))
    return normalized


def fetch_daily_data(contract_spec: dict[str, str]) -> pd.DataFrame:
    if not SHINYBROKER_AVAILABLE:
        raise RuntimeError("shinybroker is not installed in the active Python environment.")

    # Mirrors the structure the user already set up in docs/pull_data.py.
    asset = sb.Contract(contract_spec)
    response = sb.fetch_historical_data(
        asset,
        endDateTime="",
        durationStr=HISTORICAL_DURATION,
        barSizeSetting=BAR_SIZE,
        whatToShow=WHAT_TO_SHOW,
        useRTH=USE_RTH,
        host=IB_HOST,
        port=IB_PORT,
        client_id=IB_CLIENT_ID,
        timeout=IB_TIMEOUT,
    )
    return normalize_price_data(pd.DataFrame(response["hst_dta"]))


def flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    flattened = df.copy()
    flattened.columns = [
        str(level_0).lower()
        for level_0, _level_1 in flattened.columns.to_flat_index()
    ]
    return flattened


def fetch_daily_data_yfinance(contract_spec: dict[str, str]) -> pd.DataFrame:
    if not YFINANCE_AVAILABLE:
        raise RuntimeError("yfinance is not installed in the active Python environment.")

    symbol = contract_symbol(contract_spec)
    yf_df = yf.download(
        symbol,
        period="5y",
        interval="1d",
        progress=False,
        auto_adjust=False,
    )
    if yf_df.empty:
        raise ValueError(f"No yfinance history returned for {symbol}.")

    yf_df = flatten_yfinance_columns(yf_df).reset_index()
    yf_df = yf_df.rename(columns={"Date": "timestamp", "date": "timestamp"})
    if "adj close" in yf_df.columns:
        yf_df = yf_df.drop(columns=["adj close"])
    return normalize_price_data(yf_df)


def load_or_fetch_prices(contract_spec: dict[str, str], force_download: bool = False) -> pd.DataFrame:
    path = cache_path(contract_spec)
    if path.exists() and not force_download:
        return normalize_price_data(pd.read_csv(path))

    try:
        price_df = fetch_daily_data(contract_spec)
    except Exception as primary_exc:
        try:
            price_df = fetch_daily_data_yfinance(contract_spec)
        except Exception as fallback_exc:
            raise RuntimeError(
                f"ShinyBroker fetch failed ({primary_exc}) and yfinance fallback failed ({fallback_exc})."
            ) from fallback_exc
    price_df.to_csv(path, index=False)
    return price_df


def compute_atr(price_df: pd.DataFrame, window: int) -> pd.Series:
    prev_close = price_df["close"].shift(1)
    true_range = pd.concat(
        [
            price_df["high"] - price_df["low"],
            (price_df["high"] - prev_close).abs(),
            (price_df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(window=window, min_periods=window).mean()


def detect_breakouts(
    price_df: pd.DataFrame,
    lookback: int,
    atr_window: int = ATR_WINDOW,
    breakout_buffer: float = BREAKOUT_BUFFER,
) -> pd.DataFrame:
    """
    Identify breakout bars for a Donchian-style trend strategy.

    A bar is marked as a breakout when the current close is above the highest
    high from the previous `lookback` bars. The rolling high is shifted by one
    bar, so today's close is only compared against information that was known
    before today's session finished.
    """

    annotated = price_df.copy()
    annotated["prior_n_day_high"] = (
        annotated["high"].rolling(window=lookback, min_periods=lookback).max().shift(1)
    )
    annotated["atr"] = compute_atr(annotated, atr_window)
    annotated["breakout_signal"] = annotated["close"] > (
        annotated["prior_n_day_high"] * (1.0 + breakout_buffer)
    )
    return annotated


def classify_trade_outcome(exit_reason: str, pnl: float) -> str:
    if exit_reason == "stop_loss":
        return "Stop-loss triggered"
    if exit_reason == "profit_target" or pnl > 0:
        return "Successful"
    return "Timed out"


def simulate_breakout_strategy(
    price_df: pd.DataFrame,
    params: StrategyParameters,
    starting_cash: float = INITIAL_CAPITAL,
    signal_start_index: int | None = None,
    signal_end_index_exclusive: int | None = None,
) -> tuple[pd.DataFrame, float]:
    working = detect_breakouts(
        price_df.reset_index(drop=True),
        lookback=params.lookback,
        atr_window=params.atr_window,
        breakout_buffer=params.breakout_buffer,
    )

    minimum_history = max(params.lookback, params.atr_window)
    signal_start = max(signal_start_index or 0, minimum_history)
    signal_end = signal_end_index_exclusive or len(working)

    # We skip the last timeout window so every trade can fully resolve inside
    # the test segment without leaking into the next walk-forward block.
    last_signal_index = min(signal_end - 1, len(working) - params.timeout_days - 1)
    if last_signal_index < signal_start:
        return pd.DataFrame(), starting_cash

    cash = starting_cash
    trades: list[dict[str, float | int | str]] = []
    signal_index = signal_start

    while signal_index <= last_signal_index:
        signal_row = working.iloc[signal_index]
        if not signal_row["breakout_signal"] or pd.isna(signal_row["atr"]):
            signal_index += 1
            continue

        entry_index = signal_index + 1
        entry_row = working.iloc[entry_index]
        entry_price = float(entry_row["open"]) * (1.0 + SLIPPAGE_BPS / 10_000.0)
        shares = int((cash - COMMISSION_PER_TRADE) // entry_price)

        if shares < 1:
            signal_index += 1
            continue

        cash_before_trade = cash
        cash_after_entry = cash_before_trade - (shares * entry_price) - COMMISSION_PER_TRADE

        atr_value = float(signal_row["atr"])
        stop_price = entry_price - (params.stop_atr_multiplier * atr_value)
        target_price = entry_price + (params.target_atr_multiplier * atr_value)
        timeout_index = min(entry_index + params.timeout_days - 1, len(working) - 1)

        exit_reason = "timeout"
        exit_index = timeout_index
        exit_price = float(working.iloc[timeout_index]["close"]) * (1.0 - SLIPPAGE_BPS / 10_000.0)

        for idx in range(entry_index, timeout_index + 1):
            day = working.iloc[idx]
            low_hit = float(day["low"]) <= stop_price
            high_hit = float(day["high"]) >= target_price

            if low_hit and high_hit:
                exit_reason = "stop_loss"
                exit_index = idx
                exit_price = stop_price * (1.0 - SLIPPAGE_BPS / 10_000.0)
                break
            if low_hit:
                exit_reason = "stop_loss"
                exit_index = idx
                exit_price = stop_price * (1.0 - SLIPPAGE_BPS / 10_000.0)
                break
            if high_hit:
                exit_reason = "profit_target"
                exit_index = idx
                exit_price = target_price * (1.0 - SLIPPAGE_BPS / 10_000.0)
                break

        cash_after_exit = cash_after_entry + (shares * exit_price) - COMMISSION_PER_TRADE
        pnl = cash_after_exit - cash_before_trade
        gross_exposure = shares * entry_price
        return_pct = pnl / gross_exposure if gross_exposure else 0.0
        outcome = classify_trade_outcome(exit_reason, pnl)

        trades.append(
            {
                "signal_timestamp": signal_row["timestamp"],
                "entry_timestamp": entry_row["timestamp"],
                "exit_timestamp": working.iloc[exit_index]["timestamp"],
                "entry_price": round(entry_price, 4),
                "exit_price": round(exit_price, 4),
                "position_size": shares,
                "direction": "Long",
                "return_pct": return_pct * 100.0,
                "pnl": pnl,
                "holding_days": int(exit_index - entry_index + 1),
                "exit_reason": exit_reason,
                "outcome": outcome,
                "stop_price": round(stop_price, 4),
                "target_price": round(target_price, 4),
                "cash_before_trade": cash_before_trade,
                "cash_after_trade": cash_after_exit,
                "entry_source_index": int(entry_row["source_index"]),
                "exit_source_index": int(working.iloc[exit_index]["source_index"]),
                "lookback": params.lookback,
                "atr_window": params.atr_window,
                "breakout_buffer": params.breakout_buffer,
                "stop_atr_multiplier": params.stop_atr_multiplier,
                "target_atr_multiplier": params.target_atr_multiplier,
                "timeout_days": params.timeout_days,
            }
        )

        cash = cash_after_exit
        signal_index = exit_index + 1

    return pd.DataFrame(trades), cash


def annualized_trade_sharpe(trades_df: pd.DataFrame) -> float:
    if trades_df.empty or len(trades_df) < 2:
        return float("-inf")

    returns = trades_df["return_pct"] / 100.0
    std = returns.std(ddof=1)
    if std == 0 or pd.isna(std):
        return float("-inf")

    average_holding_period = max(float(trades_df["holding_days"].mean()), 1.0)
    trades_per_year = 252.0 / average_holding_period
    return float((returns.mean() / std) * math.sqrt(trades_per_year))


def optimize_parameters(training_df: pd.DataFrame) -> tuple[StrategyParameters, float]:
    best_params = StrategyParameters(
        lookback=LOOKBACK_CANDIDATES[0],
        atr_window=ATR_WINDOW,
        breakout_buffer=BREAKOUT_BUFFER,
        stop_atr_multiplier=STOP_LOSS_ATR_CANDIDATES[0],
        target_atr_multiplier=PROFIT_TARGET_ATR_CANDIDATES[0],
        timeout_days=TIMEOUT_DAYS,
    )
    best_score = float("-inf")

    for lookback in LOOKBACK_CANDIDATES:
        for stop_mult in STOP_LOSS_ATR_CANDIDATES:
            for target_mult in PROFIT_TARGET_ATR_CANDIDATES:
                candidate = StrategyParameters(
                    lookback=lookback,
                    atr_window=ATR_WINDOW,
                    breakout_buffer=BREAKOUT_BUFFER,
                    stop_atr_multiplier=stop_mult,
                    target_atr_multiplier=target_mult,
                    timeout_days=TIMEOUT_DAYS,
                )
                candidate_trades, _ = simulate_breakout_strategy(training_df, candidate)
                if len(candidate_trades) < MIN_TRAINING_TRADES:
                    continue

                score = annualized_trade_sharpe(candidate_trades)
                if score > best_score:
                    best_score = score
                    best_params = candidate

    return best_params, best_score


def run_walk_forward_backtest(
    price_df: pd.DataFrame,
    asset_symbol: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    walk_forward_rows: list[dict[str, float | int | str]] = []
    all_trades: list[pd.DataFrame] = []
    cash = INITIAL_CAPITAL

    max_history = max(max(LOOKBACK_CANDIDATES), ATR_WINDOW)
    if len(price_df) < TRAINING_WINDOW_DAYS + TEST_WINDOW_DAYS + max_history:
        raise ValueError("Not enough daily history for the requested walk-forward windows.")

    window_start = TRAINING_WINDOW_DAYS
    while window_start < len(price_df) - TIMEOUT_DAYS - 1:
        train_start = max(0, window_start - TRAINING_WINDOW_DAYS)
        test_end = min(len(price_df), window_start + TEST_WINDOW_DAYS)

        training_df = price_df.iloc[train_start:window_start].reset_index(drop=True)
        best_params, training_score = optimize_parameters(training_df)

        warmup_start = max(0, window_start - max_history - 1)
        testing_df = price_df.iloc[warmup_start:test_end].reset_index(drop=True)
        signal_start_index = window_start - warmup_start

        trades_df, cash = simulate_breakout_strategy(
            testing_df,
            best_params,
            starting_cash=cash,
            signal_start_index=signal_start_index,
            signal_end_index_exclusive=len(testing_df),
        )

        if not trades_df.empty:
            trades_df["asset"] = asset_symbol
            trades_df["train_start"] = price_df.iloc[train_start]["timestamp"]
            trades_df["train_end"] = price_df.iloc[window_start - 1]["timestamp"]
            trades_df["test_start"] = price_df.iloc[window_start]["timestamp"]
            trades_df["test_end"] = price_df.iloc[test_end - 1]["timestamp"]
            all_trades.append(trades_df)

        walk_forward_rows.append(
            {
                "asset": asset_symbol,
                "train_start": price_df.iloc[train_start]["timestamp"],
                "train_end": price_df.iloc[window_start - 1]["timestamp"],
                "test_start": price_df.iloc[window_start]["timestamp"],
                "test_end": price_df.iloc[test_end - 1]["timestamp"],
                "training_sharpe_score": training_score,
                "selected_lookback": best_params.lookback,
                "selected_stop_atr": best_params.stop_atr_multiplier,
                "selected_target_atr": best_params.target_atr_multiplier,
                "timeout_days": best_params.timeout_days,
                "test_trade_count": 0 if trades_df.empty else len(trades_df),
            }
        )

        window_start = test_end

    combined_trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    walk_forward_df = pd.DataFrame(walk_forward_rows)
    return combined_trades, walk_forward_df


def build_equity_curve(price_df: pd.DataFrame, trades_df: pd.DataFrame) -> pd.DataFrame:
    equity_df = price_df.loc[:, ["timestamp", "close", "source_index"]].copy()
    equity_df["equity"] = INITIAL_CAPITAL

    if trades_df.empty:
        equity_df["daily_return"] = equity_df["equity"].pct_change().fillna(0.0)
        return equity_df

    trades = trades_df.sort_values("entry_source_index").to_dict("records")
    trade_pointer = 0
    cash = INITIAL_CAPITAL

    for idx, row in equity_df.iterrows():
        source_index = int(row["source_index"])

        while trade_pointer < len(trades) and source_index > int(trades[trade_pointer]["exit_source_index"]):
            cash = float(trades[trade_pointer]["cash_after_trade"])
            trade_pointer += 1

        if trade_pointer < len(trades):
            trade = trades[trade_pointer]
            entry_source_index = int(trade["entry_source_index"])
            exit_source_index = int(trade["exit_source_index"])
            position_cash = float(trade["cash_before_trade"]) - (
                float(trade["position_size"]) * float(trade["entry_price"])
            ) - COMMISSION_PER_TRADE

            if source_index < entry_source_index:
                equity_df.at[idx, "equity"] = cash
            elif source_index < exit_source_index:
                equity_df.at[idx, "equity"] = position_cash + (
                    float(trade["position_size"]) * float(row["close"])
                )
            elif source_index == exit_source_index:
                equity_df.at[idx, "equity"] = float(trade["cash_after_trade"])
            else:
                equity_df.at[idx, "equity"] = cash
        else:
            equity_df.at[idx, "equity"] = cash

    equity_df["daily_return"] = equity_df["equity"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return equity_df


def compute_performance_metrics(trades_df: pd.DataFrame, equity_curve_df: pd.DataFrame) -> dict[str, float]:
    if trades_df.empty:
        return {
            "trade_count": 0,
            "average_return_per_trade_pct": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "expectancy_dollars": 0.0,
            "final_equity": INITIAL_CAPITAL,
            "risk_free_rate": RISK_FREE_RATE,
        }

    daily_rf = (1.0 + RISK_FREE_RATE) ** (1.0 / 252.0) - 1.0
    excess_daily_returns = equity_curve_df["daily_return"] - daily_rf
    daily_vol = excess_daily_returns.std(ddof=1)
    downside = excess_daily_returns[excess_daily_returns < 0]
    downside_vol = downside.std(ddof=1)

    running_peak = equity_curve_df["equity"].cummax()
    drawdown = (equity_curve_df["equity"] / running_peak) - 1.0

    gross_profit = trades_df.loc[trades_df["pnl"] > 0, "pnl"].sum()
    gross_loss = trades_df.loc[trades_df["pnl"] < 0, "pnl"].sum()

    sharpe_ratio = 0.0 if pd.isna(daily_vol) or daily_vol == 0 else float((excess_daily_returns.mean() / daily_vol) * math.sqrt(252.0))
    sortino_ratio = 0.0 if pd.isna(downside_vol) or downside_vol == 0 else float((excess_daily_returns.mean() / downside_vol) * math.sqrt(252.0))
    profit_factor = 0.0 if gross_loss == 0 else float(gross_profit / abs(gross_loss))

    return {
        "trade_count": int(len(trades_df)),
        "average_return_per_trade_pct": float(trades_df["return_pct"].mean()),
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "max_drawdown_pct": float(drawdown.min() * 100.0),
        "win_rate_pct": float((trades_df["pnl"] > 0).mean() * 100.0),
        "profit_factor": profit_factor,
        "expectancy_dollars": float(trades_df["pnl"].mean()),
        "final_equity": float(equity_curve_df["equity"].iloc[-1]),
        "risk_free_rate": RISK_FREE_RATE,
    }


def format_metric(value: float, metric_name: str) -> str:
    if metric_name.endswith("_pct") or metric_name in {"average_return_per_trade_pct", "win_rate_pct", "max_drawdown_pct"}:
        return f"{value:,.2f}%"
    if metric_name in {"final_equity", "expectancy_dollars"}:
        return f"${value:,.2f}"
    return f"{value:,.2f}"


def dataframe_records(df: pd.DataFrame, max_rows: int | None = None) -> list[dict[str, object]]:
    if df.empty:
        return []

    working = df.head(max_rows).copy() if max_rows is not None else df.copy()
    for col in working.columns:
        if pd.api.types.is_datetime64_any_dtype(working[col]):
            working[col] = working[col].dt.strftime("%Y-%m-%d")
    working = working.where(pd.notna(working), None)
    return working.to_dict("records")


def build_project_summary_text(symbol: str, price_df: pd.DataFrame) -> str:
    return (
        f"Selected asset: {symbol}. Data span: {price_df['timestamp'].min():%Y-%m-%d} to "
        f"{price_df['timestamp'].max():%Y-%m-%d}. Timeout period: {TIMEOUT_DAYS} trading days. "
        f"Stop-loss logic: entry price - ATR x selected stop multiplier. Profit-target logic: "
        f"entry price + ATR x selected target multiplier. Execution assumptions: next-day open "
        f"entry, {SLIPPAGE_BPS} bps slippage each side, ${COMMISSION_PER_TRADE:.2f} commission per side."
    )


def build_asset_dashboard_payload(
    selected_symbol: str,
    screening_df: pd.DataFrame,
    asset_runs: dict[str, dict[str, object]],
) -> dict[str, object]:
    screening_clean = screening_df.copy()
    screening_clean = screening_clean.where(pd.notna(screening_clean), None)

    dashboard_assets: dict[str, dict[str, object]] = {}
    for symbol, run in asset_runs.items():
        price_df = pd.DataFrame(run["price_df"])
        trades_df = pd.DataFrame(run["trades_df"])
        walk_forward_df = pd.DataFrame(run["walk_forward_df"])
        equity_curve_df = pd.DataFrame(run["equity_curve_df"])
        metrics = dict(run["metrics"])

        outcome_counts = (
            trades_df["outcome"].value_counts().reindex(
                ["Successful", "Timed out", "Stop-loss triggered"], fill_value=0
            ).to_dict()
            if not trades_df.empty
            else {"Successful": 0, "Timed out": 0, "Stop-loss triggered": 0}
        )

        dashboard_assets[symbol] = {
            "symbol": symbol,
            "data_start": f"{price_df['timestamp'].min():%Y-%m-%d}",
            "data_end": f"{price_df['timestamp'].max():%Y-%m-%d}",
            "metrics": metrics,
            "outcome_counts": outcome_counts,
            "trade_blotter_preview": dataframe_records(
                trades_df.loc[
                    :,
                    [
                        "entry_timestamp",
                        "exit_timestamp",
                        "entry_price",
                        "exit_price",
                        "position_size",
                        "direction",
                        "return_pct",
                        "outcome",
                    ],
                ],
                max_rows=20,
            ),
            "equity_curve": dataframe_records(
                equity_curve_df.loc[:, ["timestamp", "equity"]],
            ),
            "walk_forward_windows": dataframe_records(walk_forward_df),
            "trade_blotter_csv": f"project_outputs/assets/{symbol.lower()}_trade_blotter.csv",
            "project_summary_text": build_project_summary_text(symbol, price_df),
        }

    return {
        "selected_symbol": selected_symbol,
        "screening_summary": screening_clean.to_dict("records"),
        "assets": dashboard_assets,
        "strategy_parameters": {
            "lookback_candidates": LOOKBACK_CANDIDATES,
            "atr_window": ATR_WINDOW,
            "breakout_buffer": BREAKOUT_BUFFER,
            "stop_loss_atr_candidates": STOP_LOSS_ATR_CANDIDATES,
            "profit_target_atr_candidates": PROFIT_TARGET_ATR_CANDIDATES,
            "timeout_days": TIMEOUT_DAYS,
            "training_window_days": TRAINING_WINDOW_DAYS,
            "test_window_days": TEST_WINDOW_DAYS,
        },
    }


def render_metrics_cards(metrics: dict[str, float]) -> str:
    descriptions = {
        "average_return_per_trade_pct": "Average percentage gain or loss generated by each completed trade.",
        "sharpe_ratio": "Annualized risk-adjusted return using daily equity changes and the stated risk-free rate.",
        "sortino_ratio": "Risk-adjusted return that penalizes downside volatility more than upside swings.",
        "max_drawdown_pct": "Largest peak-to-trough decline in the backtest equity curve.",
        "win_rate_pct": "Share of trades that finished with a positive dollar PnL.",
        "profit_factor": "Gross profits divided by gross losses across all completed trades.",
        "expectancy_dollars": "Average dollar PnL per trade after slippage and commissions.",
        "final_equity": "Ending account value after compounding all walk-forward trades.",
    }

    order = [
        "average_return_per_trade_pct",
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown_pct",
        "win_rate_pct",
        "profit_factor",
        "expectancy_dollars",
        "final_equity",
    ]

    cards = []
    for key in order:
        label = key.replace("_", " ").title()
        value = format_metric(float(metrics[key]), key)
        cards.append(
            f"""
            <div class="metric-card">
              <div class="metric-label">{html.escape(label)}</div>
              <div class="metric-value">{html.escape(value)}</div>
              <p>{html.escape(descriptions[key])}</p>
            </div>
            """
        )

    rf_text = f"Risk-free rate assumption: {metrics['risk_free_rate'] * 100:.2f}%"
    return f'<div class="metric-grid">{"".join(cards)}</div><p class="metric-footnote">{html.escape(rf_text)}</p>'


def render_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return '<div class="placeholder-box"><p>No rows are available yet.</p></div>'

    preview = df.head(max_rows) if max_rows is not None else df
    table_df = preview.copy()
    for col in table_df.columns:
        if pd.api.types.is_datetime64_any_dtype(table_df[col]):
            table_df[col] = table_df[col].dt.strftime("%Y-%m-%d")
    return table_df.to_html(index=False, classes=["table", "table-striped", "table-sm"], border=0)


def render_plotly_figure(fig: go.Figure, title: str) -> str:
    if not PLOTLY_AVAILABLE:
        return f'<div class="placeholder-box"><p>{html.escape(title)} could not be rendered because Plotly is unavailable.</p></div>'
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20), template="plotly_white")
    return pio.to_html(fig, full_html=False, include_plotlyjs="cdn", config={"displayModeBar": False, "responsive": True})


def render_outcome_chart(trades_df: pd.DataFrame) -> str:
    if trades_df.empty:
        return '<div class="placeholder-box"><p>No trades were generated, so the outcome histogram is empty.</p></div>'

    counts = trades_df["outcome"].value_counts().reindex(
        ["Successful", "Timed out", "Stop-loss triggered"], fill_value=0
    )
    if not PLOTLY_AVAILABLE:
        return render_table(counts.rename_axis("outcome").reset_index(name="trade_count"))

    fig = go.Figure(
        data=[
            go.Bar(
                x=counts.index.tolist(),
                y=counts.values.tolist(),
                marker_color=["#2a9d8f", "#e9c46a", "#d62828"],
            )
        ]
    )
    fig.update_layout(title="Trade Outcome Analysis", xaxis_title="Outcome", yaxis_title="Trade count")
    return render_plotly_figure(fig, "Trade outcome chart")


def render_equity_curve(equity_curve_df: pd.DataFrame) -> str:
    if equity_curve_df.empty:
        return '<div class="placeholder-box"><p>The equity curve will appear here after the backtest runs.</p></div>'

    if not PLOTLY_AVAILABLE:
        return render_table(equity_curve_df.loc[:, ["timestamp", "equity"]], max_rows=20)

    fig = go.Figure(
        data=[
            go.Scatter(
                x=equity_curve_df["timestamp"],
                y=equity_curve_df["equity"],
                mode="lines",
                line=dict(color="#0b3d91", width=3),
                name="Equity",
            )
        ]
    )
    fig.update_layout(title="Walk-Forward Equity Curve", xaxis_title="Date", yaxis_title="Equity ($)")
    return render_plotly_figure(fig, "Equity curve")


def build_asset_selection_markdown(selected_symbol: str, screening_df: pd.DataFrame) -> str:
    asset_count = len(screening_df)
    selected_row = screening_df.loc[screening_df["asset"] == selected_symbol].iloc[0]
    trade_count = int(selected_row["trade_count"])
    sharpe = float(selected_row["sharpe_ratio"])
    average_return = float(selected_row["average_return_per_trade_pct"])

    return (
        f"I screened {asset_count} liquid U.S. assets using the same daily breakout logic and "
        f"selected **{selected_symbol}** for the final write-up because it produced the strongest "
        f"out-of-sample mix of trade frequency and risk-adjusted performance. In the walk-forward test, "
        f"{selected_symbol} generated {trade_count} completed trades with an annualized Sharpe ratio of "
        f"{sharpe:.2f} and an average return per trade of {average_return:.2f}%. "
        "To force a different single asset, rerun the pipeline with "
        "`../.venv-1/bin/python breakout_project.py --asset SYMBOL --force-download`. "
        "To screen a custom universe, use `--symbols SYMBOL1 SYMBOL2 ...`."
    )


def write_outputs(
    selected_symbol: str,
    price_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    walk_forward_df: pd.DataFrame,
    equity_curve_df: pd.DataFrame,
    metrics: dict[str, float],
    screening_df: pd.DataFrame,
    asset_runs: dict[str, dict[str, object]],
) -> None:
    asset_output_dir = OUTPUT_DIR / "assets"
    asset_output_dir.mkdir(parents=True, exist_ok=True)

    trades_df.to_csv(OUTPUT_DIR / "trade_blotter.csv", index=False)
    screening_df.to_csv(OUTPUT_DIR / "screening_results.csv", index=False)
    walk_forward_df.to_csv(OUTPUT_DIR / "walk_forward_windows.csv", index=False)
    equity_curve_df.to_csv(OUTPUT_DIR / "equity_curve.csv", index=False)
    pd.DataFrame([metrics]).to_csv(OUTPUT_DIR / "performance_metrics.csv", index=False)

    (OUTPUT_DIR / "asset_selection.md").write_text(
        build_asset_selection_markdown(selected_symbol, screening_df),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "project_summary.md").write_text(
        (
            f"Selected asset: **{selected_symbol}**  \n"
            f"Data span: **{price_df['timestamp'].min():%Y-%m-%d}** to **{price_df['timestamp'].max():%Y-%m-%d}**  \n"
            f"Timeout period: **{TIMEOUT_DAYS} trading days**  \n"
            f"Stop-loss logic: **entry price - ATR x selected stop multiplier**  \n"
            f"Profit-target logic: **entry price + ATR x selected target multiplier**  \n"
            f"Execution assumptions: **next-day open entry, {SLIPPAGE_BPS} bps slippage each side, "
            f"${COMMISSION_PER_TRADE:.2f} commission per side**."
        ),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "metrics_cards.html").write_text(render_metrics_cards(metrics), encoding="utf-8")
    (OUTPUT_DIR / "trade_blotter_preview.html").write_text(
        render_table(
            trades_df.loc[
                :,
                [
                    "entry_timestamp",
                    "exit_timestamp",
                    "entry_price",
                    "exit_price",
                    "position_size",
                    "direction",
                    "return_pct",
                    "outcome",
                ],
            ],
            max_rows=20,
        ),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "walk_forward_table.html").write_text(render_table(walk_forward_df, max_rows=20), encoding="utf-8")
    (OUTPUT_DIR / "outcome_chart.html").write_text(render_outcome_chart(trades_df), encoding="utf-8")
    (OUTPUT_DIR / "equity_curve.html").write_text(render_equity_curve(equity_curve_df), encoding="utf-8")
    (OUTPUT_DIR / "run_metadata.json").write_text(
        json.dumps(
            {
                "selected_symbol": selected_symbol,
                "metrics": metrics,
                "strategy_parameters": {
                    "lookback_candidates": LOOKBACK_CANDIDATES,
                    "atr_window": ATR_WINDOW,
                    "breakout_buffer": BREAKOUT_BUFFER,
                    "stop_loss_atr_candidates": STOP_LOSS_ATR_CANDIDATES,
                    "profit_target_atr_candidates": PROFIT_TARGET_ATR_CANDIDATES,
                    "timeout_days": TIMEOUT_DAYS,
                    "training_window_days": TRAINING_WINDOW_DAYS,
                    "test_window_days": TEST_WINDOW_DAYS,
                },
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    for symbol, run in asset_runs.items():
        pd.DataFrame(run["trades_df"]).to_csv(asset_output_dir / f"{symbol.lower()}_trade_blotter.csv", index=False)

    (OUTPUT_DIR / "asset_dashboard.json").write_text(
        json.dumps(
            build_asset_dashboard_payload(selected_symbol, screening_df, asset_runs),
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


def write_placeholder_outputs(reason: str) -> None:
    ensure_directories()

    placeholder_csv = pd.DataFrame(
        columns=[
            "entry_timestamp",
            "exit_timestamp",
            "entry_price",
            "exit_price",
            "position_size",
            "direction",
            "return_pct",
            "outcome",
        ]
    )
    placeholder_csv.to_csv(OUTPUT_DIR / "trade_blotter.csv", index=False)
    pd.DataFrame(columns=["asset", "trade_count", "average_return_per_trade_pct", "sharpe_ratio"]).to_csv(
        OUTPUT_DIR / "screening_results.csv",
        index=False,
    )
    pd.DataFrame(columns=["timestamp", "equity"]).to_csv(OUTPUT_DIR / "equity_curve.csv", index=False)
    pd.DataFrame(columns=["asset", "train_start", "train_end", "test_start", "test_end"]).to_csv(
        OUTPUT_DIR / "walk_forward_windows.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "trade_count": 0,
                "average_return_per_trade_pct": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown_pct": 0.0,
                "win_rate_pct": 0.0,
                "profit_factor": 0.0,
                "expectancy_dollars": 0.0,
                "final_equity": INITIAL_CAPITAL,
                "risk_free_rate": RISK_FREE_RATE,
            }
        ]
    ).to_csv(OUTPUT_DIR / "performance_metrics.csv", index=False)

    placeholder_box = (
        f'<div class="placeholder-box"><p>{html.escape(reason)}</p>'
        '<p>Start IB Gateway/TWS, then run '
        '<code>../.venv-1/bin/python breakout_project.py --force-download</code> '
        'from the project root and re-render the site with <code>quarto render</code>.</p></div>'
    )
    (OUTPUT_DIR / "asset_selection.md").write_text(
        "Asset selection will populate after the screening step completes against live or cached historical data.",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "project_summary.md").write_text(
        (
            f"Project outputs are in placeholder mode because the data pipeline could not run.  \n"
            f"Reason: {reason}"
        ),
        encoding="utf-8",
    )
    for filename in [
        "metrics_cards.html",
        "trade_blotter_preview.html",
        "walk_forward_table.html",
        "outcome_chart.html",
        "equity_curve.html",
    ]:
        (OUTPUT_DIR / filename).write_text(placeholder_box, encoding="utf-8")

    (OUTPUT_DIR / "run_metadata.json").write_text(
        json.dumps({"status": "placeholder", "reason": reason}, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "asset_dashboard.json").write_text(
        json.dumps({"status": "placeholder", "reason": reason, "assets": {}}, indent=2),
        encoding="utf-8",
    )


def summarize_asset_run(asset_symbol: str, trades_df: pd.DataFrame, equity_curve_df: pd.DataFrame) -> dict[str, float | str]:
    metrics = compute_performance_metrics(trades_df, equity_curve_df)
    return {
        "asset": asset_symbol,
        "trade_count": metrics["trade_count"],
        "average_return_per_trade_pct": metrics["average_return_per_trade_pct"],
        "sharpe_ratio": metrics["sharpe_ratio"],
        "win_rate_pct": metrics["win_rate_pct"],
        "max_drawdown_pct": metrics["max_drawdown_pct"],
        "final_equity": metrics["final_equity"],
    }


def screen_assets(
    asset_universe: Iterable[dict[str, str]],
    force_download: bool = False,
) -> tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float], pd.DataFrame, dict[str, dict[str, object]]]:
    screening_rows: list[dict[str, float | str]] = []
    best_run: dict[str, object] | None = None
    asset_runs: dict[str, dict[str, object]] = {}

    for contract_spec in asset_universe:
        symbol = contract_symbol(contract_spec)
        try:
            price_df = load_or_fetch_prices(contract_spec, force_download=force_download)
            trades_df, walk_forward_df = run_walk_forward_backtest(price_df, symbol)
            equity_curve_df = build_equity_curve(price_df, trades_df)
            metrics = compute_performance_metrics(trades_df, equity_curve_df)
            screening_rows.append(summarize_asset_run(symbol, trades_df, equity_curve_df))
            asset_runs[symbol] = {
                "price_df": price_df,
                "trades_df": trades_df,
                "walk_forward_df": walk_forward_df,
                "equity_curve_df": equity_curve_df,
                "metrics": metrics,
            }

            candidate_score = (
                metrics["trade_count"] > 0,
                metrics["sharpe_ratio"],
                metrics["average_return_per_trade_pct"],
            )
            if best_run is None or candidate_score > best_run["score"]:
                best_run = {
                    "score": candidate_score,
                    "symbol": symbol,
                    "price_df": price_df,
                    "trades_df": trades_df,
                    "walk_forward_df": walk_forward_df,
                    "equity_curve_df": equity_curve_df,
                    "metrics": metrics,
                }
        except Exception as exc:  # pragma: no cover - depends on external data access
            screening_rows.append(
                {
                    "asset": symbol,
                    "trade_count": 0,
                    "average_return_per_trade_pct": 0.0,
                    "sharpe_ratio": 0.0,
                    "win_rate_pct": 0.0,
                    "max_drawdown_pct": 0.0,
                    "final_equity": INITIAL_CAPITAL,
                    "error": str(exc),
                }
            )

    screening_df = pd.DataFrame(screening_rows)
    if best_run is None or pd.DataFrame(best_run["trades_df"]).empty:
        raise RuntimeError("No asset produced a completed trade set. Check IB connectivity or cached data files.")

    return (
        str(best_run["symbol"]),
        pd.DataFrame(best_run["price_df"]),
        pd.DataFrame(best_run["trades_df"]),
        pd.DataFrame(best_run["walk_forward_df"]),
        pd.DataFrame(best_run["equity_curve_df"]),
        dict(best_run["metrics"]),
        screening_df,
        asset_runs,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the breakout strategy project pipeline.")
    parser.add_argument(
        "--asset",
        help="Run the project on one explicit symbol, such as MU or NVDA.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Ignore cached CSV files in data/ and fetch fresh historical data.",
    )
    parser.add_argument(
        "--symbols",
        nargs="*",
        help="Optional list of symbols to screen instead of the default asset universe.",
    )
    return parser.parse_args()


def build_asset_universe(asset: str | None, symbols: list[str] | None) -> list[dict[str, str]]:
    if asset:
        return [make_stock_contract(asset)]
    if not symbols:
        return DEFAULT_ASSET_UNIVERSE
    return [make_stock_contract(symbol) for symbol in symbols]


def main() -> None:
    ensure_directories()
    args = parse_args()
    asset_universe = build_asset_universe(args.asset, args.symbols)

    try:
        selected_symbol, price_df, trades_df, walk_forward_df, equity_curve_df, metrics, screening_df, asset_runs = screen_assets(
            asset_universe,
            force_download=args.force_download,
        )
        write_outputs(
            selected_symbol,
            price_df,
            trades_df,
            walk_forward_df,
            equity_curve_df,
            metrics,
            screening_df,
            asset_runs,
        )
        print(f"Completed breakout project pipeline for {selected_symbol}.")
        print(f"Trade blotter saved to {OUTPUT_DIR / 'trade_blotter.csv'}")
    except Exception as exc:  # pragma: no cover - depends on external data access
        write_placeholder_outputs(str(exc))
        print("Pipeline fell back to placeholder outputs.")
        print(exc)


if __name__ == "__main__":
    main()
