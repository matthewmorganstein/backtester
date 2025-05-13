"""Backtesting engine for trading signals using Pointland and Sphere strategies."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from asyncpg.exceptions import PostgresError
from backtest_dao import BacktestDAO
from dao import PostgresDAO
from settings import Settings

logger = logging.getLogger(__name__)

# Constants
INITIAL_CASH = 100_000
TARGET_BARS = 1000
TIME_ABOVE_STOP_THRESHOLD = 30
DEFAULT_DATE_RANGE_DAYS = 30
BUY_SIGNAL = 1
SELL_SIGNAL = -1
HOLD_SIGNAL = 0
DATE_FORMAT = "%Y-%m-%d"
COL_TIMESTAMP = "timestamp"
COL_CLOSE = "close"
COL_HIGH = "high"
COL_LOW = "low"
COL_R1 = "r_1"
COL_R2 = "r_2"
COLUMNS_REQUIRED = [COL_TIMESTAMP, COL_CLOSE, COL_HIGH, COL_LOW, COL_R1, COL_R2]

class TradeManager:
    """Manages portfolio updates for backtesting."""
    def __init__(self, initial_cash: float = INITIAL_CASH) -> None:
        self.cash: float = initial_cash
        self.portfolio_value: float = initial_cash
        self.position: int = 0
        self.trades: List[Dict[str, Any]] = []
        self.entry_price: Optional[float] = None
        self.entry_idx: Optional[int] = None

    def reset(self) -> None:
        self.cash = INITIAL_CASH
        self.portfolio_value = self.cash
        self.position = 0
        self.trades = []
        self.entry_price = None
        self.entry_idx = None

    def enter_position(self, signal: int, price: float, timestamp: datetime) -> Dict[str, Any]:
        if signal == BUY_SIGNAL and self.cash < price:
            logger.warning("Insufficient cash for buy at %s", timestamp)
            return {}
        self.position = signal
        self.entry_price = price
        self.cash -= price if signal == BUY_SIGNAL else -price
        trade = {
            "action": "buy" if signal == BUY_SIGNAL else "sell",
            "price": price,
            "timestamp": timestamp,
            "profit": 0,
        }
        self.trades.append(trade)
        return trade

    def exit_position(self, exit_result: Dict[str, Any]) -> Dict[str, Any]:
        exit_price = exit_result["exit_price"]
        profit = exit_result["profit"]
        self.cash += exit_price if self.position == BUY_SIGNAL else -exit_price
        trade = {
            "action": "sell" if self.position == BUY_SIGNAL else "buy",
            "price": exit_price,
            "timestamp": exit_result["exit_time"],
            "profit": profit,
            "success": exit_result["success"],
            "failure": exit_result["failure"],
            "time_above_stop": exit_result["time_above_stop"],
        }
        self.trades.append(trade)
        self.position = 0
        self.entry_price = None
        self.entry_idx = None
        return trade

    def update_portfolio(self, price: float) -> Dict[str, Any]:
        self.portfolio_value = self.cash + (self.position * price)
        return {
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "position": self.position,
            "position_size": abs(self.position),
            "position_value": self.position * price,
        }

@dataclass
class Trade:
    """Encapsulates a single trade's state and exit logic."""
    entry_idx: int
    entry_time: datetime
    entry_price: float
    direction: str
    target: float
    stop: float
    square_threshold: float

    def check_exit(
        self,
        future_data: pd.DataFrame,
        opposing_signal: pd.Series,
        target_bars: int,
        bar_duration_minutes: float,
    ) -> Dict[str, Any]:
        if future_data.empty:
            logger.warning(f"No future data for trade at {self.entry_time}")
            return {
                "success": False,
                "failure": True,
                "profit": 0.0,
                "exit_price": self.entry_price,
                "exit_time": self.entry_time,
                "time_above_stop": 0.0,
            }

        stop_condition = (
            future_data[COL_CLOSE] <= self.stop if self.direction == "buy"
            else future_data[COL_CLOSE] >= self.stop
        )
        target_condition = (
            future_data[COL_CLOSE] >= self.target if self.direction == "buy"
            else future_data[COL_CLOSE] <= self.target
        )
        max_duration = future_data.index <= future_data.index[0] + target_bars
        exit_conditions = (stop_condition | target_condition | opposing_signal) & max_duration

        if exit_conditions.any():
            exit_idx = exit_conditions.idxmax()
            exit_row = future_data.loc[exit_idx]
            success = target_condition.loc[exit_idx]
            failure = stop_condition.loc[exit_idx] or opposing_signal.loc[exit_idx]
        else:
            exit_idx = future_data.index[-1]
            exit_row = future_data.loc[exit_idx]
            success = False
            failure = stop_condition.loc[exit_idx]

        time_above_stop = self._calculate_time_above_stop(future_data.loc[:exit_idx], bar_duration_minutes)
        profit = (
            exit_row[COL_CLOSE] - self.entry_price if self.direction == "buy"
            else self.entry_price - exit_row[COL_CLOSE]
        )
        logger.debug(f"Trade outcome: success={success}, failure={failure}, profit={profit:.2f}")
        return {
            "success": success,
            "failure": failure,
            "profit": profit,
            "exit_price": exit_row[COL_CLOSE],
            "exit_time": exit_row[COL_TIMESTAMP],
            "time_above_stop": time_above_stop,
        }

    def _calculate_time_above_stop(self, future_data: pd.DataFrame, bar_duration_minutes: float) -> float:
        if future_data.empty:
            return 0.0
        above_stop = (
            future_data[COL_CLOSE] > self.stop if self.direction == "buy"
            else future_data[COL_CLOSE] < self.stop
        )
        if not above_stop.any():
            return 0.0
        time_above_stop = above_stop.sum() * bar_duration_minutes
        logger.debug(f"Time above stop: {time_above_stop:.2f} minutes")
        return time_above_stop

class SignalBacktester:
    """Backtests trading signals using Pointland and Sphere strategies."""
    def __init__(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        square_threshold: float = 350.0,
        distance_threshold: float = 0.01,
        bar_duration_minutes: float = 1.0,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or Settings()
        self.start_date, self.end_date = self._resolve_date_range(start_date, end_date)
        self.square_threshold = square_threshold
        self.distance_threshold = distance_threshold
        self.bar_duration_minutes = bar_duration_minutes
        self.trade_manager = TradeManager()
        self.dao = PostgresDAO(self.settings)
        self.backtest_dao = BacktestDAO(self.settings)
        self.target_bars = TARGET_BARS
        if not -200 <= square_threshold <= 650:
            raise ValueError("square_threshold must be between -200 and 650")
        if not 0 < distance_threshold < 1:
            raise ValueError("distance_threshold must be between 0 and 1")
        if bar_duration_minutes <= 0:
            raise ValueError("bar_duration_minutes must be positive")
        logger.info(
            f"SignalBacktester initialized: {self.start_date} to {self.end_date}, "
            f"square_threshold={square_threshold}, distance_threshold={distance_threshold}"
        )

    def _resolve_date_range(self, start_date: Optional[str], end_date: Optional[str]) -> Tuple[str, str]:
        try:
            end_dt = (
                datetime.strptime(end_date, DATE_FORMAT).replace(tzinfo=timezone.utc)
                if end_date else datetime.now(timezone.utc)
            )
            start_dt = (
                datetime.strptime(start_date, DATE_FORMAT).replace(tzinfo=timezone.utc)
                if start_date else end_dt - timedelta(days=DEFAULT_DATE_RANGE_DAYS)
            )
            if start_dt >= end_dt:
                raise ValueError("start_date must be before end_date")
            if (end_dt - start_dt).days > self.settings.MAX_DATE_RANGE_DAYS:
                raise ValueError(
                    f"Date range exceeds {self.settings.MAX_DATE_RANGE_DAYS} days"
                )
            return start_dt.strftime(DATE_FORMAT), end_dt.strftime(DATE_FORMAT)
        except ValueError as e:
            logger.error(f"Invalid date: {e}")
            raise

    async def _fetch_data(self) -> pd.DataFrame:
        try:
            df = await self.dao.get_data(self.start_date, self.end_date)
            if df.empty:
                logger.error(f"No data for {self.start_date} to {self.end_date}")
                raise ValueError("Empty DataFrame from DAO")
            return df
        except PostgresError as e:
            logger.exception("Failed to connect to database")
            raise ValueError("Database error") from e

    def _validate_and_clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        missing_cols = [col for col in COLUMNS_REQUIRED if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing columns: {missing_cols}")
            raise ValueError(f"Missing columns: {missing_cols}")

        df = df.copy()
        df[COL_TIMESTAMP] = pd.to_datetime(df[COL_TIMESTAMP], errors="coerce", utc=True)
        numeric_cols = [COL_CLOSE, COL_HIGH, COL_LOW, COL_R1, COL_R2]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        original_rows = len(df)
        df = df.dropna(subset=COLUMNS_REQUIRED)
        df = df[(df[COL_CLOSE] > 0) & (df[COL_HIGH] > 0) & (df[COL_LOW] > 0) & (df[COL_HIGH] >= df[COL_LOW])]
        df = df[df[COL_R1].between(-200, 650) & df[COL_R2].between(-200, 650)]

        dropped_rows = original_rows - len(df)
        if dropped_rows > 0:
            logger.warning(f"Dropped {dropped_rows} rows due to NaN or invalid data")
        if df.empty:
            logger.error("No valid data after cleaning")
            raise ValueError("Empty DataFrame after cleaning")

        time_diffs = df[COL_TIMESTAMP].diff().dropna()
        if not time_diffs.empty and not (time_diffs == time_diffs.iloc[0]).all():
            logger.warning("Inconsistent time intervals in data")

        logger.info(
            f"Data range: {df[COL_TIMESTAMP].min()} to {df[COL_TIMESTAMP].max()}, "
            f"rows: {len(df)}, signals: {len(df[(df[COL_R1] > self.square_threshold) | (df[COL_R2] > self.square_threshold)])}"
        )
        return df.sort_values(COL_TIMESTAMP).reset_index(drop=True)

    async def load_data(self) -> pd.DataFrame:
        df = await self._fetch_data()
        return self._validate_and_clean_data(df)

    def pointland_signal(self, df: pd.DataFrame) -> pd.Series:
        prev_low = df[COL_LOW].shift(1)
        prev_high = df[COL_HIGH].shift(1)
        buy_signal = (df[COL_CLOSE] < prev_low) & (
            (df[COL_R1] > self.square_threshold) | (df[COL_R2] > self.square_threshold)
        )
        sell_signal = (df[COL_CLOSE] > prev_high) & (
            (df[COL_R1] > self.square_threshold) | (df[COL_R2] > self.square_threshold)
        )
        signals = pd.Series(HOLD_SIGNAL, index=df.index, dtype=int)
        signals[buy_signal] = BUY_SIGNAL
        signals[sell_signal] = SELL_SIGNAL
        return signals

    def _calculate_target_and_stop(
        self, direction: str, signal_high: float, signal_low: float
    ) -> Tuple[float, float]:
        if direction not in ["buy", "sell"]:
            raise ValueError(f"Invalid direction: {direction}")
        if signal_high <= 0 or signal_low <= 0:
            raise ValueError("Signal high and low must be positive")
        if direction == "buy":
            target = signal_high * (1 + self.distance_threshold)
            stop = signal_low * (1 - self.distance_threshold)
        else:
            target = signal_low * (1 - self.distance_threshold)
            stop = signal_high * (1 + self.distance_threshold)
        logger.debug(f"Calculated target={target:.2f}, stop={stop:.2f} for {direction}")
        return target, stop

    async def run_backtest(self) -> List[Dict[str, Any]]:
        df = await self.load_data()
        signals = self.pointland_signal(df)
        trades = []
        backtest_id = str(hash(f"{self.start_date}{self.end_date}{self.square_threshold}"))

        for idx, signal in signals[signals != HOLD_SIGNAL].items():
            if self.trade_manager.position != 0:
                continue  # Skip if already in a position
            entry_data = df.iloc[idx]
            direction = "buy" if signal == BUY_SIGNAL else "sell"
            trade_entry = self.trade_manager.enter_position(signal, entry_data[COL_CLOSE], entry_data[COL_TIMESTAMP])
            if not trade_entry:
                continue

            target, stop = self._calculate_target_and_stop(direction, entry_data[COL_HIGH], entry_data[COL_LOW])
            trade = Trade(
                entry_idx=idx,
                entry_time=entry_data[COL_TIMESTAMP],
                entry_price=entry_data[COL_CLOSE],
                direction=direction,
                target=target,
                stop=stop,
                square_threshold=self.square_threshold,
            )
            await self.backtest_dao.save_signal_event(backtest_id, {
                "trade_id": str(idx),
                "timestamp": entry_data[COL_TIMESTAMP],
                "event_type": "entry",
                "price": entry_data[COL_CLOSE],
                "details": {"direction": direction},
            })

            future_data = df.iloc[idx + 1:]
            opposing_signal = self.pointland_signal(future_data).shift(1).fillna(0) == (
                SELL_SIGNAL if direction == "buy" else BUY_SIGNAL
            )
            exit_result = trade.check_exit(future_data, opposing_signal, self.target_bars, self.bar_duration_minutes)
            trade_exit = self.trade_manager.exit_position(exit_result)
            trades.append({**trade_entry, **trade_exit, **exit_result})

            await self.backtest_dao.save_signal_event(backtest_id, {
                "trade_id": str(idx),
                "timestamp": exit_result["exit_time"],
                "event_type": "exit",
                "price": exit_result["exit_price"],
                "details": {
                    "success": exit_result["success"],
                    "profit": exit_result["profit"],
                    "time_above_stop": exit_result["time_above_stop"],
                },
            })

            # Save portfolio update
            portfolio_update = self.trade_manager.update_portfolio(exit_result["exit_price"])
            await self.backtest_dao.save_portfolio_update(backtest_id, {
                "timestamp": exit_result["exit_time"],
                "portfolio_value": portfolio_update["portfolio_value"],
                "cash": portfolio_update["cash"],
                "position_size": portfolio_update["position_size"],
                "position_value": portfolio_update["position_value"],
                "details": {},
            })

        # Save backtest result
        final_portfolio = self.trade_manager.update_portfolio(df[COL_CLOSE].iloc[-1])
        await self.backtest_dao.save_backtest_result({
            "backtest_id": backtest_id,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "final_portfolio_value": final_portfolio["portfolio_value"],
            "returns": (final_portfolio["portfolio_value"] - INITIAL_CASH) / INITIAL_CASH,
            "parameters": {
                "square_threshold": self.square_threshold,
                "distance_threshold": self.distance_threshold,
            },
        })
        return trades

    def polygon_profit(self) -> Dict[str, Any]:
        valid_trades = [
            t for t in self.trade_manager.trades if "profit" in t and t.get("price") is not None
        ]
        last_15 = valid_trades[-15:] if len(valid_trades) >= 15 else valid_trades
        if not last_15:
            return {"win_rate": 0.0, "total_profit": 0.0, "num_trades": 0, "success_rate_time": 0.0}
        wins = sum(1 for trade in last_15 if trade["profit"] > 0)
        time_successes = sum(
            1 for trade in last_15 if trade.get("time_above_stop", 0) > TIME_ABOVE_STOP_THRESHOLD
        )
        return {
            "win_rate": wins / len(last_15),
            "total_profit": sum(trade["profit"] for trade in last_15),
            "num_trades": len(last_15),
            "success_rate_time": time_successes / len(last_15),
        }
