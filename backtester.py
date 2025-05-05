"""Backtesting engine for trading signals using Pointland and Sphere strategies."""
from __future__ import annotations
from datetime import datetime, timedelta
import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from asyncpg.exceptions import PostgresError
from backtest_dao import BacktestDAO
from dao import PostgresDAO

logger = logging.getLogger(__name__)

INITIAL_CASH = 100_000
TARGET_BARS = 1000
TIME_ABOVE_STOP_THRESHOLD = 30

class TradeManager:
    """Manages trading positions and portfolio updates."""

    def __init__(self, initial_cash: float = INITIAL_CASH) -> None:
        """Initialize the TradeManager with starting cash.

        Args:
            initial_cash: Initial cash balance for trading.
        """
        self.cash: float = initial_cash
        self.portfolio_value: float = initial_cash
        self.position: int = 0
        self.trades: list[Dict[str, Any]] = []
        self.entry_price: Optional[float] = None
        self.entry_idx: Optional[int] = None

    def reset(self) -> None:
        """Reset the trade manager to initial state."""
        self.cash = INITIAL_CASH
        self.portfolio_value = self.cash
        self.position = 0
        self.trades = []
        self.entry_price = None
        self.entry_idx = None

    def enter_position(self, signal: int, price: float, timestamp: datetime) -> Dict[str, Any]:
        """Enter a trading position based on a signal.

        Args:
            signal: 1 for buy, -1 for sell.
            price: Entry price.
            timestamp: Timestamp of the trade.

        Returns:
            Dictionary containing trade details or empty dict if insufficient cash.
        """
        if signal == 1 and self.cash < price:
            logger.warning("Insufficient cash for buy at %s", timestamp)
            return {}
        self.position = signal
        self.entry_price = price
        self.cash -= price if signal == 1 else -price
        trade = {"action": "buy" if signal == 1 else "sell", "price": price, "timestamp": timestamp, "profit": 0}
        self.trades.append(trade)
        return trade

    def exit_position(self, exit_result: Dict[str, Any]) -> Dict[str, Any]:
        """Exit a trading position.

        Args:
            exit_result: Dictionary containing exit details (exit_price, profit, exit_time, etc.).

        Returns:
            Dictionary containing trade details.
        """
        exit_price = exit_result["exit_price"]
        profit = exit_result["profit"]
        self.cash += exit_price if self.position == 1 else -exit_price
        trade = {
            "action": "sell" if self.position == 1 else "buy",
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
        """Update portfolio value based on current price.

        Args:
            price: Current market price.

        Returns:
            Dictionary containing portfolio details.
        """
        self.portfolio_value = self.cash + (self.position * price)
        return {
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "position": self.position,
        }

class SignalBacktester:
    """Backtests trading signals using Pointland and Sphere strategies."""

    def __init__(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        square_threshold: float = 350,
        distance_threshold: float = 0.01,
    ) -> None:
        """Initialize the SignalBacktester.

        Args:
            start_date: Start date in YYYY-MM-DD format (default: 30 days ago).
            end_date: End date in YYYY-MM-DD format (default: today).
            square_threshold: Threshold for r_1/r_2 signals.
            distance_threshold: Threshold for target/stop in sphere_exit.
        """
        default_end = datetime.now().strftime("%Y-%m-%d")
        default_start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        self.start_date = start_date or default_start
        self.end_date = end_date or default_end
        self.square_threshold = square_threshold
        self.distance_threshold = distance_threshold
        self.trade_manager = TradeManager()
        self.dao = PostgresDAO()
        self.backtest_dao = BacktestDAO()
        self.target_bars = TARGET_BARS

    async def _fetch_data(self) -> pd.DataFrame:
        """Fetch data from the database.

        Returns:
            DataFrame with market data.

        Raises:
            ValueError: If data is empty or database error occurs.
        """
        try:
            df = await self.dao.get_data(self.start_date, self.end_date)
            if df.empty:
                logger.error("No data for %s to %s", self.start_date, self.end_date)
                raise ValueError("Empty DataFrame from DAO")
            return df
        except PostgresError:
            logger.exception("Failed to connect to RISE database")
            raise ValueError("RISE database error")

    def _validate_and_clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean the input DataFrame.

        Args:
            df: Input DataFrame with market data.

        Returns:
            Cleaned DataFrame.

        Raises:
            ValueError: If required columns are missing or data is invalid.
        """
        required_columns = ["timestamp", "close", "high", "low", "r_1", "r_2"]
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            logger.error("Missing columns: %s", missing_cols)
            raise ValueError("DataFrame missing columns: %s" % missing_cols)

        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        numeric_cols = ["close", "high", "low", "r_1", "r_2"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        original_rows = len(df)
        df = df.dropna(subset=required_columns)
        df = df[(df["close"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["high"] >= df["low"])]

        dropped_rows = original_rows - len(df)
        if dropped_rows > 0:
            logger.warning("Dropped %d rows due to NaN or invalid data", dropped_rows)
        if len(df) > 0:
            signals_triggered = len(df[(df["r_1"] > self.square_threshold) | (df["r_2"] > self.square_threshold)])
            logger.info(
                "Data range: %s to %s, r_1: %.2f to %.2f, r_2: %.2f to %.2f, Signals: %d",
                df["timestamp"].min(),
                df["timestamp"].max(),
                df["r_1"].min(),
                df["r_1"].max(),
                df["r_2"].min(),
                df["r_2"].max(),
                signals_triggered,
            )

        if len(df) < self.target_bars:
            logger.warning("Loaded %d rows, expected ~%d", len(df), self.target_bars)
        if df.empty:
            logger.error("No valid data after cleaning")
            raise ValueError("Empty DataFrame after cleaning")

        return df.sort_values("timestamp").reset_index(drop=True)

    async def load_data(self) -> pd.DataFrame:
        """Load and validate market data.

        Returns:
            Cleaned DataFrame with market data.
        """
        df = await self._fetch_data()
        return self._validate_and_clean_data(df)

    def pointland_signal(self, df: pd.DataFrame) -> pd.Series:
        """Generate Pointland trading signals.

        Args:
            df: DataFrame with market data (close, high, low, r_1, r_2).

        Returns:
            Series with signals (1 for buy, -1 for sell, 0 for hold).
        """
        prev_low = df["low"].shift(1)
        prev_high = df["high"].shift(1)
        buy_signal = (df["close"] < prev_low) & (
            (df["r_1"] > self.square_threshold) | (df["r_2"] > self.square_threshold)
        )
        sell_signal = (df["close"] > prev_high) & (
            (df["r_1"] > self.square_threshold) | (df["r_2"] > self.square_threshold)
        )
        signals = pd.Series(0, index=df.index, dtype=int)
        signals[buy_signal] = 1
        signals[sell_signal] = -1
        return signals

    def sphere_exit(self, df: pd.DataFrame, entry_idx: int, position: int) -> Dict[str, Any]:
        """Determine exit conditions for a position.

        Args:
            df: DataFrame with market data.
            entry_idx: Index of the entry point.
            position: Current position (1 for long, -1 for short).

        Returns:
            Dictionary with exit details (success, failure, profit, etc.).
        """
        entry_data = df.iloc[entry_idx]
        entry_price = entry_data["close"]
        signal_high = entry_data["high"]
        signal_low = entry_data["low"]
        direction = "buy" if position == 1 else "sell"

        target = signal_high * (1 + self.distance_threshold) if direction == "buy" else signal_low * (1 - self.distance_threshold)
        stop = signal_low * (1 - self.distance_threshold) if direction == "buy" else signal_high * (1 + self.distance_threshold)

        future_data = df.iloc[entry_idx + 1 :].copy()
        if future_data.empty:
            logger.warning("No future data for exit at %s", entry_data["timestamp"])
            return {
                "success": False,
                "failure": True,
                "profit": 0,
                "exit_price": entry_price,
                "exit_time": entry_data["timestamp"],
                "time_above_stop": 0,
            }

        if direction == "buy":
            stop_condition = future_data["close"] <= stop
            target_condition = future_data["close"] >= target
            opposing_signal = self.pointland_signal(future_data).shift(1).fillna(0) == -1
            above_stop = future_data["close"] > stop
        else:
            stop_condition = future_data["close"] >= stop
            target_condition = future_data["close"] <= target
            opposing_signal = self.pointland_signal(future_data).shift(1).fillna(0) == 1
            above_stop = future_data["close"] < stop

        exit_conditions = stop_condition | target_condition | opposing_signal
        exit_idx = exit_conditions.idxmax() if exit_conditions.any() else future_data.index[-1]
        exit_row = future_data.loc[exit_idx]
        success = target_condition.loc[exit_idx] if exit_conditions.any() else False
        failure = (stop_condition.loc[exit_idx] or opposing_signal.loc[exit_idx]) if exit_conditions.any() else (
            direction == "buy" and exit_row["close"] <= stop
        ) or (direction == "sell" and exit_row["close"] >= stop)

        time_above_stop = 0
        if above_stop.any():
            above_stop_data = future_data[above_stop][["timestamp"]].copy()
            above_stop_data["time_diff"] = above_stop_data["timestamp"].diff().fillna(pd.Timedelta(seconds=0))
            time_above_stop = above_stop_data["time_diff"].sum().total_seconds() / 60.0

        profit = (exit_row["close"] - entry_price) if direction == "buy" else (entry_price - exit_row["close"])
        return {
            "success": success,
            "failure": failure,
            "profit": profit,
            "exit_price": exit_row["close"],
            "exit_time": exit_row["timestamp"],
            "time_above_stop": time_above_stop,
        }

    def polygon_profit(self) -> Dict[str, Any]:
        """Calculate profit metrics for recent trades.

        Returns:
            Dictionary with win rate, total profit, number of trades, and success rate based on time above stop.
        """
        valid_trades = [t for t in self.trade_manager.trades if "profit" in t and t.get("price") is not None]
        last_15 = valid_trades[-15:] if len(valid_trades) >= 15 else valid_trades
        if not last_15:
            return {"win_rate": 0.0, "total_profit": 0.0, "num_trades": 0, "success_rate_time": 0.0}

        wins = sum(1 for trade in last_15 if trade["profit"] > 0)
        time_successes = sum(1 for trade in last_15 if trade.get("time_above_stop", 0) > TIME_ABOVE_STOP_THRESHOLD)
        return {
            "win_rate": wins / len(last_15),
            "total_profit": sum(trade["profit"] for trade in last_15),
            "num_trades": len(last_15),
            "success_rate_time": time_successes / len(last_15),
        }

    async def backtest(self, backtest_id: str | None = None) -> Dict[str, Any]:
        """Run the backtest over the specified date range.

        Args:
            backtest_id: Unique identifier for the backtest (optional).

        Returns:
            Dictionary with trades, final portfolio value, returns, metrics, and signal count.

        Raises:
            ValueError: If data loading or validation fails.
        """
        self.trade_manager.reset()
        try:
            df = await self.load_data()
        except ValueError:
            logger.exception("Backtest failed")
            return {
                "trades": [],
                "final_portfolio_value": self.trade_manager.cash,
                "returns": 0.0,
                "metrics": {},
                "signals_triggered": 0,
            }

        signals_triggered = len(df[(df["r_1"] > self.square_threshold) | (df["r_2"] > self.square_threshold)])
        signals = self.pointland_signal(df)
        logger.info("Backtest started: %s to %s, %d potential signals", self.start_date, self.end_date, signals_triggered)

        backtest_id = backtest_id or f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        for idx, (signal, row) in enumerate(zip(signals, df.itertuples())):
            timestamp = row.timestamp

            if self.trade_manager.position != 0:
                exit_result = self.sphere_exit(df, self.trade_manager.entry_idx, self.trade_manager.position)
                if exit_result["exit_price"]:
                    trade = self.trade_manager.exit_position(exit_result)
                    if trade:
                        try:
                            await self.backtest_dao.save_signal_event(
                                backtest_id=backtest_id,
                                event={
                                    "timestamp": exit_result["exit_time"],
                                    "event_type": "trade",
                                    "price": trade["price"],
                                },
                            )
                        except PostgresError:
                            logger.exception("Failed to store trade result")

            if signal != 0 and self.trade_manager.position == 0:
                trade = self.trade_manager.enter_position(signal, row.close, timestamp)
                if trade:
                    self.trade_manager.entry_idx = idx
                    try:
                        await self.backtest_dao.save_signal_event(
                            backtest_id=backtest_id,
                            event={
                                "timestamp": timestamp,
                                "event_type": "trade",
                                "price": trade["price"],
                            },
                        )
                        signal_details = {"signal": signal, "price": row.close, "r_1": row.r_1, "r_2": row.r_2}
                        await self.backtest_dao.save_signal_event(
                            backtest_id=backtest_id,
                            event={
                                "timestamp": timestamp,
                                "event_type": "buy_signal" if signal == 1 else "sell_signal",
                                "price": row.close,
                            },
                        )
                    except PostgresError:
                        logger.exception("Failed to store trade/signal result")

            portfolio_details = self.trade_manager.update_portfolio(row.close)
            try:
                await self.backtest_dao.save_portfolio_update(
                    backtest_id=backtest_id,
                    timestamp=timestamp,
                    details=portfolio_details,
                )
            except PostgresError:
                logger.exception("Failed to store portfolio update")

        metrics = self.polygon_profit()
        returns = (self.trade_manager.portfolio_value - INITIAL_CASH) / INITIAL_CASH
        result = {
            "trades": self.trade_manager.trades,
            "final_portfolio_value": self.trade_manager.portfolio_value,
            "returns": returns,
            "metrics": metrics,
            "signals_triggered": signals_triggered,
        }
        try:
            await self.backtest_dao.save_backtest_result(backtest_id, result)
        except PostgresError:
            logger.exception("Failed to store backtest result")
        logger.info("Backtest completed: %d signals, %d trades", signals_triggered, metrics["num_trades"])
        return result
