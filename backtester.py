"""Backtesting engine for trading signals using Pointland and Sphere strategies."""
from __future__ import annotations
from datetime import datetime, timedelta, timezone
import logging
from typing import Optional, Any
import numpy as np
import pandas as pd
from backtest_dao import BacktestDAO
from dao import PostgresDAO
from asyncpg.exceptions import PostgresError

logger = logging.getLogger(__name__)

INITIAL_CASH = 100_000
TARGET_BARS = 1000
TIME_ABOVE_STOP_THRESHOLD = 30
DEFAULT_DATE_RANGE_DAYS = 30
BUY_SIGNAL = 1
SELL_SIGNAL = -1
HOLD_SIGNAL = 0


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
        self.trades: list[dict[str, Any]] = []
        self.entry_price: float | None = None
        self.entry_idx: int | None = None

    def reset(self) -> None:
        """Reset the trade manager to initial state."""
        self.cash = INITIAL_CASH
        self.portfolio_value = self.cash
        self.position = 0
        self.trades = []
        self.entry_price = None
        self.entry_idx = None

    def enter_position(self, signal: int, price: float, timestamp: datetime) -> dict[str, Any]:
        """Enter a trading position based on a signal.

        Args:
            signal: 1 for buy, -1 for sell.
            price: Entry price.
            timestamp: Timestamp of the trade.

        Returns:
            Dictionary containing trade details or empty dict if insufficient cash.
        """
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

    def exit_position(self, exit_result: dict[str, Any]) -> dict[str, Any]:
        """Exit a trading position.

        Args:
            exit_result: Dictionary containing exit details (exit_price, profit, exit_time, etc.).

        Returns:
            Dictionary containing trade details.
        """
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

    def update_portfolio(self, price: float) -> dict[str, Any]:
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
        default_end = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        default_start = (
            datetime.now(timezone.utc) - timedelta(days=DEFAULT_DATE_RANGE_DAYS)
        ).strftime("%Y-%m-%d")
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
                logger.error(f"No data for {self.start_date} to {self.end_date}")
                raise ValueError("Empty DataFrame from DAO")
        except PostgresError as e:
            logger.exception("Failed to connect to RISE database")
            raise ValueError("RISE database error") from e

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
            logger.error(f"Missing columns: {missing_cols}")
            raise ValueError(f"DataFrame missing columns: {missing_cols}")

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
            logger.warning(f"Dropped {dropped_rows} rows due to NaN or invalid data")
        if len(df) > 0:
            signals_triggered = len(
                df[(df["r_1"] > self.square_threshold) | (df["r_2"] > self.square_threshold)]
            )
            logger.info(
                f"Data range: {df['timestamp'].min()} to {df['timestamp'].max()}, "
                f"r_1: {df['r_1'].min():.2f} to {df['r_1'].max():.2f}, "
                f"r_2: {df['r_2'].min():.2f} to {df['r_2'].max():.2f}, "
                f"Signals: {signals_triggered}"
            )

        if len(df) < self.target_bars:
            logger.warning(f"Loaded {len(df)} rows, expected ~{self.target_bars}")
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
        signals = pd.Series(HOLD_SIGNAL, index=df.index, dtype=int)
        signals[buy_signal] = BUY_SIGNAL
        signals[sell_signal] = SELL_SIGNAL
        return signals

    def _calculate_target_and_stop(
        self,
        direction: str,
        signal_high: float,
        signal_low: float,
    ) -> tuple[float, float]:
        """Calculate target and stop prices.

        Args:
            direction: "buy" or "sell".
            signal_high: High price at entry.
            signal_low: Low price at entry.

        Returns:
            Tuple of (target, stop) prices.
        """
        if direction == "buy":
            target = signal_high * (1 + self.distance_threshold)
            stop = signal_low * (1 - self.distance_threshold)
        else:
            target = signal_low * (1 - self.distance_threshold)
            stop = signal_high * (1 + self.distance_threshold)
        return target, stop

    def _check_exit_conditions(
        self,
        future_data: pd.DataFrame,
        stop: float,
        target: float,
        direction: str,
        opposing_signal: pd.Series,
    ) -> pd.Series:
        """Check if any exit condition is met.

        Args:
            future_data: DataFrame of price data after entry.
            stop: Stop loss price.
            target: Target profit price.
            direction: "buy" or "sell".
            opposing_signal: Series of opposing signals.

        Returns:
            Series indicating if any exit condition is met.
        """
        if direction == "buy":
            stop_condition = future_data["close"] <= stop
            target_condition = future_data["close"] >= target
        else:
            stop_condition = future_data["close"] >= stop
            target_condition = future_data["close"] <= target
        return stop_condition | target_condition | opposing_signal

    def _determine_exit_outcome(
        self,
        exit_conditions: pd.Series,
        future_data: pd.DataFrame,
        direction: str,
        stop: float,
        target: float,
        opposing_signal: pd.Series
    ) -> tuple[bool, bool, pd.Series]:
        """Determine the outcome (success/failure) of the exit.

        Args:
            exit_conditions: Series indicating if any exit condition is met.
            future_data: DataFrame of price data after entry.
            direction: "buy" or "sell"
            stop: stop price
            target: target price
            opposing_signal: signal indicating the opposite trade

        Returns:
            Tuple of (success, failure, exit_row).
        """
        if exit_conditions.any():
            exit_idx = exit_conditions.idxmax()
            exit_row = future_data.loc[exit_idx]
            if direction == "buy":
                success = future_data["close"].loc[exit_idx] >= target
                failure = future_data["close"].loc[exit_idx] <= stop or opposing_signal.loc[exit_idx]
            else:
                success = future_data["close"].loc[exit_idx] <= target
                failure = future_data["close"].loc[exit_idx] >= stop or opposing_signal.loc[exit_idx]
        else:
            exit_idx = future_data.index[-1]
            exit_row = future_data.loc[exit_idx]
            success = False
            failure = (direction == "buy" and exit_row["close"] <= stop) or (
                direction == "sell" and exit_row["close"] >= stop
            )
        return success, failure, exit_row

    def _calculate_time_above_stop(self, future_data: pd.DataFrame, stop: float, direction: str) -> float:
        """Calculate time above/below stop price.

        Args:
            future_data: DataFrame of price data after entry.
            stop: Stop loss price.
            direction: "buy" or "sell"

        Returns:
            Time in minutes.
        """
        if direction == 'buy':
            above_stop = future_data["close"] > stop
        else:
            above_stop = future_data["close"] < stop

        time_above_stop = 0
        if above_stop.any():
            above_stop_data = future_data[above_stop][["timestamp"]].copy()
            above_stop_data["time_diff"] = above_stop_data["timestamp"].diff().fillna(
                pd.Timedelta(seconds=0)
            )
            time_above_stop = above_stop_data["time_diff"].sum().total_seconds() / 60.0
        return time_above_stop

    def sphere_exit(self, df: pd.DataFrame, entry_idx: int, position: int) -> dict[str, Any]:
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
        direction = "buy" if position == BUY_SIGNAL else "sell"

        target, stop = self._calculate_target_and_stop(direction, signal_high, signal_low)

        future_data = df.iloc[entry_idx + 1 :].copy()
        if future_data.empty:
            logger.warning(f"No future data for exit at {entry_data['timestamp']}")
            return {
                "success": False,
                "failure": True,
                "profit": 0,
                "exit_price": entry_price,
                "exit_time": entry_data["timestamp"],
                "time_above_stop": 0,
            }

        opposing_signal = self.pointland_signal(future_data).shift(1).fillna(0) == (
            SELL_SIGNAL if direction == "buy" else BUY_SIGNAL
        )
        exit_conditions = self._check_exit_conditions(
            future_data, stop, target, direction, opposing_signal
        )
        success, failure, exit_row = self._determine_exit_outcome(
            exit_conditions, future_data, direction, stop, target, opposing_signal
        )

        time_above_stop = self._calculate_time_above_stop(future_data, stop, direction)
        profit = (
            exit_row["close"] - entry_price
            if direction == "buy"
            else entry_price - exit_row["close"]
        )
        return {
            "success": success,
            "failure": failure,
            "profit": profit,
            "exit_price": exit_row["close"],
            "exit_time": exit_row["timestamp"],
            "time_above_stop": time_above_stop,
        }

    def polygon_profit(self) -> dict[str, Any]:
        """Calculate profit metrics for recent trades.

        Returns:
            Dictionary with win rate, total profit, number of trades, and success rate based on time above stop.
        """
        valid_trades = [
            t for t in self.trade_manager.trades if "profit" in t and t.get("price") is not None
        ]
        last_15 = valid_trades[-15:] if len(valid_trades) >= 15 else valid_trades
        if not last_15:
            return {
                "win_rate": 0.0,
                "total_profit": 0.0,
                "num_trades": 0,
                "success_rate_time": 0.0,
            }

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

    async def process_signals(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        backtest_id: str,
    ) -> tuple[list[dict[str, Any]], int]:
        """Process trading signals and execute trades.

        Args:
            df: DataFrame with market data.
            signals: Series with trading signals.
            backtest_id: Unique identifier for the backtest.

        Returns:
            Tuple of trades list and number of signals triggered.
        """
        signals_triggered = len(
            df[(df["r_1"] > self.square_threshold) | (df["r_2"] > self.square_threshold)]
        )
        for idx, (signal, row) in enumerate(zip(signals, df.itertuples())):
            timestamp = row.timestamp

            if self.trade_manager.position != 0:
                exit_result = self.sphere_exit(
                    df,
                    self.trade_manager.entry_idx,
                    self.trade_manager.position,
                )
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

            if signal != HOLD_SIGNAL and self.trade_manager.position == 0:
                trade = self.trade_manager.enter_position(
                    signal,
                    row.close,
                    timestamp,
                )
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
                        signal_details = {
                            "signal": signal,
                            "price": row.close,
                            "r_1": row.r_1,
                            "r_2": row.r_2,
                        }
                        await self.backtest_dao.save_signal_event(
                            backtest_id=backtest_id,
                            event={
                                "timestamp": timestamp,
                                "event_type": "buy_signal"
                                if signal == BUY_SIGNAL
                                else "sell_signal",
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

        return self.trade_manager.trades, signals_triggered

    async def save_backtest_results(self, backtest_id: str, result: dict[str, Any]) -> None:
        """Save backtest results to the database.

        Args:
            backtest_id: Unique identifier for the backtest.
            result: Dictionary containing backtest results.

        Raises:
            PostgresError: If database write fails.
        """
        try:
            await self.backtest_dao.save_backtest_result(backtest_id, result)
        except PostgresError:
            logger.exception("Failed to store backtest result")

    async def backtest(self, backtest_id: str | None = None) -> dict[str, Any]:
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
            market_data = await self.load_data()
        except ValueError:
            logger.exception("Backtest failed")
            return {
                "trades": [],
                "final_portfolio_value": self.trade_manager.cash,
                "returns": 0.0,
                "metrics": {},
                "signals_triggered": 0,
            }

        signals = self.pointland_signal(market_data)
        backtest_id = backtest_id or f"backtest_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

        trades, signals_triggered = await self.process_signals(
            market_data,
            signals,
            backtest_id,
        )
        metrics = self.polygon_profit()
        returns = (self.trade_manager.portfolio_value - INITIAL_CASH) / INITIAL_CASH
        result = {
            "trades": trades,
            "final_portfolio_value": self.trade_manager.portfolio_value,
            "returns": returns,
            "metrics": metrics,
            "signals_triggered": signals_triggered,
        }

        await self.save_backtest_results(backtest_id, result)
        logger.info(
            "Backtest completed: %s signals, %s trades", signals_triggered, metrics["num_trades"]
        )
        return result
