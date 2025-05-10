from typing import TypedDict
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class ExitResult(TypedDict):
    success: bool
    failure: bool
    profit: float
    exit_price: float
    exit_time: datetime
    time_above_stop: float

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
            raise ValueError(f"Missing columns: {missing_cols}")

        df = df.dropna(subset=required_columns)
        if df.empty:
            raise ValueError("Empty DataFrame after cleaning")

        # Ensure chronological order
        df = df.sort_values("timestamp")

        # Log basic summary (optional, can be removed in production)
        logger.debug(f"Loaded {len(df)} rows from {df['timestamp'].min()} to {df['timestamp'].max()}")

        return df

    def _calculate_target_and_stop(
        self,
        direction: str,
        signal_high: float,
        signal_low: float,
    ) -> tuple[float, float]:
        """Calculate target and stop prices.

        Args:
            direction: Trade direction ("buy" or "sell").
            signal_high: High price at entry.
            signal_low: Low price at entry.

        Returns:
            Tuple of (target, stop) prices.

        Raises:
            ValueError: If direction is invalid or prices are non-positive.
        """
        if direction not in ["buy", "sell"]:
            raise ValueError(f"Invalid direction: {direction}. Must be 'buy' or 'sell'.")
        if signal_high <= 0 or signal_low <= 0:
            raise ValueError("Signal high and low must be positive.")
        
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
            direction: Trade direction ("buy" or "sell").
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

    def _calculate_time_above_stop(self, future_data: pd.DataFrame, stop: float, direction: str) -> float:
        """Calculate time above/below stop price.

        Args:
            future_data: DataFrame of price data after entry.
            stop: Stop loss price.
            direction: Trade direction ("buy" or "sell").

        Returns:
            Time in minutes.
        """
        if future_data.empty:
            logger.warning("No future data for time above stop calculation")
            return 0.0

        if direction == "buy":
            above_stop = future_data["close"] > stop
        else:
            above_stop = future_data["close"] < stop

        if not above_stop.any():
            return 0.0

        timestamps = future_data.loc[above_stop, "timestamp"]
        time_diff = timestamps.diff().fillna(pd.Timedelta(seconds=0))
        return time_diff.sum().total_seconds() / 60.0

    def _determine_exit_outcome(
        self,
        exit_conditions: pd.Series,
        future_data: pd.DataFrame,
        direction: str,
        stop: float,
        target: float,
        opposing_signal: pd.Series
    ) -> tuple[bool, bool, pd.Series]:
        """Determine the outcome (success/failure) of the trade exit.

        Args:
            exit_conditions: Series indicating if any exit condition is met.
            future_data: DataFrame of price data after entry.
            direction: Trade direction ("buy" or "sell").
            stop: Stop loss price.
            target: Target profit price.
            opposing_signal: Series indicating opposing trade signals.

        Returns:
            Tuple of (success, failure, exit_row).
        """
        if not isinstance(exit_conditions, pd.Series) or exit_conditions.empty:
            logger.warning("Invalid exit conditions, using last row as fallback")
            exit_row = future_data.iloc[-1]
            success = False
            failure = (direction == "buy" and exit_row["close"] <= stop) or (
                direction == "sell" and exit_row["close"] >= stop
            )
            return success, failure, exit_row

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
            exit_row = future_data.iloc[-1]
            success = False
            failure = (direction == "buy" and exit_row["close"] <= stop) or (
                direction == "sell" and exit_row["close"] >= stop
            )
        return success, failure, exit_row

    def sphere_exit(self, df: pd.DataFrame, entry_idx: int, position: int) -> ExitResult:
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

        future_data = df.iloc[entry_idx + 1:]
        if future_data.empty:
            logger.warning(f"No future data for exit at {entry_data['timestamp']}")
            return ExitResult(
                success=False,
                failure=True,
                profit=0,
                exit_price=entry_price,
                exit_time=entry_data["timestamp"],
                time_above_stop=0,
            )

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
        return ExitResult(
            success=success,
            failure=failure,
            profit=profit,
            exit_price=exit_row["close"],
            exit_time=exit_row["timestamp"],
            time_above_stop=time_above_stop,
        )
