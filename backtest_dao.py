"""Data Access Object for managing backtest results, signal events, and portfolio updates in PostgreSQL."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from datetime import datetime, timezone # Added timezone

import pandas as pd
from asyncpg.exceptions import PostgresError, UniqueViolationError # Added UniqueViolationError

# Assuming these are correctly implemented in their respective files
from settings import Settings
from dao import PostgresDAO, DAOError, PoolNotInitializedError, DataFetchError # Import custom exceptions

if TYPE_CHECKING:
    # This is only for type hinting, not a runtime import
    # from asyncpg import Pool # Not directly used by BacktestDAO, PostgresDAO handles it
    pass

logger = logging.getLogger(__name__)

# --- Constants for Table Names and Error Messages ---
BACKTEST_RESULTS_TABLE: str = "backtest_results"
SIGNAL_EVENTS_TABLE: str = "signal_events"
PORTFOLIO_UPDATES_TABLE: str = "portfolio_updates"

# Required keys for data validation
REQUIRED_RESULT_KEYS: set = {"final_portfolio_value", "returns", "start_date", "end_date"}
REQUIRED_EVENT_KEYS: set = {"timestamp", "event_type", "price", "trade_id"} # Added trade_id for better linking
REQUIRED_PORTFOLIO_KEYS: set = {"timestamp", "portfolio_value", "cash", "position_size", "position_value"} # Made position more specific

# Error Messages
MISSING_KEYS_ERROR_MSG: str = "Input dictionary missing required keys: {missing_keys}"
DATA_SAVE_ERROR_MSG: str = "Failed to save {item_type} to the database."
DATA_FETCH_ERROR_MSG: str = "Failed to fetch {item_type} from the database."

class BacktestDataError(DAOError):
    """Base exception for BacktestDAO specific errors."""
    pass

class DataValidationError(BacktestDataError, ValueError):
    """Exception for data validation errors."""
    pass

class DataPersistenceError(BacktestDataError):
    """Exception for errors during data saving."""
    pass

class BacktestDAO:
    """
    DAO for managing backtest results, signal events, and portfolio updates.
    It encapsulates all database interactions related to backtesting artifacts.
    """

    def __init__(self, settings: Optional = None) -> None:
        """
        Initialize the BacktestDAO.

        It creates and manages its own PostgresDAO instance for database operations.

        Args:
            settings: Configuration settings. If None, default Settings are used.
        """
        self.settings: Settings = settings or Settings()
        # BacktestDAO owns and manages its PostgresDAO instance
        self.postgres_dao: PostgresDAO = PostgresDAO(self.settings)
        logger.debug("BacktestDAO initialized with its own PostgresDAO instance.")

    async def setup(self) -> None:
        """
        Set up the underlying PostgresDAO, which initializes the database connection pool.

        Raises:
            RuntimeError: If the database pool initialization in PostgresDAO fails.
        """
        logger.info("Setting up BacktestDAO (delegating to PostgresDAO setup)...")
        await self.postgres_dao.setup() # PostgresDAO.setup() already logs and raises RuntimeError
        logger.info("BacktestDAO setup complete.")

    async def close_pool(self) -> None:
        """Close the database connection pool managed by the underlying PostgresDAO."""
        logger.info("Closing BacktestDAO pool (delegating to PostgresDAO close_pool)...")
        await self.postgres_dao.close_pool() # PostgresDAO.close_pool() handles logging and checks
        logger.info("BacktestDAO pool closed.")

    def _validate_data(self, data: Dict[str, Any], required_keys: set, item_name: str) -> None:
        """
        Validates if the provided dictionary contains all required keys.

        Args:
            data: The dictionary to validate.
            required_keys: A set of keys that must be present in the data.
            item_name: A descriptive name of the item being validated (e.g., "backtest result").

        Raises:
            DataValidationError: If any required keys are missing.
        """
        missing = required_keys - data.keys()
        if missing:
            error_detail = MISSING_KEYS_ERROR_MSG.format(missing_keys=missing)
            logger.error(f"Validation failed for {item_name}: {error_detail}")
            raise DataValidationError(f"Validation failed for {item_name}: {error_detail}")

    async def save_backtest_result(self, backtest_id: str, result_data: Dict[str, Any]) -> None:
        """
        Save a backtest result to the database.

        Args:
            backtest_id: Unique identifier for the backtest.
            result_data: Dictionary containing backtest results. Must include keys defined
                         in REQUIRED_RESULT_KEYS. Values will be type-converted.

        Raises:
            DataValidationError: If required keys are missing in result_data.
            DataPersistenceError: If a database error occurs during saving.
        """
        self._validate_data(result_data, REQUIRED_RESULT_KEYS, "backtest result")
        logger.debug(f"Attempting to save backtest result for ID: {backtest_id}")

        query = f"""
            INSERT INTO {BACKTEST_RESULTS_TABLE}
                (backtest_id, final_portfolio_value, returns, start_date, end_date, parameters_json)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (backtest_id) DO UPDATE SET
                final_portfolio_value = EXCLUDED.final_portfolio_value,
                returns = EXCLUDED.returns,
                start_date = EXCLUDED.start_date,
                end_date = EXCLUDED.end_date,
                parameters_json = EXCLUDED.parameters_json,
                updated_at = CURRENT_TIMESTAMP;
        """
        try:
            # Ensure dates are proper datetime objects, preferably timezone-aware
            start_dt = pd.to_datetime(result_data["start_date"]).to_pydatetime()
            if start_dt.tzinfo is None:
                start_dt = start_dt.replace(tzinfo=timezone.utc)

            end_dt = pd.to_datetime(result_data["end_date"]).to_pydatetime()
            if end_dt.tzinfo is None:
                end_dt = end_dt.replace(tzinfo=timezone.utc)
            
            # Store other parameters as JSONB if needed
            parameters_json = result_data.get("parameters", {}) # Example: store strategy params

            async with self.postgres_dao.pool.acquire() as conn:
                await conn.execute(
                    query,
                    backtest_id,
                    float(result_data["final_portfolio_value"]),
                    float(result_data["returns"]),
                    start_dt,
                    end_dt,
                    parameters_json # Pass parameters as JSON
                )
            logger.info(f"Successfully saved/updated backtest result for ID: {backtest_id}")
        except PostgresError as e:
            error_msg = DATA_SAVE_ERROR_MSG.format(item_type="backtest result")
            logger.exception(f"{error_msg} for ID {backtest_id}: {e}")
            raise DataPersistenceError(f"{error_msg} for ID {backtest_id}.") from e
        except (ValueError, TypeError) as e: # Catch type conversion errors
            error_msg = f"Type error in backtest result data for ID {backtest_id}: {e}"
            logger.error(error_msg)
            raise DataValidationError(error_msg) from e


    async def save_signal_event(self, backtest_id: str, event_data: Dict[str, Any]) -> None:
        """
        Save a single signal event to the database.

        Args:
            backtest_id: Unique identifier for the backtest this event belongs to.
            event_data: Dictionary containing signal event details. Must include keys
                        defined in REQUIRED_EVENT_KEYS.

        Raises:
            DataValidationError: If required keys are missing or data types are incorrect.
            DataPersistenceError: If a database error occurs.
        """
        self._validate_data(event_data, REQUIRED_EVENT_KEYS, "signal event")
        logger.debug(f"Attempting to save signal event for backtest ID: {backtest_id}, event time: {event_data.get('timestamp')}")

        query = f"""
            INSERT INTO {SIGNAL_EVENTS_TABLE}
                (backtest_id, trade_id, timestamp, event_type, price, details_json)
            VALUES ($1, $2, $3, $4, $5, $6);
        """
        try:
            event_ts = pd.to_datetime(event_data["timestamp"]).to_pydatetime()
            if event_ts.tzinfo is None:
                event_ts = event_ts.replace(tzinfo=timezone.utc)
            
            details_json = event_data.get("details", {}) # For extra info like order_id, quantity

            async with self.postgres_dao.pool.acquire() as conn:
                await conn.execute(
                    query,
                    backtest_id,
                    str(event_data["trade_id"]), # Assuming trade_id is a string or can be cast
                    event_ts,
                    str(event_data["event_type"]),
                    float(event_data["price"]),
                    details_json
                )
            logger.info(f"Saved signal event for backtest ID {backtest_id} at {event_ts}")
        except PostgresError as e:
            error_msg = DATA_SAVE_ERROR_MSG.format(item_type="signal event")
            logger.exception(f"{error_msg} for backtest ID {backtest_id}: {e}")
            raise DataPersistenceError(f"{error_msg} for backtest ID {backtest_id}.") from e
        except (ValueError, TypeError) as e:
            error_msg = f"Type error in signal event data for backtest ID {backtest_id}: {e}"
            logger.error(error_msg)
            raise DataValidationError(error_msg) from e

    async def save_portfolio_update(self, backtest_id: str, update_data: Dict[str, Any]) -> None:
        """
        Save a single portfolio update (snapshot) to the database.

        Args:
            backtest_id: Unique identifier for the backtest.
            update_data: Dictionary containing portfolio details. Must include keys
                         defined in REQUIRED_PORTFOLIO_KEYS.

        Raises:
            DataValidationError: If required keys are missing or data types are incorrect.
            DataPersistenceError: If a database error occurs.
        """
        self._validate_data(update_data, REQUIRED_PORTFOLIO_KEYS, "portfolio update")
        logger.debug(f"Attempting to save portfolio update for backtest ID: {backtest_id}, time: {update_data.get('timestamp')}")

        query = f"""
            INSERT INTO {PORTFOLIO_UPDATES_TABLE}
                (backtest_id, timestamp, portfolio_value, cash, position_size, position_value, details_json)
            VALUES ($1, $2, $3, $4, $5, $6, $7);
        """
        try:
            update_ts = pd.to_datetime(update_data["timestamp"]).to_pydatetime()
            if update_ts.tzinfo is None:
                update_ts = update_ts.replace(tzinfo=timezone.utc)
            
            details_json = update_data.get("details", {}) # For asset breakdown, etc.

            async with self.postgres_dao.pool.acquire() as conn:
                await conn.execute(
                    query,
                    backtest_id,
                    update_ts,
                    float(update_data["portfolio_value"]),
                    float(update_data["cash"]),
                    float(update_data["position_size"]), # Assuming size can be float (e.g. crypto)
                    float(update_data["position_value"]),
                    details_json
                )
            logger.info(f"Saved portfolio update for backtest ID {backtest_id} at {update_ts}")
        except PostgresError as e:
            error_msg = DATA_SAVE_ERROR_MSG.format(item_type="portfolio update")
            logger.exception(f"{error_msg} for backtest ID {backtest_id}: {e}")
            raise DataPersistenceError(f"{error_msg} for backtest ID {backtest_id}.") from e
        except (ValueError, TypeError) as e:
            error_msg = f"Type error in portfolio update data for backtest ID {backtest_id}: {e}"
            logger.error(error_msg)
            raise DataValidationError(error_msg) from e

    async def get_plot_data(self, backtest_id: str, start_date_str: str, end_date_str: str) -> List]:
        """
        Fetch portfolio updates for performance visualization, filtered by backtest_id and date range.

        Args:
            backtest_id: The identifier of the backtest to fetch data for.
            start_date_str: Start date string in 'YYYY-MM-DD' format.
            end_date_str: End date string in 'YYYY-MM-DD' format.

        Returns:
            List of dictionaries, each containing 'timestamp' and 'portfolio_value'.
            Returns an empty list if no data is found or an error occurs.

        Raises:
            PoolNotInitializedError: If the database pool is not initialized.
            DataFetchError: If a database error occurs.
            ValueError: If date strings are invalid.
        """
        # Pool access via property ensures it's initialized
        active_pool = self.postgres_dao.pool
        logger.debug(f"Fetching plot data for backtest ID {backtest_id}, range: {start_date_str} to {end_date_str}")

        query = f"""
            SELECT timestamp, portfolio_value
            FROM {PORTFOLIO_UPDATES_TABLE}
            WHERE backtest_id = $1 AND timestamp >= $2 AND timestamp <= $3
            ORDER BY timestamp ASC;
        """
        try:
            start_dt = datetime.strptime(start_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            end_dt = datetime.strptime(end_date_str, "%Y-%m-%d").replace(hour=23, minute=59, second=59, microsecond=999999).replace(tzinfo=timezone.utc)

            async with active_pool.acquire() as conn:
                rows = await conn.fetch(query, backtest_id, start_dt, end_dt)

            if not rows:
                logger.warning(f"No plot data found for backtest ID {backtest_id} in range {start_date_str}-{end_date_str}")
                return
            return [dict(row) for row in rows]
        except ValueError as ve:
            logger.error(f"Invalid date format for plot data: {start_date_str}, {end_date_str}. Error: {ve}")
            raise ValueError(f"Invalid date format for plot data. Use YYYY-MM-DD. Details: {ve}") from ve
        except PostgresError as e:
            error_msg = DATA_FETCH_ERROR_MSG.format(item_type="plot data")
            logger.exception(f"{error_msg} for backtest ID {backtest_id}: {e}")
            # Decide: raise DataFetchError or return? Raising is often better for API clarity.
            raise DataFetchError(f"{error_msg} for backtest ID {backtest_id}.") from e

    async def get_signal_events(self, backtest_id: str, start_date_str: str, end_date_str: str) -> List]:
        """
        Fetch signal events for a specific backtest and date range.

        Args:
            backtest_id: The identifier of the backtest.
            start_date_str: Start date string in 'YYYY-MM-DD' format.
            end_date_str: End date string in 'YYYY-MM-DD' format.

        Returns:
            List of dictionaries, each representing a signal event.
            Returns an empty list if no events are found or an error occurs.

        Raises:
            PoolNotInitializedError: If the database pool is not initialized.
            DataFetchError: If a database error occurs.
            ValueError: If date strings are invalid.
        """
        active_pool = self.postgres_dao.pool
        logger.debug(f"Fetching signal events for backtest ID {backtest_id}, range: {start_date_str} to {end_date_str}")

        query = f"""
            SELECT timestamp, event_type, price, details_json
            FROM {SIGNAL_EVENTS_TABLE}
            WHERE backtest_id = $1 AND timestamp >= $2 AND timestamp <= $3
            ORDER BY timestamp ASC;
        """
        try:
            start_dt = datetime.strptime(start_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            end_dt = datetime.strptime(end_date_str, "%Y-%m-%d").replace(hour=23, minute=59, second=59, microsecond=999999).replace(tzinfo=timezone.utc)

            async with active_pool.acquire() as conn:
                rows = await conn.fetch(query, backtest_id, start_dt, end_dt)

            if not rows:
                logger.warning(f"No signal events found for backtest ID {backtest_id} in range {start_date_str}-{end_date_str}")
                return
            return [dict(row) for row in rows]
        except ValueError as ve:
            logger.error(f"Invalid date format for signal events: {start_date_str}, {end_date_str}. Error: {ve}")
            raise ValueError(f"Invalid date format for signal events. Use YYYY-MM-DD. Details: {ve}") from ve
        except PostgresError as e:
            error_msg = DATA_FETCH_ERROR_MSG.format(item_type="signal events")
            logger.exception(f"{error_msg} for backtest ID {backtest_id}: {e}")
            raise DataFetchError(f"{error_msg} for backtest ID {backtest_id}.") from e
