"""Data Access Object (DAO) for interacting with PostgreSQL database."""
from __future__ import annotations

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone # Added timezone for UTC awareness

import asyncpg
import pandas as pd
from asyncpg import Pool, create_pool
from asyncpg.exceptions import PostgresError # Specific exception

from settings import Settings # Assuming settings.py is in the project root or accessible

logger = logging.getLogger(__name__)

# Module-level constants for clarity
POOL_NOT_INITIALIZED_ERROR_MSG: str = "Database connection pool has not been initialized."
DAO_FETCH_ERROR_MSG: str = "An error occurred while fetching data from the database."
DAO_SETUP_ERROR_MSG: str = "Database pool initialization failed."

class DAOError(Exception):
    """Base exception for DAO-related errors."""
    pass

class PoolNotInitializedError(DAOError):
    """Exception raised when the database pool is not initialized."""
    pass

class DataFetchError(DAOError):
    """Exception raised for errors during data fetching."""
    pass

class PostgresDAO:
    """
    Data Access Object for interacting with a PostgreSQL database
    using asyncpg.
    """
    def __init__(self, settings: Optional = None) -> None:
        """
        Initialize PostgresDAO with optional settings.

        Args:
            settings: Configuration settings for the DAO. Defaults to a new Settings instance.
        """
        self.settings: Settings = settings or Settings()
        self._pool: Optional[Pool] = None # Changed to _pool for internal use convention

    @property
    def pool(self) -> Pool:
        """
        Provides access to the connection pool, raising an error if not initialized.
        """
        if self._pool is None:
            logger.error(POOL_NOT_INITIALIZED_ERROR_MSG)
            raise PoolNotInitializedError(POOL_NOT_INITIALIZED_ERROR_MSG)
        return self._pool

    async def setup(self) -> None:
        """
        Set up the database connection pool.

        Raises:
            RuntimeError: If the database pool initialization fails.
        """
        if self._pool is not None:
            logger.info("PostgresDAO pool already initialized.")
            return

        logger.info(f"Initializing PostgresDAO pool for DSN: {self.settings.BACKTEST_DB_URL[:30]}...") # Log partial DSN for security
        try:
            self._pool = await create_pool(
                dsn=self.settings.BACKTEST_DB_URL,
                min_size=self.settings.DB_POOL_MIN_SIZE if hasattr(self.settings, 'DB_POOL_MIN_SIZE') else 1,
                max_size=self.settings.DB_POOL_MAX_SIZE if hasattr(self.settings, 'DB_POOL_MAX_SIZE') else 10,
                # Consider adding timeout settings if needed
                # command_timeout=60,  # Example: timeout for individual commands
                # connection_class=CustomConnection, # If you need custom connection behavior
            )
            logger.info("PostgresDAO pool initialized successfully.")
        except PostgresError as e:
            logger.exception(f"{DAO_SETUP_ERROR_MSG}: {e}")
            raise RuntimeError(DAO_SETUP_ERROR_MSG) from e
        except Exception as e: # Catch other potential errors during pool creation
            logger.exception(f"Unexpected error during {DAO_SETUP_ERROR_MSG}: {e}")
            raise RuntimeError(f"Unexpected error during {DAO_SETUP_ERROR_MSG}") from e

    async def get_data(self, start_date_str: str, end_date_str: str) -> pd.DataFrame:
        """
        Fetch BTC price data from the database for a given date range.

        Args:
            start_date_str: Start date string in 'YYYY-MM-DD' format.
            end_date_str: End date string in 'YYYY-MM-DD' format.

        Returns:
            A pandas DataFrame containing the price data.
            Returns an empty DataFrame if no data is found for the range.

        Raises:
            PoolNotInitializedError: If the database pool is not initialized.
            DataFetchError: If a database error or other unexpected error occurs during data fetching.
            ValueError: If date strings are not in the correct format.
        """
        # Ensure pool is accessed via property to get the check
        active_pool = self.pool

        query = """
            SELECT timestamp, open, high, low, close, r_1, r_2
            FROM btc_prices
            WHERE timestamp >= $1 AND timestamp <= $2
            ORDER BY timestamp ASC
        """
        try:
            # Convert date strings to timezone-aware datetime objects (UTC)
            # This assumes your database stores timestamps in UTC or handles timezone conversion appropriately.
            start_dt = datetime.strptime(start_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            end_dt = datetime.strptime(end_date_str, "%Y-%m-%d").replace(hour=23, minute=59, second=59, microsecond=999999).replace(tzinfo=timezone.utc)

            logger.debug(f"Executing query for date range: {start_dt} to {end_dt}")

            async with active_pool.acquire() as conn:
                rows: List = await conn.fetch(query, start_dt, end_dt)

            if not rows:
                logger.warning(f"No data found for date range: {start_date_str} to {end_date_str}")
                return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'r_1', 'r_2'])

            # Convert list of asyncpg.Record objects to DataFrame
            # Using a list of dicts can be more robust if column order/names might vary slightly
            # or for easier debugging, though direct conversion from records is efficient.
            data_list = [dict(row) for row in rows]
            price_data = pd.DataFrame(data_list)

            # Ensure timestamp column is of datetime type if not already
            if not pd.api.types.is_datetime64_any_dtype(price_data['timestamp']):
                 price_data['timestamp'] = pd.to_datetime(price_data['timestamp'], utc=True)
            elif price_data['timestamp'].dt.tz is None: # If datetime but not timezone-aware
                 price_data['timestamp'] = price_data['timestamp'].dt.tz_localize('UTC')


            logger.info(f"Fetched {len(price_data)} price records from {start_date_str} to {end_date_str}.")
            return price_data

        except ValueError as ve: # Catch strptime errors
            logger.error(f"Invalid date format provided: {start_date_str}, {end_date_str}. Error: {ve}")
            raise ValueError(f"Invalid date format. Please use YYYY-MM-DD. Details: {ve}") from ve
        except PostgresError as e:
            logger.exception(f"Database error while fetching data for {start_date_str}-{end_date_str}: {e}")
            raise DataFetchError(f"{DAO_FETCH_ERROR_MSG} Database query failed.") from e
        except Exception as e: # Catch any other unexpected errors
            logger.exception(f"Unexpected error while fetching data for {start_date_str}-{end_date_str}: {e}")
            raise DataFetchError(f"{DAO_FETCH_ERROR_MSG} An unexpected error occurred.") from e

    async def close_pool(self) -> None:
        """Close the database connection pool if it was initialized."""
        if self._pool:
            logger.info("Closing PostgresDAO pool...")
            try:
                await self._pool.close()
                self._pool = None # Reset the pool attribute
                logger.info("PostgresDAO pool closed successfully.")
            except Exception as e: # Catch potential errors during pool closing
                logger.exception(f"Error encountered while closing PostgresDAO pool: {e}")
        else:
            logger.warning("Attempted to close PostgresDAO pool, but it was not initialized or already closed.")
