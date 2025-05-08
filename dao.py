"""Data Access Object (DAO) for interacting with PostgreSQL database."""
from __future__ import annotations
import logging
from typing import Union
from settings import Settings
import asyncpg
import pandas as pd
from asyncpg import Pool, create_pool
from asyncpg.exceptions import PostgresError

logger = logging.getLogger(__name__)
POOL_NOT_INITIALIZED = "Database pool not initialized"

class PostgresDAO:
    """Data Access Object for PostgreSQL database interactions."""

class PostgresDAO:
    def __init__(self, settings: Settings = Settings()) -> None:
        self.pool: Pool | None = None
        self.settings = settings

    async def setup(self) -> None:
        try:
            self.pool = await create_pool(
                dsn=self.settings.BACKTEST_DB_URL,
                min_size=1,
                max_size=10,
            )
            logger.info("PostgresDAO pool initialized")
        except PostgresError as e:
            logger.exception("Failed to initialize pool")
            raise RuntimeError("Database pool initialization failed: %s" % str(e)) from e

    async def get_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch BTC price data from the database for a given date range.

        Args:
            start_date: Start date in ISO format (e.g., '2023-01-01').
            end_date: End date in ISO format (e.g., '2023-12-31').

        Returns:
            A pandas DataFrame containing price data or an empty DataFrame if no data is found.

        Raises:
            ValueError: If the database pool is not initialized.
            PostgresError: If a database error occurs.
        """
        if not self.pool:
            raise ValueError(POOL_NOT_INITIALIZED)
        query = """
            SELECT timestamp, open, high, low, close, r_1, r_2
            FROM btc_prices
            WHERE timestamp >= $1 AND timestamp <= $2
            ORDER BY timestamp
        """
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query, pd.to_datetime(start_date), pd.to_datetime(end_date))
        except PostgresError as _:
            logger.exception("Error fetching data")
            return pd.DataFrame()
        else:
            if not rows:
                logger.warning("No data found for %s to %s", start_date, end_date)
                return pd.DataFrame()
            price_data = pd.DataFrame(
                rows,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'r_1', 'r_2']
            )
            logger.info("Fetched %d price records", len(price_data))
            return price_data
