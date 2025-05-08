"""Data Access Object (DAO) for interacting with PostgreSQL database."""
from __future__ import annotations
import logging
from typing import Union, Optional
from settings import Settings
import asyncpg
import pandas as pd
from asyncpg import Pool, create_pool
from asyncpg.exceptions import PostgresError

logger = logging.getLogger(__name__)
POOL_NOT_INITIALIZED = "Database pool not initialized"


class PostgresDAO:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        """Initialize PostgresDAO with optional settings."""
        self.settings = settings or Settings()
        self.pool: Pool | None = None

    async def setup(self) -> None:
        """Set up the database connection pool."""
        try:
            self.pool = await create_pool(
                dsn=self.settings.BACKTEST_DB_URL,
                min_size=1,
                max_size=10,
            )
            logger.info("PostgresDAO pool initialized")
        except PostgresError as e:
            logger.exception("Failed to initialize pool")
            raise RuntimeError(f"Database pool initialization failed: {e}")

    async def get_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch BTC price data from the database for a given date range."""

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
                rows = await conn.fetch(query, pd.to_datetime(start_date),
                                        pd.to_datetime(end_date))
        except PostgresError as _:
            logger.exception("Error fetching data")
            return pd.DataFrame()
        else:
            if not rows:
                logger.warning("No data found for %s to %s", start_date, end_date)
                return pd.DataFrame()
            price_data = pd.DataFrame(
                rows,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'r_1', 'r_2',]
            )
            logger.info("Fetched %d price records", len(price_data))
            return price_data

    async def close_pool(self) -> None:
        """Close the database connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("PostgresDAO pool closed")
        else:
            logger.warning("Pool already closed or not initialized.")