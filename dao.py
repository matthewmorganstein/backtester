"""Data Access Object for interacting with the PostgreSQL database."""
from __future__ import annotations

import logging
from typing import Any

import asyncpg
from asyncpg.exceptions import PostgresError
from pandas import DataFrame
from settings import Settings

logger = logging.getLogger(__name__)

class PostgresDAO:
    """Handles PostgreSQL database interactions."""
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.pool: asyncpg.Pool | None = None

    async def setup(self) -> None:
        """Initialize the connection pool."""
        try:
            self.pool = await asyncpg.create_pool(
                dsn=self.settings.DATABASE_URL,
                min_size=self.settings.DB_POOL_MIN_SIZE,
                max_size=self.settings.DB_POOL_MAX_SIZE,
            )
            logger.info("Database pool initialized")
        except PostgresError as e:
            logger.exception("Failed to create database pool")
            raise RuntimeError("Database pool creation failed") from e

    async def close_pool(self) -> None:
        """Close the connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("Database pool closed")

    async def get_data(self, start_date: str, end_date: str) -> DataFrame:
        """Fetch market data for a given date range."""
        if not self.pool:
            logger.error("Database pool not initialized")
            raise RuntimeError("Database pool not initialized")
        query = """
            SELECT timestamp, open, high, low, close, volume, r_1, r_2
            FROM btc_prices
            WHERE timestamp >= $1 AND timestamp <= $2
            ORDER BY timestamp
        """
        try:
            async with self.pool.acquire() as conn:
                records = await conn.fetch(query, start_date, end_date)
                if not records:
                    logger.warning(f"No data found for {start_date} to {end_date}")
                    return DataFrame()
                df = DataFrame(records)
                df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")
                logger.debug(f"Fetched {len(df)} rows from {start_date} to {end_date}")
                return df
        except PostgresError as e:
            logger.exception(f"Database error fetching data for {start_date} to {end_date}")
            raise RuntimeError("Failed to fetch data") from e
