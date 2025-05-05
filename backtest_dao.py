"""Data Access Object for managing backtest results and signal events in PostgreSQL."""
from __future__ import annotations
import logging
from typing import Any

import asyncpg
import pandas as pd
from asyncpg.exceptions import PostgresError

from dao import PostgresDAO

logger = logging.getLogger(__name__)

BACKTEST_RESULTS_TABLE = "backtest_results"
SIGNAL_EVENTS_TABLE = "signal_events"
REQUIRED_RESULT_KEYS = {"final_portfolio_value", "returns", "start_date", "end_date"}
REQUIRED_EVENT_KEYS = {"timestamp", "event_type", "price"}
POOL_NOT_INITIALIZED = "Database pool not initialized"

class BacktestDAO:
    """DAO for backtest results and signal events."""

    def __init__(self) -> None:
        """Initialize the BacktestDAO with a PostgresDAO instance."""
        self.dao = PostgresDAO()

    async def setup(self) -> None:
        """Set up the underlying PostgresDAO.

        Raises:
            PostgresError: If the database pool initialization fails.
        """
        await self.dao.setup()
        logger.info("BacktestDAO setup complete")

    async def close_pool(self) -> None:
        """Close the database connection pool."""
        if self.dao.pool:
            await self.dao.pool.close()
            logger.info("BacktestDAO pool closed")

    async def save_backtest_result(self, backtest_id: str, result: dict[str, Any]) -> None:
        """Save a backtest result to the database.

        Args:
            backtest_id: Unique identifier for the backtest.
            result: Dictionary containing backtest results (final_portfolio_value, returns, start_date, end_date).

        Raises:
            ValueError: If required keys are missing in result.
            PostgresError: If a database error occurs.
        """
        if not all(key in result for key in REQUIRED_RESULT_KEYS):
            raise ValueError("Result dictionary missing required keys: %s" % REQUIRED_RESULT_KEYS)
        
        query = """
            INSERT INTO backtest_results (backtest_id, final_portfolio_value, returns, start_date, end_date)
            VALUES ($1, $2, $3, $4, $5)
        """
        try:
            async with self.dao.pool.acquire() as conn:
                await conn.execute(
                    query,
                    backtest_id,
                    float(result["final_portfolio_value"]),
                    float(result["returns"]),
                    pd.to_datetime(result["start_date"]),
                    pd.to_datetime(result["end_date"]),
                )
            logger.info("Saved backtest result for ID %s", backtest_id)
        except PostgresError as _:
            logger.exception("Failed to save backtest result")
            raise

    async def save_signal_event(self, backtest_id: str, event: dict[str, Any]) -> None:
        """Save a signal event to the database.

        Args:
            backtest_id: Unique identifier for the backtest.
            event: Dictionary containing signal event (timestamp, event_type, price).

        Raises:
            ValueError: If required keys are missing in event.
            PostgresError: If a database error occurs.
        """
        if not all(key in event for key in REQUIRED_EVENT_KEYS):
            raise ValueError("Event dictionary missing required keys: %s" % REQUIRED_EVENT_KEYS)
        
        query = """
            INSERT INTO signal_events (backtest_id, timestamp, event_type, price)
            VALUES ($1, $2, $3, $4)
        """
        try:
            async with self.dao.pool.acquire() as conn:
                await conn.execute(
                    query,
                    backtest_id,
                    pd.to_datetime(event["timestamp"]),
                    event["event_type"],
                    float(event["price"]),
                )
            logger.info("Saved signal event for backtest ID %s at %s", backtest_id, event["timestamp"])
        except PostgresError as _:
            logger.exception("Failed to save signal event")
            raise

    async def get_plot_data(self, start_date: str, end_date: str) -> list[dict]:
        """Fetch plot data for performance visualization.

        Args:
            start_date: Start date in YYYY-MM-DD format.
            end_date: End date in YYYY-MM-DD format.

        Returns:
            List of dictionaries containing timestamp and portfolio_value.

        Raises:
            ValueError: If the database pool is not initialized.
            PostgresError: If a database error occurs.
        """
        if not self.dao.pool:
            raise ValueError(POOL_NOT_INITIALIZED)
        
        query = """
            SELECT timestamp, portfolio_value
            FROM backtest_results
            WHERE timestamp >= $1 AND timestamp <= $2
            ORDER BY timestamp
        """
        try:
            async with self.dao.pool.acquire() as conn:
                rows = await conn.fetch(query, pd.to_datetime(start_date), pd.to_datetime(end_date))
            return [dict(row) for row in rows]
        except PostgresError as _:
            logger.exception("Failed to fetch plot data")
            return []

    async def get_signal_events(self, start_date: str, end_date: str) -> list[dict]:
        """Fetch signal events (e.g., buy/sell) for a date range.

        Args:
            start_date: Start date in YYYY-MM-DD format.
            end_date: End date in YYYY-MM-DD format.

        Returns:
            List of dictionaries containing timestamp, event_type, and price.

        Raises:
            ValueError: If the database pool is not initialized.
            PostgresError: If a database error occurs.
        """
        if not self.dao.pool:
            raise ValueError(POOL_NOT_INITIALIZED)
        
        query = """
            SELECT timestamp, event_type, price
            FROM signal_events
            WHERE timestamp >= $1 AND timestamp <= $2
            ORDER BY timestamp
        """
        try:
            async with self.dao.pool.acquire() as conn:
                rows = await conn.fetch(query, pd.to_datetime(start_date), pd.to_datetime(end_date))
            return [dict(row) for row in rows]
        except PostgresError as _:
            logger.exception("Failed to fetch signal events")
            return []
