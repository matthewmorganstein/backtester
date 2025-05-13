"""Data Access Object for backtest results and signal events."""
from __future__ import annotations

import logging
from typing import Any

import asyncpg
from asyncpg.exceptions import PostgresError
from pandas import DataFrame
from settings import Settings

logger = logging.getLogger(__name__)

class BacktestDAO:
    """Handles storage and retrieval of backtest results and events."""
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
            logger.info("BacktestDAO pool initialized")
        except PostgresError as e:
            logger.exception("Failed to create BacktestDAO pool")
            raise RuntimeError("BacktestDAO pool creation failed") from e

    async def close_pool(self) -> None:
        """Close the connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("BacktestDAO pool closed")

    async def save_backtest_result(self, result: dict[str, Any]) -> None:
        """Save backtest result to the database."""
        if not self.pool:
            logger.error("BacktestDAO pool not initialized")
            raise RuntimeError("BacktestDAO pool not initialized")
        required_keys = {"backtest_id", "start_date", "end_date", "final_portfolio_value"}
        if not all(key in result for key in required_keys):
            logger.error(f"Missing required keys in backtest result: {required_keys}")
            raise ValueError("Missing required keys in backtest result")
        query = """
            INSERT INTO backtest_results (
                backtest_id, start_date, end_date, final_portfolio_value,
                returns, parameters
            )
            VALUES ($1, $2, $3, $4, $5, $6)
        """
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    query,
                    result["backtest_id"],
                    result["start_date"],
                    result["end_date"],
                    result["final_portfolio_value"],
                    result.get("returns"),
                    result.get("parameters"),
                )
                logger.debug(f"Saved backtest result: {result['backtest_id']}")
        except PostgresError as e:
            logger.exception("Failed to save backtest result")
            raise RuntimeError("Failed to save backtest result") from e

    async def save_signal_event(self, backtest_id: str, event: dict[str, Any]) -> None:
        """Save a signal event to the database."""
        if not self.pool:
            logger.error("BacktestDAO pool not initialized")
            raise RuntimeError("BacktestDAO pool not initialized")
        required_keys = {"trade_id", "timestamp", "event_type", "price"}
        if not all(key in event for key in required_keys):
            logger.error(f"Missing required keys in signal event: {required_keys}")
            raise ValueError("Missing required keys in signal event")
        query = """
            INSERT INTO signal_events (
                backtest_id, trade_id, timestamp, event_type, price, details
            )
            VALUES ($1, $2, $3, $4, $5, $6)
        """
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    query,
                    backtest_id,
                    event["trade_id"],
                    event["timestamp"],
                    event["event_type"],
                    event["price"],
                    event.get("details"),
                )
                logger.debug(f"Saved signal event: {backtest_id}, {event['trade_id']}")
        except PostgresError as e:
            logger.exception("Failed to save signal event")
            raise RuntimeError("Failed to save signal event") from e

    async def save_portfolio_update(self, backtest_id: str, update: dict[str, Any]) -> None:
        """Save a portfolio update to the database."""
        if not self.pool:
            logger.error("BacktestDAO pool not initialized")
            raise RuntimeError("BacktestDAO pool not initialized")
        required_keys = {"timestamp", "portfolio_value"}
        if not all(key in update for key in required_keys):
            logger.error(f"Missing required keys in portfolio update: {required_keys}")
            raise ValueError("Missing required keys in portfolio update")
        query = """
            INSERT INTO portfolio_updates (
                backtest_id, timestamp, portfolio_value, cash, position_size,
                position_value, details
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7)
        """
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    query,
                    backtest_id,
                    update["timestamp"],
                    update["portfolio_value"],
                    update.get("cash"),
                    update.get("position_size"),
                    update.get("position_value"),
                    update.get("details"),
                )
                logger.debug(f"Saved portfolio update: {backtest_id}, {update['timestamp']}")
        except PostgresError as e:
            logger.exception("Failed to save portfolio update")
            raise RuntimeError("Failed to save portfolio update") from e

    async def get_plot_data(self, backtest_id: str, start_date: str, end_date: str) -> list[dict]:
        """Fetch portfolio updates for visualization."""
        if not self.pool:
            logger.error("BacktestDAO pool not initialized")
            raise RuntimeError("BacktestDAO pool not initialized")
        query = """
            SELECT timestamp, portfolio_value, cash, position_size, position_value
            FROM portfolio_updates
            WHERE backtest_id = $1 AND timestamp >= $2 AND timestamp <= $3
            ORDER BY timestamp
        """
        try:
            async with self.pool.acquire() as conn:
                records = await conn.fetch(query, backtest_id, start_date, end_date)
                result = [
                    {
                        "timestamp": record["timestamp"],
                        "portfolio_value": record["portfolio_value"],
                        "cash": record["cash"],
                        "position_size": record["position_size"],
                        "position_value": record["position_value"],
                    }
                    for record in records
                ]
                logger.debug(f"Fetched {len(result)} portfolio updates for {backtest_id}")
                return result
        except PostgresError as e:
            logger.exception(f"Failed to fetch plot data for {backtest_id}")
            raise RuntimeError("Failed to fetch plot data") from e

    async def get_signal_events(self, backtest_id: str, start_date: str, end_date: str) -> list[dict]:
        """Fetch signal events for visualization."""
        if not self.pool:
            logger.error("BacktestDAO pool not initialized")
            raise RuntimeError("BacktestDAO pool not initialized")
        query = """
            SELECT trade_id, timestamp, event_type, price, details
            FROM signal_events
            WHERE backtest_id = $1 AND timestamp >= $2 AND timestamp <= $3
            ORDER BY timestamp
        """
        try:
            async with self.pool.acquire() as conn:
                records = await conn.fetch(query, backtest_id, start_date, end_date)
                result = [
                    {
                        "trade_id": record["trade_id"],
                        "timestamp": record["timestamp"],
                        "event_type": record["event_type"],
                        "price": record["price"],
                        "details": record["details"],
                    }
                    for record in records
                ]
                logger.debug(f"Fetched {len(result)} signal events for {backtest_id}")
                return result
        except PostgresError as e:
            logger.exception(f"Failed to fetch signal events for {backtest_id}")
            raise RuntimeError("Failed to fetch signal events") from e
