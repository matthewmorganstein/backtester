from asyncpg import create_pool
import logging
from typing import List, Dict
from datetime import datetime
from settings import settings

logger = logging.getLogger(__name__)

class BacktestDAO:
    """DAO for the FlatFire container's backtest results storage."""
    def __init__(self, connection_string: str = settings.FLATFIRE_DB_URL):
        self.connection_string = connection_string
        self.pool = None

    async def setup(self):
        """Initialize the DAO, setting up the connection pool."""
        await self.init_pool()

    async def init_pool(self):
        """Initialize the database connection pool."""
        if not self.pool:
            try:
                self.pool = await create_pool(self.connection_string)
                logger.info("BacktestDAO pool initialized")
            except Exception as e:
                logger.error(f"Failed to initialize pool: {e}")
                raise

    async def close_pool(self):
        if self.pool:
            try:
                await self.pool.close()
                logger.info("BacktestDAO pool closed")
                self.pool = None
            except Exception as e:
                logger.error(f"Failed to close pool: {e}")
                raise

    async def store_result(self, timestamp: datetime, event_type: str, details: Dict) -> None:
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO backtest_results (timestamp, event_type, details)
                    VALUES ($1, $2, $3)
                """, timestamp, event_type, details)
        except Exception as e:
            logger.error(f"Error storing result: {e}")
            raise

    async def get_plot_data(self, start_date: str, end_date: str, limit: int = 1000, offset: int = 0) -> List[Dict]:
        async with self.pool.acquire() as conn:
            records = await conn.fetch("""
                SELECT timestamp, event_type, details
                FROM backtest_results
                WHERE timestamp BETWEEN $1 AND $2
                ORDER BY timestamp
                LIMIT $3 OFFSET $4
            """, datetime.strptime(start_date, "%Y-%m-%d"), datetime.strptime(end_date, "%Y-%m-%d"), limit, offset)
            return [{
                    "timestamp": r["timestamp"], 
                    "event_type": r["event_type"], 
                    "details": dict(r["details"])
                } for r in records]

    async def get_signal_events(self, start_date: str, end_date: str, limit: int = 1000, offset: int = 0) -> List[Dict]:
        """Fetch buy_signal and sell_signal events from backtest_results."""
        try:
            async with self.pool.acquire() as conn:
                records = await conn.fetch("""
                    SELECT timestamp, event_type, details
                    FROM backtest_results
                    WHERE event_type IN ('buy_signal', 'sell_signal')
                    AND timestamp BETWEEN $1 AND $2
                    ORDER BY timestamp
                """, datetime.strptime(start_date, "%Y-%m-%d"), datetime.strptime(end_date, "%Y-%m-%d"))
                result = [{
                    "timestamp": r["timestamp"],
                    "event_type": r["event_type"],
                    "details": dict(r["details"])
                } for r in records]
                logger.info(f"Fetched {len(result)} signal events from {start_date} to {end_date}")
                return result
        except Exception as e:
            logger.error(f"Error fetching signal events: {e}")
            raise
