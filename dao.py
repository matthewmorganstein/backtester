import pandas as pd
from asyncpg import Pool, create_pool
from asyncpg.exceptions import PostgresError
import logging
import settings

logger = logging.getLogger(__name__)

class PostgresDAO:
    def __init__(self):
        self.pool: Pool | None = None

    async def setup(self):
        try:
            self.pool = await create_pool(
                dsn=settings.BACKTEST_DB_URL,
                min_size=1,
                max_size=10,
            )
            logger.info("PostgresDAO pool initialized")
        except PostgresError as e:
            logger.error(f"Failed to initialize pool: {e}")
            raise

    async def get_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        if not self.pool:
            raise ValueError("Database pool not initialized")
        query = """
            SELECT timestamp, open, high, low, close, r_1, r_2
            FROM btc_prices
            WHERE timestamp >= $1 AND timestamp <= $2
            ORDER BY timestamp
        """
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query, pd.to_datetime(start_date), pd.to_datetime(end_date))
            if not rows:
                logger.warning(f"No data found for {start_date} to {end_date}")
                return pd.DataFrame()
            df = pd.DataFrame(rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'r_1', 'r_2'])
            logger.info(f"Fetched {len(df)} price records")
            return df
        except PostgresError as e:
            logger.error(f"Error fetching data: {e}")
            return pd.DataFrame()
