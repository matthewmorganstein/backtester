"""Application settings for the backtesting API."""
from __future__ import annotations

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_PORT = 8000

class Settings(BaseSettings):
    """Configuration settings for the backtesting API."""
    RISE_DB_URL: str
    BACKTEST_DB_URL: str
    API_KEY: SecretStr
    PORT: int = DEFAULT_PORT
    DB_POOL_MIN_SIZE: int = 1
    DB_POOL_MAX_SIZE: int = 10
    MAX_DATE_RANGE_DAYS: int = 30
    CORS_ALLOWED_ORIGINS: list[str] = ["*"] # Default to all, but ideally specify
    CORS_ALLOWED_METHODS: list[str] = # More specific default
    CORS_ALLOWED_HEADERS: list[str] = # Common headers
    LOG_LEVEL: str = "INFO"
    TARGET_BARS: int = 1000
    INITIAL_CASH: float = 100_000
    TIME_ABOVE_STOP_THRESHOLD: float = 30
    BUY_SIGNAL: int = 1
    SELL_SIGNAL: int = -1
    HOLD_SIGNAL: int = 0

    model_config = SettingsConfigDict(
        env_prefix="BACKTESTING_API_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
