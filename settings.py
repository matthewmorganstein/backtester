"""Application settings for the backtesting API."""
from __future__ import annotations
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_PORT = 8000

class Settings(BaseSettings):
    """Configuration settings for the backtesting API."""
    BACKTEST_DB_URL: Optional[str] = None
    RISE_DB_URL: Optional[str] = None
    API_KEY: Optional[str] = None
    PORT: int = DEFAULT_PORT

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def validate_settings(self) -> None:
        """Validate critical settings."""
        if not self.RISE_DB_URL:
            raise ValueError("RISE_DB_URL environment variable is not set")
        if not self.BACKTEST_DB_URL:
            raise ValueError("BACKTEST_DB_URL environment variable is not set")
        if not self.API_KEY:
            raise ValueError("API_KEY environment variable is not set")
