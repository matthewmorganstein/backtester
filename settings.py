"""Application settings for the backtesting API."""
from __future__ import annotations
import os
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

# Error message constants
MISSING_RISE_DB_URL = "RISE_DB_URL environment variable is not set"
MISSING_BACKTEST_DB_URL = "BACKTEST_DB_URL environment variable is not set"
MISSING_API_KEY = "API_KEY environment variable is not set"

DEFAULT_PORT = 8000

class Settings(BaseSettings):
    """Configuration settings for the backtesting API."""

    BACKTEST_DB_URL: str | None = None
    RISE_DB_URL: str | None = None
    API_KEY: str | None = None
    PORT: int = DEFAULT_PORT

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def validate_settings(self) -> None:
        """Validate critical settings."""
        if not self.RISE_DB_URL:
            raise ValueError(MISSING_RISE_DB_URL)
        if not self.BACKTEST_DB_URL:
            raise ValueError(MISSING_BACKTEST_DB_URL)
        if not self.API_KEY:
            raise ValueError(MISSING_API_KEY)
