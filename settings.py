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

    model_config = SettingsConfigDict(
        env_prefix="BACKTESTING_API_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
