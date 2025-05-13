"""Application settings and configuration."""
from __future__ import annotations

from pydantic import BaseSettings, PostgresDsn, ValidationInfo, field_validator
from typing import Any, List

class Settings(BaseSettings):
    """Application settings validated by Pydantic."""
    API_KEY: str
    DB_HOST: str
    DB_PORT: int = 5432
    DB_USER: str
    DB_PASSWORD: str
    DB_NAME: str
    DB_POOL_MIN_SIZE: int = 5
    DB_POOL_MAX_SIZE: int = 20
    DATABASE_URL: PostgresDsn | None = None
    MAX_DATE_RANGE_DAYS: int = 30
    LOG_LEVEL: str = "INFO"
    PORT: int = 8000
    HOST: str = "127.0.0.1"
    CORS_ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    CORS_ALLOWED_METHODS: List[str] = ["GET", "POST", "OPTIONS"]
    CORS_ALLOWED_HEADERS: List[str] = ["Authorization", "Content-Type", "Accept", "X-API-Key"]
    TARGET_BARS: int = 1000
    INITIAL_CASH: float = 100_000
    TIME_ABOVE_STOP_THRESHOLD: float = 30
    BUY_SIGNAL: int = 1
    SELL_SIGNAL: int = -1
    HOLD_SIGNAL: int = 0

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def build_database_url(cls, v: Any, info: ValidationInfo) -> str:
        """Build DATABASE_URL from individual DB components if not provided."""
        if v is not None:
            return str(v)
        values = info.data
        return PostgresDsn.build(
            scheme="postgresql+asyncpg",
            host=values.get("DB_HOST", ""),
            port=values.get("DB_PORT", 5432),
            user=values.get("DB_USER", ""),
            password=values.get("DB_PASSWORD", ""),
            path=f"/{values.get('DB_NAME', '')}",
        ).unicode_string()

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Ensure LOG_LEVEL is valid."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of {valid_levels}")
        return v.upper()

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
