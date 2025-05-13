"""Utility functions for the Backtest API."""
import logging
from typing import Any

from fastapi import HTTPException, Request, status
from settings import Settings

logger = logging.getLogger(__name__)

def configure_logging(settings: Settings) -> None:
    """Configure logging with console and file handlers."""
    logging.basicConfig(
        level=settings.LOG_LEVEL,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler("backtest_api.log")
    file_handler.setLevel(settings.LOG_LEVEL)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logging.getLogger().addHandler(file_handler)
    logger.info("Logging configured with console and file handlers")

async def verify_api_key(request: Request, settings: Settings | None = None) -> None:
    """Verify the API key in the request headers."""
    settings = settings or Settings()
    api_key = request.headers.get("X-API-Key")
    if not api_key or api_key != settings.API_KEY:
        logger.warning("Invalid or missing API key")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
