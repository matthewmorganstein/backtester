"""Utilities for API key verification and configuration."""
from __future__ import annotations
import logging
import os
from typing import Optional

from dotenv import load_dotenv
from fastapi import Depends, Header, HTTPException

from settings import Settings, settings

load_dotenv()

def configure_logging() -> None:
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

configure_logging()
logger = logging.getLogger(__name__)

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    logger.error("API_KEY not set in environment variables")
    raise RuntimeError("API_KEY not configured")
VALID_API_KEYS = {API_KEY}
VALID_API_KEYS_LOWER = {key.lower() for key in VALID_API_KEYS}


def get_settings() -> Settings:
    """Provide settings as a dependency."""
    return settings

async def verify_api_key(
    x_api_key: Optional[str] = Header(None),
) -> str:
    """Verify the API key provided in the X-API-Key header.

    Args:
        x_api_key: API key from the X-API-Key header.

    Returns:
        The validated API key.

    Raises:
        HTTPException: If the API key is missing or invalid.
    """
    # TODO: Implement rate limiting and key rotation for production
    if not x_api_key:
        logger.warning("API key missing in request")
        raise HTTPException(status_code=401, detail="API key missing")
    if x_api_key.lower() not in VALID_API_KEYS_LOWER:
        logger.warning("Invalid API key provided")
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key
