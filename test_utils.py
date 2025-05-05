"""Unit tests for utilities."""
from __future__ import annotations
import logging
import os
from typing import Optional

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from settings import Settings
from utils import configure_logging, get_settings, verify_api_key

@pytest.fixture
def mock_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock environment variables."""
    monkeypatch.setenv("API_KEY", "test-api-key")

@pytest.fixture
def settings() -> Settings:
    """Create a mock Settings instance."""
    class MockSettings:
        API_KEY = "test-api-key"
    return MockSettings()  # type: ignore

def test_configure_logging(caplog: pytest.LogCaptureFixture) -> None:
    """Test logging configuration."""
    caplog.set_level(logging.INFO)
    configure_logging()
    logger = logging.getLogger("test_utils")
    logger.info("Test log message")
    assert "Test log message" in caplog.text

def test_get_settings(settings: Settings) -> None:
    """Test retrieving settings."""
    result = get_settings()
    assert result.API_KEY == "test-api-key"

@pytest.mark.asyncio
async def test_verify_api_key_valid(mock_env: None, settings: Settings) -> None:
    """Test verifying a valid API key."""
    api_key = "test-api-key"
    result = await verify_api_key(x_api_key=api_key, settings=settings)
    assert result == api_key

@pytest.mark.asyncio
async def test_verify_api_key_case_insensitive(mock_env: None, settings: Settings) -> None:
    """Test verifying a case-insensitive API key."""
    api_key = "TEST-API-KEY"
    result = await verify_api_key(x_api_key=api_key, settings=settings)
    assert result == api_key

@pytest.mark.asyncio
async def test_verify_api_key_missing(mock_env: None, settings: Settings) -> None:
    """Test verifying a missing API key."""
    with pytest.raises(HTTPException) as exc:
        await verify_api_key(x_api_key=None, settings=settings)
    assert exc.value.status_code == 401
    assert exc.value.detail == "API key missing"

@pytest.mark.asyncio
async def test_verify_api_key_invalid(mock_env: None, settings: Settings) -> None:
    """Test verifying an invalid API key."""
    with pytest.raises(HTTPException) as exc:
        await verify_api_key(x_api_key="wrong-key", settings=settings)
    assert exc.value.status_code == 401
    assert exc.value.detail == "Invalid API key"

def test_api_key_not_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test behavior when API_KEY is not set."""
    monkeypatch.delenv("API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="API_KEY not configured"):
        from utils import VALID_API_KEYS  # noqa: F401

if __name__ == "__main__":
    pytest.main(["-v"])
