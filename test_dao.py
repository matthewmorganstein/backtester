"""Unit tests for PostgresDAO."""
import logging
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock
import pandas as pd
import pytest
from asyncpg import Pool
from asyncpg.exceptions import PostgresError

from dao import PostgresDAO, POOL_NOT_INITIALIZED

@pytest.fixture
def dao():
    """Create a PostgresDAO instance for testing."""
    dao = PostgresDAO()
    dao.pool = AsyncMock(spec=Pool)
    return dao

@pytest.mark.asyncio
async def test_setup_success(monkeypatch):
    """Test successful pool initialization."""
    mock_create_pool = AsyncMock(return_value=MagicMock(spec=Pool))
    monkeypatch.setattr("dao.create_pool", mock_create_pool)

    dao = PostgresDAO()
    await dao.setup()

    assert dao.pool is not None
    mock_create_pool.assert_awaited_once()

@pytest.mark.asyncio
async def test_setup_failure(monkeypatch):
    """Test pool initialization failure."""
    mock_create_pool = AsyncMock(side_effect=PostgresError("Connection failed"))
    monkeypatch.setattr("dao.create_pool", mock_create_pool)

    dao = PostgresDAO()
    with pytest.raises(RuntimeError, match="Database pool initialization failed"):
        await dao.setup()

@pytest.mark.asyncio
async def test_get_data_success(dao):
    """Test successful data retrieval."""
    mock_conn = AsyncMock()
    mock_conn.fetch = AsyncMock(
        return_value=[
            {"timestamp": datetime(2023, 1, 1), "open": 100, "high": 110, "low": 90, "close": 105, "r_1": 1.0, "r_2": 2.0}
        ]
    )
    dao.pool.acquire = AsyncMock(return_value=mock_conn)

    result = await dao.get_data("2023-01-01", "2023-01-02")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    assert list(result.columns) == ["timestamp", "open", "high", "low", "close", "r_1", "r_2"]
    mock_conn.fetch.assert_awaited_once()

@pytest.mark.asyncio
async def test_get_data_no_pool(dao):
    """Test get_data when pool is not initialized."""
    dao.pool = None
    with pytest.raises(ValueError, match=POOL_NOT_INITIALIZED):
        await dao.get_data("2023-01-01", "2023-01-02")

@pytest.mark.asyncio
async def test_get_data_no_data(dao):
    """Test get_data when no data is found."""
    mock_conn = AsyncMock()
    mock_conn.fetch = AsyncMock(return_value=[])
    dao.pool.acquire = AsyncMock(return_value=mock_conn)

    result = await dao.get_data("2023-01-01", "2023-01-02")

    assert isinstance(result, pd.DataFrame)
    assert result.empty
    mock_conn.fetch.assert_awaited_once()

@pytest.mark.asyncio
async def test_get_data_postgres_error(dao, caplog):
    """Test get_data when a PostgresError occurs."""
    mock_conn = AsyncMock()
    mock_conn.fetch = AsyncMock(side_effect=PostgresError("Query failed"))
    dao.pool.acquire = AsyncMock(return_value=mock_conn)

    with caplog.at_level(logging.ERROR):
        result = await dao.get_data("2023-01-01", "2023-01-02")

    assert isinstance(result, pd.DataFrame)
    assert result.empty
    assert "Error fetching data" in caplog.text
    mock_conn.fetch.assert_awaited_once()

if __name__ == "__main__":
    pytest.main(["-v"])
