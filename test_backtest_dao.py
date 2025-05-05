"""Unit tests for BacktestDAO."""
from __future__ import annotations
import logging
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest
from asyncpg.exceptions import PostgresError

from backtest_dao import BacktestDAO, REQUIRED_RESULT_KEYS, REQUIRED_EVENT_KEYS, POOL_NOT_INITIALIZED

@pytest.fixture
def dao():
    """Create a BacktestDAO instance with a mocked PostgresDAO."""
    dao = BacktestDAO()
    dao.dao = MagicMock()
    dao.dao.pool = AsyncMock()
    dao.dao.setup = AsyncMock()
    dao.dao.pool.close = AsyncMock()
    return dao

@pytest.mark.asyncio
async def test_setup_success(dao):
    """Test successful setup."""
    await dao.setup()
    dao.dao.setup.assert_awaited_once()

@pytest.mark.asyncio
async def test_close_pool_success(dao):
    """Test successful pool closure."""
    await dao.close_pool()
    dao.dao.pool.close.assert_awaited_once()

@pytest.mark.asyncio
async def test_save_backtest_result_success(dao):
    """Test saving a backtest result."""
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock()
    dao.dao.pool.acquire = AsyncMock(return_value=mock_conn)
    
    result = {
        "final_portfolio_value": 1000.0,
        "returns": 0.1,
        "start_date": "2023-01-01",
        "end_date": "2023-01-02",
    }
    await dao.save_backtest_result("test_id", result)
    
    mock_conn.execute.assert_awaited_once()

@pytest.mark.asyncio
async def test_save_backtest_result_missing_keys(dao):
    """Test saving a result with missing keys."""
    result = {"final_portfolio_value": 1000.0}
    with pytest.raises(ValueError, match="Result dictionary missing required keys: %s" % REQUIRED_RESULT_KEYS):
        await dao.save_backtest_result("test_id", result)

@pytest.mark.asyncio
async def test_save_backtest_result_postgres_error(dao, caplog):
    """Test saving a result with a PostgresError."""
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(side_effect=PostgresError("Insert failed"))
    dao.dao.pool.acquire = AsyncMock(return_value=mock_conn)
    
    result = {
        "final_portfolio_value": 1000.0,
        "returns": 0.1,
        "start_date": "2023-01-01",
        "end_date": "2023-01-02",
    }
    with caplog.at_level(logging.ERROR):
        with pytest.raises(PostgresError):
            await dao.save_backtest_result("test_id", result)
    assert "Failed to save backtest result" in caplog.text

@pytest.mark.asyncio
async def test_save_signal_event_success(dao):
    """Test saving a signal event."""
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock()
    dao.dao.pool.acquire = AsyncMock(return_value=mock_conn)
    
    event = {
        "timestamp": "2023-01-01 12:00:00",
        "event_type": "buy",
        "price": 100.0,
    }
    await dao.save_signal_event("test_id", event)
    
    mock_conn.execute.assert_awaited_once()

@pytest.mark.asyncio
async def test_save_signal_event_missing_keys(dao):
    """Test saving a signal event with missing keys."""
    event = {"timestamp": "2023-01-01"}
    with pytest.raises(ValueError, match="Event dictionary missing required keys: %s" % REQUIRED_EVENT_KEYS):
        await dao.save_signal_event("test_id", event)

@pytest.mark.asyncio
async def test_save_signal_event_postgres_error(dao, caplog):
    """Test saving a signal event with a PostgresError."""
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(side_effect=PostgresError("Insert failed"))
    dao.dao.pool.acquire = AsyncMock(return_value=mock_conn)
    
    event = {
        "timestamp": "2023-01-01 12:00:00",
        "event_type": "buy",
        "price": 100.0,
    }
    with caplog.at_level(logging.ERROR):
        with pytest.raises(PostgresError):
            await dao.save_signal_event("test_id", event)
    assert "Failed to save signal event" in caplog.text

@pytest.mark.asyncio
async def test_get_plot_data_success(dao):
    """Test fetching plot data."""
    mock_conn = AsyncMock()
    mock_conn.fetch = AsyncMock(return_value=[{"timestamp": datetime(2023, 1, 1), "portfolio_value": 1000}])
    dao.dao.pool.acquire = AsyncMock(return_value=mock_conn)
    
    result = await dao.get_plot_data("2023-01-01", "2023-01-02")
    
    assert len(result) == 1
    assert result[0]["portfolio_value"] == 1000
    mock_conn.fetch.assert_awaited_once()

@pytest.mark.asyncio
async def test_get_plot_data_no_pool(dao):
    """Test fetching plot data with no pool."""
    dao.dao.pool = None
    with pytest.raises(ValueError, match=POOL_NOT_INITIALIZED):
        await dao.get_plot_data("2023-01-01", "2023-01-02")

@pytest.mark.asyncio
async def test_get_plot_data_postgres_error(dao, caplog):
    """Test fetching plot data with a PostgresError."""
    mock_conn = AsyncMock()
    mock_conn.fetch = AsyncMock(side_effect=PostgresError("Query failed"))
    dao.dao.pool.acquire = AsyncMock(return_value=mock_conn)
    
    with caplog.at_level(logging.ERROR):
        result = await dao.get_plot_data("2023-01-01", "2023-01-02")
    
    assert result == []
    assert "Failed to fetch plot data" in caplog.text

@pytest.mark.asyncio
async def test_get_signal_events_success(dao):
    """Test fetching signal events."""
    mock_conn = AsyncMock()
    mock_conn.fetch = AsyncMock(return_value=[{"timestamp": datetime(2023, 1, 1), "event_type": "buy", "price": 100}])
    dao.dao.pool.acquire = AsyncMock(return_value=mock_conn)
    
    result = await dao.get_signal_events("2023-01-01", "2023-01-02")
    
    assert len(result) == 1
    assert result[0]["event_type"] == "buy"
    mock_conn.fetch.assert_awaited_once()

@pytest.mark.asyncio
async def test_get_signal_events_no_pool(dao):
    """Test fetching signal events with no pool."""
    dao.dao.pool = None
    with pytest.raises(ValueError, match=POOL_NOT_INITIALIZED):
        await dao.get_signal_events("2023-01-01", "2023-01-02")

@pytest.mark.asyncio
async def test_get_signal_events_postgres_error(dao, caplog):
    """Test fetching signal events with a PostgresError."""
    mock_conn = AsyncMock()
    mock_conn.fetch = AsyncMock(side_effect=PostgresError("Query failed"))
    dao.dao.pool.acquire = AsyncMock(return_value=mock_conn)
    
    with caplog.at_level(logging.ERROR):
        result = await dao.get_signal_events("2023-01-01", "2023-01-02")
    
    assert result == []
    assert "Failed to fetch signal events" in caplog.text

if __name__ == "__main__":
    pytest.main(["-v"])
