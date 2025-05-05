"""Unit tests for SignalBacktester and TradeManager."""
from __future__ import annotations
from datetime import datetime, timedelta, timezone
import logging
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest
from asyncpg.exceptions import PostgresError

from backtester import SignalBacktester, TradeManager, INITIAL_CASH, TARGET_BARS, TIME_ABOVE_STOP_THRESHOLD

@pytest.fixture
def trade_manager():
    """Create a TradeManager instance."""
    return TradeManager()

@pytest.fixture
def backtester():
    """Create a SignalBacktester instance with mocked DAOs."""
    backtester = SignalBacktester(start_date="2023-01-01", end_date="2023-01-02")
    backtester.dao = MagicMock()
    backtester.backtest_dao = MagicMock()
    backtester.dao.get_data = AsyncMock()
    backtester.backtest_dao.save_signal_event = AsyncMock()
    backtester.backtest_dao.save_portfolio_update = AsyncMock()
    backtester.backtest_dao.save_backtest_result = AsyncMock()
    return backtester

@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    dates = pd.date_range("2023-01-01", "2023-01-02", freq="H")
    return pd.DataFrame({
        "timestamp": dates,
        "close": np.linspace(100, 110, len(dates)),
        "high": np.linspace(101, 111, len(dates)),
        "low": np.linspace(99, 109, len(dates)),
        "r_1": np.full(len(dates), 400),
        "r_2": np.full(len(dates), 400),
    })

def test_trade_manager_init(trade_manager):
    """Test TradeManager initialization."""
    assert trade_manager.cash == INITIAL_CASH
    assert trade_manager.portfolio_value == INITIAL_CASH
    assert trade_manager.position == 0
    assert trade_manager.trades == []
    assert trade_manager.entry_price is None

def test_trade_manager_reset(trade_manager):
    """Test TradeManager reset."""
    trade_manager.cash = 50000
    trade_manager.trades = [{"action": "buy"}]
    trade_manager.reset()
    assert trade_manager.cash == INITIAL_CASH
    assert trade_manager.trades == []

def test_trade_manager_enter_position(trade_manager):
    """Test entering a position."""
    trade = trade_manager.enter_position(signal=1, price=100, timestamp=datetime(2023, 1, 1))
    assert trade["action"] == "buy"
    assert trade["price"] == 100
    assert trade_manager.position == 1
    assert trade_manager.cash == INITIAL_CASH - 100

def test_trade_manager_exit_position(trade_manager):
    """Test exiting a position."""
    trade_manager.enter_position(signal=1, price=100, timestamp=datetime(2023, 1, 1))
    exit_result = {
        "exit_price": 110,
        "profit": 10,
        "exit_time": datetime(2023, 1, 1),
        "success": True,
        "failure": False,
        "time_above_stop": 40,
    }
    trade = trade_manager.exit_position(exit_result)
    assert trade["action"] == "sell"
    assert trade["profit"] == 10
    assert trade_manager.position == 0
    assert trade_manager.cash == INITIAL_CASH - 100 + 110

def test_trade_manager_update_portfolio(trade_manager):
    """Test updating portfolio value."""
    trade_manager.enter_position(signal=1, price=100, timestamp=datetime(2023, 1, 1))
    details = trade_manager.update_portfolio(price=105)
    assert details["portfolio_value"] == INITIAL_CASH - 100 + 105
    assert details["cash"] == INITIAL_CASH - 100

@pytest.mark.asyncio
async def test_backtester_fetch_data_empty(backtester):
    """Test fetching empty data."""
    backtester.dao.get_data.return_value = pd.DataFrame()
    with pytest.raises(ValueError, match="Empty DataFrame from DAO"):
        await backtester._fetch_data()

@pytest.mark.asyncio
async def test_backtester_fetch_data_postgres_error(backtester):
    """Test fetching data with PostgresError."""
    backtester.dao.get_data.side_effect = PostgresError("Connection failed")
    with pytest.raises(ValueError, match="RISE database error"):
        await backtester._fetch_data()

def test_backtester_validate_data_missing_columns(backtester):
    """Test data validation with missing columns."""
    df = pd.DataFrame({"timestamp": [datetime(2023, 1, 1)], "close": [100]})
    with pytest.raises(ValueError, match="DataFrame missing columns"):
        backtester._validate_and_clean_data(df)

def test_pointland_signal(backtester, sample_df):
    """Test Pointland signal generation."""
    signals = backtester.pointland_signal(sample_df)
    assert isinstance(signals, pd.Series)
    assert signals.dtype == int
    assert (signals.isin([0, 1, -1])).all()

def test_sphere_exit(backtester, sample_df):
    """Test sphere_exit logic."""
    result = backtester.sphere_exit(sample_df, entry_idx=0, position=1)
    assert "success" in result
    assert "profit" in result
    assert "exit_price" in result
    assert isinstance(result["time_above_stop"], float)

def test_polygon_profit_empty(backtester):
    """Test polygon_profit with no trades."""
    result = backtester.polygon_profit()
    assert result["win_rate"] == 0.0
    assert result["num_trades"] == 0
    assert result["success_rate_time"] == 0.0

@pytest.mark.asyncio
async def test_process_signals(backtester, sample_df, monkeypatch):
    """Test signal processing."""
    monkeypatch.setattr(backtester, "_validate_and_clean_data", lambda x: x)
    signals = backtester.pointland_signal(sample_df)
    trades, signals_triggered = await backtester.process_signals(sample_df, signals, "test_id")
    assert isinstance(trades, list)
    assert isinstance(signals_triggered, int)
    assert backtester.backtest_dao.save_portfolio_update.called

@pytest.mark.asyncio
async def test_save_backtest_results(backtester, sample_df):
    """Test saving backtest results."""
    result = {
        "trades": [],
        "final_portfolio_value": INITIAL_CASH,
        "returns": 0.0,
        "metrics": {},
        "signals_triggered": 0,
    }
    await backtester.save_backtest_results("test_id", result)
    backtester.backtest_dao.save_backtest_result.assert_awaited_once()

@pytest.mark.asyncio
async def test_backtest_success(backtester, sample_df, monkeypatch):
    """Test full backtest execution."""
    backtester.dao.get_data.return_value = sample_df
    monkeypatch.setattr(backtester, "_validate_and_clean_data", lambda x: x)
    result = await backtester.backtest(backtest_id="test_id")
    assert "trades" in result
    assert "final_portfolio_value" in result
    assert "returns" in result
    assert "metrics" in result
    assert backtester.backtest_dao.save_backtest_result.called

@pytest.mark.asyncio
async def test_backtest_postgres_error(backtester, sample_df, monkeypatch):
    """Test backtest with a PostgresError."""
    backtester.dao.get_data.return_value = sample_df
    monkeypatch.setattr(backtester, "_validate_and_clean_data", lambda x: x)
    backtester.backtest_dao.save_signal_event.side_effect = PostgresError("Insert failed")
    result = await backtester.backtest(backtest_id="test_id")
    assert result["final_portfolio_value"] == INITIAL_CASH

if __name__ == "__main__":
    pytest.main(["-v"])
