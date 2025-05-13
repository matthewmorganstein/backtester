import pytest
import pandas as pd
from datetime import datetime, timedelta
from backtester import SignalBacktester, Trade, TradeManager
from settings import Settings

@pytest.fixture
def settings():
    return Settings(
        API_KEY="test_key",
        DB_HOST="localhost",
        DB_PORT=5432,
        DB_USER="test",
        DB_PASSWORD="test",
        DB_NAME="test",
    )

@pytest.mark.asyncio
async def test_signal_backtester_init(settings):
    backtester = SignalBacktester(settings=settings)
    assert backtester.square_threshold == 350.0
    assert backtester.distance_threshold == 0.01
    assert backtester.bar_duration_minutes == 1.0

def test_pointland_signal():
    df = pd.DataFrame({
        "timestamp": [datetime.now() - timedelta(minutes=i) for i in range(3)],
        "close": [100, 99, 101],
        "high": [101, 100, 102],
        "low": [99, 98, 100],
        "r_1": [400, 300, 200],
        "r_2": [300, 400, 200],
    })
    backtester = SignalBacktester(square_threshold=350.0)
    signals = backtester.pointland_signal(df)
    assert signals.iloc[0] == 1  # Buy signal: close < prev_low and r_1 > 350
    assert signals.iloc[1] == 0
    assert signals.iloc[2] == 0

def test_trade_check_exit():
    trade = Trade(
        entry_idx=0,
        entry_time=datetime.now(),
        entry_price=100.0,
        direction="buy",
        target=101.0,
        stop=99.0,
        square_threshold=350.0,
    )
    future_data = pd.DataFrame({
        "timestamp": [datetime.now() + timedelta(minutes=i) for i in range(3)],
        "close": [100.5, 101.5, 98.5],
    })
    opposing_signal = pd.Series([False, False, False], index=future_data.index)
    result = trade.check_exit(future_data, opposing_signal, target_bars=5, bar_duration_minutes=1.0)
    assert result["success"] == True  # Hits target at 101.5
    assert result["time_above_stop"] == 2.0  # 2 bars above stop

@pytest.mark.asyncio
async def test_run_backtest(mocker, settings):
    mocker.patch("dao.PostgresDAO.get_data", return_value=pd.DataFrame({
        "timestamp": [datetime.now() - timedelta(minutes=i) for i in range(10)],
        "close": [100 + i for i in range(10)],
        "high": [101 + i for i in range(10)],
        "low": [99 + i for i in range(10)],
        "r_1": [400] * 10,
        "r_2": [300] * 10,
    }))
    backtester = SignalBacktester(settings=settings)
    trades = await backtester.run_backtest()
    assert len(trades) > 0
