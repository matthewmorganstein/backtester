"""Unit tests for the Backtest API."""
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from api import BacktestAPI, BacktestRequest, app
from utils import InvalidToken

@pytest.fixture
def client():
    """Create a FastAPI test client."""
    return TestClient(app)

@pytest.fixture
def backtest_api(monkeypatch):
    """Create a BacktestAPI instance with mocked DAOs."""
    api = BacktestAPI()
    api.dao = MagicMock()
    api.backtest_dao = MagicMock()
    api.dao.setup = AsyncMock()
    api.backtest_dao.setup = AsyncMock()
    api.dao.close_pool = AsyncMock()
    api.backtest_dao.close_pool = AsyncMock()
    api.backtest_dao.get_plot_data = AsyncMock(return_value=[{"timestamp": "2023-01-01", "value": 100}])
    api.backtest_dao.get_signal_events = AsyncMock(return_value=[{"event": "buy", "timestamp": "2023-01-01"}])
    monkeypatch.setattr("api.backtest_api", api)
    return api

@pytest.mark.asyncio
async def test_lifespan_setup_and_cleanup(backtest_api):
    """Test the FastAPI lifespan for setup and cleanup."""
    async with app.lifespan_context(app):
        backtest_api.dao.setup.assert_awaited_once()
        backtest_api.backtest_dao.setup.assert_awaited_once()
    backtest_api.dao.close_pool.assert_awaited_once()
    backtest_api.backtest_dao.close_pool.assert_awaited_once()

@pytest.mark.asyncio
async def test_run_backtest_success(backtest_api, monkeypatch):
    """Test successful backtest execution."""
    mock_backtester = MagicMock()
    mock_backtester.backtest = AsyncMock(return_value={"final_portfolio_value": 1000, "returns": 0.1})
    monkeypatch.setattr("api.SignalBacktester", mock_backtester)
    
    request = BacktestRequest(start_date="2023-01-01", end_date="2023-01-02", square_threshold=350, distance_threshold=0.01)
    result = await backtest_api.run_backtest(request)
    
    assert result == {"final_portfolio_value": 1000, "returns": 0.1}
    mock_backtester.backtest.assert_awaited_once()

@pytest.mark.asyncio
async def test_run_backtest_postgres_error(backtest_api, monkeypatch):
    """Test backtest with a PostgresError."""
    mock_backtester = MagicMock()
    mock_backtester.backtest = AsyncMock(side_effect=asyncpg.exceptions.PostgresError("Database error"))
    monkeypatch.setattr("api.SignalBacktester", mock_backtester)
    
    request = BacktestRequest(start_date="2023-01-01", end_date="2023-01-02")
    with pytest.raises(api.HTTPException) as exc:
        await backtest_api.run_backtest(request)
    
    assert exc.value.status_code == 500
    assert exc.value.detail == "Database error during backtest"

@pytest.mark.asyncio
async def test_run_backtest_invalid_token(backtest_api, monkeypatch):
    """Test backtest with an InvalidToken error."""
    mock_backtester = MagicMock()
    mock_backtester.backtest = AsyncMock(side_effect=InvalidToken("Invalid token"))
    monkeypatch.setattr("api.SignalBacktester", mock_backtester)
    
    request = BacktestRequest(start_date="2023-01-01", end_date="2023-01-2")
    with pytest.raises(api.HTTPException) as exc:
        await backtest_api.run_backtest(request)
    
    assert exc.value.status_code == 500
    assert exc.value.detail == "Failed to decrypt indicators"

def test_backtest_request_validation():
    """Test BacktestRequest model validation."""
    # Valid request
    request = BacktestRequest(start_date="2023-01-01", end_date="2023-01-02")
    assert request.start_date == "2023-01-01"
    
    # Invalid date format
    with pytest.raises(ValidationError):
        BacktestRequest(start_date="2023-13-01", end_date="2023-01-02")
    
    # End date before start date
    with pytest.raises(ValidationError):
        BacktestRequest(start_date="2023-01-02", end_date="2023-01-01")
    
    # Date range exceeds 30 days
    with pytest.raises(ValidationError):
        BacktestRequest(start_date="2023-01-01", end_date="2023-03-01")

def test_root_endpoint(client):
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Backtest API"}

def test_backtest_endpoint(client, backtest_api, monkeypatch):
    """Test the /backtest endpoint."""
    mock_verify_api_key = MagicMock(return_value=True)
    monkeypatch.setattr("api.verify_api_key", mock_verify_api_key)
    
    response = client.post(
        "/backtest",
        json={"start_date": "2023-01-01", "end_date": "2023-01-02", "square_threshold": 350, "distance_threshold": 0.01}
    )
    assert response.status_code == 200
    assert "final_portfolio_value" in response.json()

def test_performance_endpoint(client, backtest_api, monkeypatch):
    """Test the /performance endpoint."""
    mock_verify_api_key = MagicMock(return_value=True)
    monkeypatch.setattr("api.verify_api_key", mock_verify_api_key)
    mock_create_chart = MagicMock()
    mock_create_chart.to_json = MagicMock(return_value="{}")
    monkeypatch.setattr("api.create_performance_chart", mock_create_chart)
    
    response = client.get("/performance?start_date=2023-01-01&end_date=2023-01-02")
    assert response.status_code == 200
    assert "performance.html" in response.template.name

if __name__ == "__main__":
    pytest.main(["-v"])
