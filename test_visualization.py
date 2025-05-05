"""Unit tests for visualization utilities."""
from __future__ import annotations
from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
import pytest

from visualization import create_performance_chart

EXPECTED_TRACES = 3  # Portfolio line + buy + sell markers

@pytest.fixture
def sample_plot_data() -> list[dict]:
    """Create sample plot data."""
    return [
        {"timestamp": "2023-01-01 00:00:00+00:00", "portfolio_value": 100_000},
        {"timestamp": "2023-01-01 01:00:00+00:00", "portfolio_value": 100_500},
    ]

@pytest.fixture
def sample_trade_data() -> list[dict]:
    """Create sample trade data."""
    return [
        {
            "timestamp": "2023-01-01 00:30:00+00:00",
            "event_type": "buy_signal",
            "price": 100,
        },
        {
            "timestamp": "2023-01-01 01:00:00+00:00",
            "event_type": "sell_signal",
            "price": 105,
        },
    ]

def test_create_performance_chart_valid_data(
    sample_plot_data: list[dict], sample_trade_data: list[dict],
) -> None:
    """Test creating a chart with valid plot and trade data."""
    fig = create_performance_chart(sample_plot_data, sample_trade_data)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == EXPECTED_TRACES
    assert fig.layout.title.text == "Backtest Performance"
    assert fig.data[0].name == "Portfolio Value"
    assert fig.data[1].name == "Buy Trade"
    assert fig.data[2].name == "Sell Trade"

def test_create_performance_chart_empty_data() -> None:
    """Test creating a chart with empty data."""
    fig = create_performance_chart([])
    assert isinstance(fig, go
