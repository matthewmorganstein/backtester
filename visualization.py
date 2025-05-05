"""Visualization utilities for backtest performance."""
from __future__ import annotations
import logging

import pandas as pd
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

PORTFOLIO_COLOR = "blue"
BUY_COLOR = "green"
SELL_COLOR = "red"
MARKER_SIZE = 10
PLOT_TEMPLATE = "plotly_white"

def create_performance_chart(plot_data: list[dict], trade_data: list[dict] | None = None) -> go.Figure:
    """Create a Plotly chart from backtest performance data.

    Args:
        plot_data: List of dictionaries containing timestamp and portfolio_value (from get_plot_data).
        trade_data: Optional list of dictionaries containing timestamp, event_type, and price (from get_signal_events).

    Returns:
        A Plotly Figure object for rendering performance metrics.

    Raises:
        ValueError: If required fields are missing in plot_data or trade_data.
    """
    if not plot_data:
        logger.warning("Empty plot data provided")
        fig = go.Figure()
        fig.update_layout(
            title="No Data Available",
            xaxis_title="Time",
            yaxis_title="Value",
            template=PLOT_TEMPLATE,
        )
        return fig

    # Convert plot_data to DataFrame
    df = pd.DataFrame(plot_data)
    if "timestamp" not in df.columns or "portfolio_value" not in df.columns:
        logger.error("Missing required columns in plot_data: 'timestamp', 'portfolio_value'")
        raise ValueError("Invalid plot_data: missing required columns")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Create figure
    fig = go.Figure()

    # Portfolio value line
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["portfolio_value"],
            mode="lines",
            name="Portfolio Value",
            line=dict(color=PORTFOLIO_COLOR),
        )
    )

    # Process trade_data if provided
    if trade_data:
        trade_df = pd.DataFrame(trade_data)
        if not trade_df.empty and {"timestamp", "event_type", "price"}.issubset(trade_df.columns):
            trade_df["timestamp"] = pd.to_datetime(trade_df["timestamp"], utc=True)
            buy_trades = trade_df[trade_df["event_type"] == "buy_signal"]
            sell_trades = trade_df[trade_df["event_type"] == "sell_signal"]

            # Buy trades
            if not buy_trades.empty:
                fig.add_trace(
                    go.Scatter(
                        x=buy_trades["timestamp"],
                        y=buy_trades["price"],
                        mode="markers",
                        name="Buy Trade",
                        marker=dict(symbol="triangle-up", size=MARKER_SIZE, color=BUY_COLOR),
                    )
                )

            # Sell trades
            if not sell_trades.empty:
                fig.add_trace(
                    go.Scatter(
                        x=sell_trades["timestamp"],
                        y=sell_trades["price"],
                        mode="markers",
                        name="Sell Trade",
                        marker=dict(symbol="triangle-down", size=MARKER_SIZE, color=SELL_COLOR),
                    )
                )
        else:
            logger.warning("Invalid or empty trade_data provided")

    # Layout
    fig.update_layout(
        title="Backtest Performance",
        xaxis_title="Time",
        yaxis_title="Value",
        showlegend=True,
        template=PLOT_TEMPLATE,
    )

    return fig
