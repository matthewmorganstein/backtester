"""Visualization utilities for backtest performance."""
from __future__ import annotations

import logging
import pandas as pd
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

# Constants
COL_TIMESTAMP = "timestamp"
COL_PORTFOLIO_VALUE = "portfolio_value"
COL_EVENT_TYPE = "event_type"
COL_PRICE = "price"
EVENT_TYPE_BUY = "entry"
EVENT_TYPE_SELL = "exit"
PORTFOLIO_COLOR = "blue"
BUY_COLOR = "green"
SELL_COLOR = "red"
MARKER_SIZE = 10
MARKER_SYMBOL_BUY = "triangle-up"
MARKER_SYMBOL_SELL = "triangle-down"
PLOT_TEMPLATE = "plotly_white"

def _add_trade_markers_to_figure(fig: go.Figure, trade_data: list[dict] | pd.DataFrame | None) -> None:
    """Add trade markers (buy/sell signals) to the Plotly figure."""
    if not trade_data:
        logger.info("No trade data provided; skipping trade markers.")
        return

    trade_df = trade_data if isinstance(trade_data, pd.DataFrame) else pd.DataFrame(trade_data)
    if trade_df.empty:
        logger.warning("Trade data resulted in an empty DataFrame.")
        return

    required_trade_cols = {COL_TIMESTAMP, COL_EVENT_TYPE, COL_PRICE}
    if not required_trade_cols.issubset(trade_df.columns):
        missing_cols = required_trade_cols - set(trade_df.columns)
        logger.warning(f"Trade data missing required columns: {missing_cols}.")
        return

    trade_df[COL_TIMESTAMP] = pd.to_datetime(trade_df[COL_TIMESTAMP], utc=True)
    buy_trades = trade_df[trade_df[COL_EVENT_TYPE] == EVENT_TYPE_BUY]
    sell_trades = trade_df[trade_df[COL_EVENT_TYPE] == EVENT_TYPE_SELL]

    if not buy_trades.empty:
        fig.add_trace(
            go.Scatter(
                x=buy_trades[COL_TIMESTAMP],
                y=buy_trades[COL_PRICE],
                mode="markers",
                name="Buy Trade",
                marker={"symbol": MARKER_SYMBOL_BUY, "size": MARKER_SIZE, "color": BUY_COLOR},
            )
        )

    if not sell_trades.empty:
        fig.add_trace(
            go.Scatter(
                x=sell_trades[COL_TIMESTAMP],
                y=sell_trades[COL_PRICE],
                mode="markers",
                name="Sell Trade",
                marker={"symbol": MARKER_SYMBOL_SELL, "size": MARKER_SIZE, "color": SELL_COLOR},
            )
        )
    logger.info(f"Plotted {len(buy_trades)} buy and {len(sell_trades)} sell trades")

def create_performance_chart(plot_data: list[dict], trade_data: list[dict] | None = None) -> go.Figure:
    """Create a Plotly chart from backtest performance data."""
    if not plot_data:
        logger.warning("Empty plot data provided. Returning a blank chart.")
        fig = go.Figure()
        fig.update_layout(
            title="No Data Available",
            xaxis_title="Time",
            yaxis_title="Value",
            template=PLOT_TEMPLATE,
        )
        return fig

    portfolio_df = pd.DataFrame(plot_data)
    required_plot_cols = {COL_TIMESTAMP, COL_PORTFOLIO_VALUE}
    if not required_plot_cols.issubset(portfolio_df.columns):
        missing_cols = required_plot_cols - set(portfolio_df.columns)
        logger.error(f"Missing required columns in plot_data: {missing_cols}")
        raise ValueError(f"Missing required columns: {missing_cols}")

    portfolio_df[COL_TIMESTAMP] = pd.to_datetime(portfolio_df[COL_TIMESTAMP], utc=True)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=portfolio_df[COL_TIMESTAMP],
            y=portfolio_df[COL_PORTFOLIO_VALUE],
            mode="lines",
            name="Portfolio Value",
            line={"color": PORTFOLIO_COLOR},
        )
    )

    _add_trade_markers_to_figure(fig, trade_data)
    fig.update_layout(
        title="Backtest Performance",
        xaxis_title="Time",
        yaxis_title="Value",
        showlegend=True,
        template=PLOT_TEMPLATE,
    )
    return fig
