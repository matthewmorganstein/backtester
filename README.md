# pointland_signal_backtester

Core Components
Lineland Break
Purpose: Core backtesting engine.

Description: Backtests the last 15 trades on 30m candles with a 1 unit position. Triggers on price breakouts + r_1/r_2 signals.

Trading Context: Spots reversal potential.

Pointland Signal
Purpose: Entry signal generator.

Description:

Buy: Close < previous low AND (r_1 OR r_2 > Square Threshold).

Sell: Close > previous high AND (r_1 OR r_2 > Square Threshold).

Technical Analysis: RISE r_1/r_2 (350-600).

Square Threshold
Purpose: Filters signal strength.

Description: Default 350; users can optimize it.

Trading Context: Cuts noise, focuses on momentum.

Sphere Exit
Purpose: Trade exit rules.

Description: 1% target/stop or opposite breakout.

Trading Context: Quick, practical risk-reward.

Polygon Profit
Purpose: Performance metrics.

Description: Win rate, profit factor, total profit over 15 trades.

Trading Context: Proof the strategy works—or doesn’t.

Tesseract Visual
Purpose: Visual feedback.

Description: Plotly chart of price + trades (green wins, red losses).

Trading Context: See the edge at a glance.

Processors
Flatfire Strategy Processor
Purpose: Runs the core backtest.

Description: Ties Lineland Break, Pointland Signal, Square Threshold, Sphere Exit, and Polygon Profit into one flow.

Endpoint: POST /api/v1/flatfire-backtest

Parameters: symbol (str): "BTC". start_time (datetime): Start of period. end_time (datetime): End of period. threshold (float, optional): Default 350.0. distance_threshold (float, optional): Default 0.01 (1%).

Example: bash curl -X POST "http://your_server_ip:8000/api/v1/flatfire-backtest"
-H "X-API-Key: flarland-free-test"
-d '{"symbol": "BTC", "start_time": "2025-03-28T00:00:00", "end_time": "2025-03-29T00:00:00"}'

Output: HTML Plotly chart or JSON: json { "symbol": "BTC", "trades": [{"entry_time": "2025-03-28T00:30:00", "entry_price": 60250.0, "profit": 550.0, ...}], "performance": {"total_trades": 15, "win_rate": 0.6, "profit_factor": 1.8, "total_profit": 4500.0} }
