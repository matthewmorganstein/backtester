## Backtester

*Backtester* is a blazing phenomenon of financial combustion, where the light of precise trading signals, the heat of strategic backtesting, and the flame of portfolio mastery converge in a radiant, modular API. Built on the "Pointland entry / Sphere exit" strategy, it uses `r_1` and `r_2` indicators to spark trades and spherical logic to exit with precision. *Backtester* illuminates market opportunities, fuels performance, and flares with stunning visualizations, redefining the gravity of financial innovation.

## Features
- **Signal Generation**: Pointland signals using `r_1`/`r_2` > 350 and price breakouts.
- **Backtesting**: Modular `Trade` class with optimized exit logic.
- **Visualization**: Plotly charts for portfolio value and trade events.
- **Scalability**: FastAPI endpoints and async PostgreSQL integration.

# File 1: backtester.py
This file is the heart of the backtesting engine, handling signal generation, trade execution, and portfolio management.

## Module 1: Constants

Purpose: Defines global constants for backtesting (e.g., initial portfolio, signal values, data columns).

Role: Ensures consistency across classes (e.g., TradeManager uses INITIAL_CASH, SignalBacktester uses COLUMNS_REQUIRED).

Integration: Used by all modules; COLUMNS_REQUIRED ensures data validation in _validate_and_clean_data.

## Module 2: Indicator Interface and Implementations (Upcoming update)

Purpose: Defines a modular interface for indicators.

Role: Generates buy/sell/hold signals as pd.Series (1, -1, 0). Pointland Indicator is the primary indicator.

## Module 4: TradeManager

Purpose: Manages portfolio state (cash, position, trades) during backtesting.

Role: Executes trades based Pointland signal in solo mode (combined mode will be available in the full release), updates portfolio value, and records trade details.

Integration: Used by SignalBacktester.run_backtest to process trades and save results via BacktestDAO.

## Module 6: SignalBacktester

Purpose: Orchestrates backtesting, from data loading to signal generation, trade execution, and result storage.

Role: Combines indicator signals, manages trades via TradeManager, and saves results (error rate, diversity, portfolio value) to Postgres via BacktestDAO.

# File 2: main.py

This file runs the FastAPI app, serving backtests and visualizations. 

## Module 1: Setup and Constants

Purpose: Configures the app (logging, settings, constants).

Role: Initializes FastAPI and middleware (CORS, API key verification).

Integration: Provides settings to SignalBacktester and BacktestDAO.

## Module 2: Pydantic Models

Purpose: Defines data models for API requests/responses.

Role: Validates inputs (BacktestRequest) and structures outputs (TradeResult, SignalEvent).

Integration: Used in /backtest and /events endpoints to ensure valid data.

## Module 3: BacktestAPIService

Purpose: Manages backtest logic and data access for the API.

Role: Initializes SignalBacktester, runs backtests, and fetches visualization data.

## Module 4: FastAPI App and Endpoints

Purpose: Defines API endpoints for backtesting and visualization.

Role: Serves /backtest for running backtests, /events for signal data, and /performance for charts.

Integration: Calls BacktestAPIService, which uses SignalBacktester.
