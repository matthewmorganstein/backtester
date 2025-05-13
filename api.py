"""FastAPI-based API for backtesting and performance visualization."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta # Use date for date-only comparisons
from typing import Any, AsyncGenerator, Dict, List, Optional # Use List from typing

from asyncpg.exceptions import PostgresError
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, field_validator, model_validator, ValidationInfo

from backtest_dao import BacktestDAO
from backtester import SignalBacktester # Assuming ExitResult is implicitly handled or part of list[dict]
from dao import PostgresDAO
from settings import Settings # Your enhanced Settings class
from utils import logger, verify_api_key # Assuming logger is configured
from visualization import create_performance_chart

# --- Constants ---
DATE_FORMAT = "%Y-%m-%d"
ERROR_MSG_INVALID_DATE_FORMAT = f"Date must be a string in {DATE_FORMAT} format."
ERROR_MSG_END_DATE_BEFORE_START = "end_date must be after start_date."
ERROR_MSG_DATE_RANGE_EXCEEDED = "Date range exceeds the maximum allowed limit of {max_days} days."

# --- Application Setup ---

# Instantiate settings first
# Pydantic v2 automatically validates on instantiation if fields are non-optional
# and no default is provided.
try:
    settings = Settings()
except Exception as e: # Catch Pydantic's ValidationError or other init errors
    # Log critical error and exit if settings can't load, as app can't run.
    # Using a basic print here as logger might not be fully set up if settings fail.
    print(f"CRITICAL: Failed to load application settings: {e}")
    raise SystemExit(f"CRITICAL: Failed to load application settings: {e}") from e


# Configure logger based on settings (example)
# This should ideally be in your utils.py or a dedicated logging_config.py
# For simplicity, showing a basic configuration here.
logging.basicConfig(level=settings.LOG_LEVEL.upper())


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage FastAPI application lifespan for DAO initialization and cleanup."""
    logger.info("Initializing Backtest API DAOs...")
    try:
        await backtest_api_service.initialize_daos()
        logger.info("Backtest API DAOs initialized successfully.")
    except PostgresError as db_err:
        logger.exception(f"Failed to initialize DAOs due to database error: {db_err}")
        raise RuntimeError(f"DAO Initialization failed: {db_err}") from db_err
    except Exception as e:
        logger.exception(f"An unexpected error occurred during DAO initialization: {e}")
        raise RuntimeError(f"DAO Initialization failed with an unexpected error: {e}") from e
    
    yield
    
    logger.info("Shutting down Backtest API...")
    try:
        await backtest_api_service.close_dao_pools()
        logger.info("Backtest API DAO pools closed successfully.")
    except PostgresError as db_err:
        logger.exception(f"Failed to close DAO pools due to database error: {db_err}")
    except Exception as e:
        logger.exception(f"An unexpected error occurred while closing DAO pools: {e}")


# --- Pydantic Models ---

class BacktestRequest(BaseModel):
    """Model for backtest request parameters."""
    start_date: str
    end_date: str
    square_threshold: float = 350.0
    distance_threshold: float = 0.01

    @field_validator("start_date", "end_date", mode="before")
    @classmethod
    def validate_and_parse_date_format(cls, v: Any) -> str:
        """Validate date format (YYYY-MM-DD) and return as string."""
        if not isinstance(v, str):
            raise ValueError(ERROR_MSG_INVALID_DATE_FORMAT)
        try:
            # Validate format by parsing, but return original string for further model validation
            datetime.strptime(v, DATE_FORMAT)
            return v
        except ValueError as e:
            logger.debug(f"Invalid date format provided: {v}")
            raise ValueError(ERROR_MSG_INVALID_DATE_FORMAT) from e

    @model_validator(mode='after')
    def ensure_date_order_and_range(self) -> 'BacktestRequest':
        """Ensure end_date is after start_date and within the allowed range."""
        try:
            # Dates are already validated for format by field_validator
            start_dt = datetime.strptime(self.start_date, DATE_FORMAT).date()
            end_dt = datetime.strptime(self.end_date, DATE_FORMAT).date()
        except ValueError:
            # Should not happen if field validators ran correctly, but as a safeguard
            raise ValueError("Internal error: Invalid date format encountered during model validation.")

        if end_dt <= start_dt:
            raise ValueError(ERROR_MSG_END_DATE_BEFORE_START)

        if (end_dt - start_dt).days > settings.MAX_DATE_RANGE_DAYS:
            raise ValueError(
                ERROR_MSG_DATE_RANGE_EXCEEDED.format(max_days=settings.MAX_DATE_RANGE_DAYS)
            )
        return self

# --- Service Class ---

class BacktestAPIService:
    """Encapsulates business logic for the Backtest API."""
    def __init__(self, app_settings: Settings):
        self.settings = app_settings
        # DAOs are initialized as None and set up in initialize_daos
        self.dao: Optional = None
        self.backtest_dao: Optional = None

    async def initialize_daos(self) -> None:
        """Initialize DAO connections."""
        self.dao = PostgresDAO(self.settings)
        self.backtest_dao = BacktestDAO(self.settings)
        await self.dao.setup() # Assuming setup creates connection pools
        await self.backtest_dao.setup()

    async def close_dao_pools(self) -> None:
        """Close DAO connection pools."""
        if self.dao:
            await self.dao.close_pool()
        if self.backtest_dao:
            await self.backtest_dao.close_pool()

    async def run_backtest_logic(self, request: BacktestRequest) -> list[dict]: # Assuming ExitResult can be dict
        """Run a backtest with the given parameters."""
        logger.info(f"Received backtest request: {request.model_dump_json(exclude={'API_KEY'})}") # Use model_dump_json
        try:
            backtester = SignalBacktester(
                start_date=request.start_date,
                end_date=request.end_date,
                square_threshold=request.square_threshold,
                distance_threshold=request.distance_threshold,
                # Pass DAOs or settings to SignalBacktester if it needs them
                # e.g., dao=self.dao, backtest_dao=self.backtest_dao
            )
            trade_results: list[dict] = await backtester.run_backtest() # Ensure return type matches
            logger.info(f"Backtest completed. Number of trades: {len(trade_results)}")
            return trade_results
        except PostgresError as e:
            logger.exception("Database error during backtest execution.")
            raise HTTPException(status_code=503, detail="Service temporarily unavailable due to a database error.") from e
        except ValueError as e: # Catch specific ValueErrors from backtester logic
            logger.warning(f"Validation or data error during backtest: {e}")
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            logger.exception("Unexpected error during backtest execution.")
            raise HTTPException(status_code=500, detail="An unexpected error occurred during the backtest.") from e

    async def _validate_date_query_params(self, start_date_str: str, end_date_str: str) -> tuple[date, date]:
        """Helper to validate and parse date query parameters."""
        try:
            start_dt = datetime.strptime(start_date_str, DATE_FORMAT).date()
            end_dt = datetime.strptime(end_date_str, DATE_FORMAT).date()
        except ValueError:
            raise HTTPException(status_code=400, detail=ERROR_MSG_INVALID_DATE_FORMAT)

        if end_dt <= start_dt:
            raise HTTPException(status_code=400, detail=ERROR_MSG_END_DATE_BEFORE_START)
        
        # You might want to apply MAX_DATE_RANGE_DAYS here too for consistency
        # if (end_dt - start_dt).days > self.settings.MAX_DATE_RANGE_DAYS:
        #     raise HTTPException(
        #         status_code=400,
        #         detail=ERROR_MSG_DATE_RANGE_EXCEEDED.format(max_days=self.settings.MAX_DATE_RANGE_DAYS)
        #     )
        return start_dt, end_dt

    async def get_portfolio_plot_data(self, start_date_str: str, end_date_str: str) -> list[dict]:
        """Fetch plot data for portfolio visualization."""
        await self._validate_date_query_params(start_date_str, end_date_str) # Use validated dates
        logger.info(f"Fetching portfolio plot data for range: {start_date_str} to {end_date_str}")
        if not self.backtest_dao:
            raise HTTPException(status_code=503, detail="Service not fully initialized.")
        return await self.backtest_dao.get_plot_data(start_date_str, end_date_str)

    async def get_trade_signal_events(self, start_date_str: str, end_date_str: str) -> list[dict]:
        """Fetch signal events for visualization."""
        await self._validate_date_query_params(start_date_str, end_date_str)
        logger.info(f"Fetching signal events for range: {start_date_str} to {end_date_str}")
        if not self.backtest_dao:
            raise HTTPException(status_code=503, detail="Service not fully initialized.")
        return await self.backtest_dao.get_signal_events(start_date_str, end_date_str)

    async def generate_performance_chart_figure(self, start_date_str: str, end_date_str: str) -> go.Figure:
        """Generates a Plotly figure for performance."""
        # Validation is handled by the calling methods or _validate_date_query_params
        plot_data = await self.get_portfolio_plot_data(start_date_str, end_date_str)
        trade_data = await self.get_trade_signal_events(start_date_str, end_date_str)
        
        # create_performance_chart might raise ValueError if data is bad, catch it or let it propagate
        # For now, assuming create_performance_chart handles empty data gracefully as per its original code.
        fig = create_performance_chart(plot_data, trade_data)
        return fig

# Instantiate the service class
backtest_api_service = BacktestAPIService(app_settings=settings)

# FastAPI app instance
app = FastAPI(
    title="Financial Backtesting API",
    description="API for running trading strategy backtests and visualizing performance.",
    version="1.0.0",
    lifespan=lifespan, # Use the refined lifespan manager
    #openapi_url="/api/v1/openapi.json" # Example: Custom OpenAPI URL
    #docs_url="/api/docs" # Example: Custom Docs URL
)

# CORS Middleware Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ALLOWED_ORIGINS,
    allow_credentials=True, # Set to False if not using cookies/auth headers from frontend
    allow_methods=settings.CORS_ALLOWED_METHODS, # Use specific methods from settings
    allow_headers=settings.CORS_ALLOWED_HEADERS, # Use specific headers from settings
    # expose_headers=["X-Custom-Header"], # If frontend needs to access custom response headers
    # max_age=600 # Default is 600 seconds
)

templates = Jinja2Templates(directory="templates")

# --- API Endpoints ---

@app.get("/", summary="Root Endpoint", description="Welcome message for the API.", tags=["General"])
async def root() -> Dict[str, str]:
    """Root endpoint for the API."""
    return {"message": "Welcome to the Backtest API"}

@app.post("/backtest",
          summary="Run a Trading Strategy Backtest",
          description="Submits parameters for a backtest and returns the trade results.",
          response_model=List], # Be more specific if trade result structure is known
          dependencies=,
          tags=)
async def run_backtest_endpoint(request_data: BacktestRequest) -> List]:
    """Endpoint to run a backtest."""
    return await backtest_api_service.run_backtest_logic(request_data)

@app.get("/events",
         summary="Get Signal Events",
         description="Fetches signal event data for a specified date range.",
         response_model=List], # Specify if structure known
         dependencies=,
         tags=)
async def get_events_endpoint(start_date: str, end_date: str) -> List]:
    """Fetch signal events for a date range."""
    return await backtest_api_service.get_trade_signal_events(start_date, end_date)

@app.get("/performance_chart_json",
         summary="Get Performance Chart Data (JSON)",
         description="Fetches data for a performance chart and returns it as Plotly JSON.",
         response_class=JSONResponse, # Explicitly use JSONResponse for direct JSON
         dependencies=,
         tags=["Visualization"])
async def get_performance_chart_json_endpoint(start_date: str, end_date: str) -> JSONResponse:
    """Fetch performance chart data as JSON."""
    try:
        fig = await backtest_api_service.generate_performance_chart_figure(start_date, end_date)
        return JSONResponse(content=fig.to_json()) # Use Plotly's to_json method
    except HTTPException: # Re-raise HTTPExceptions from service layer
        raise
    except Exception as e:
        logger.exception("Error generating performance chart JSON.")
        raise HTTPException(status_code=500, detail="Failed to generate performance chart data.")

@app.get("/performance",
         summary="View Performance Chart Page",
         description="Renders an HTML page displaying the performance chart.",
         response_class=HTMLResponse,
         dependencies=, # Secure the HTML page too
         tags=["Visualization"])
async def performance_page_endpoint(
    http_request: Request, # Renamed for clarity, standard FastAPI practice
    start_date: str,
    end_date: str
) -> HTMLResponse:
    """Render performance chart page."""
    logger.info(f"Performance page request: start_date={start_date}, end_date={end_date}")
    try:
        fig = await backtest_api_service.generate_performance_chart_figure(start_date, end_date)
        plot_json_data = fig.to_json()

        return templates.TemplateResponse(
            "performance.html", # Ensure this template exists
            {
                "request": http_request,
                "plot_json": plot_json_data,
                "start_date": start_date,
                "end_date": end_date,
                "page_title": "Backtest Performance" # Example: adding more context
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error rendering performance page.")
        # For an HTML page, you might want to render a generic error template
        # instead of just raising HTTPException, or provide a user-friendly error message.
        # For now, re-raising for consistency, but consider a custom error page.
        return templates.TemplateResponse(
            "error.html", # Assume you have an error.html template
            {
                "request": http_request,
                "error_message": "Failed to render performance page. Please try again later."
            },
            status_code=500
        )
        # Or, if you prefer to stick to HTTPException for API-like behavior:
        # raise HTTPException(status_code=500, detail=f"Failed to render performance page: {str(e)}")

# --- Main Execution Guard ---
if __name__ == "__main__":
    import uvicorn
    # settings.PORT is already an int due to Pydantic type coercion
    # settings.HOST would be a str
    uvicorn.run(
        "main:app", # Important: points to this file (main.py) and the app instance
        host=getattr(settings, 'HOST', "127.0.0.1"), # Use getattr for safety if HOST might not exist
        port=settings.PORT, # Already an int
        reload=True, # Useful for development, disable in production
        log_level=settings.LOG_LEVEL.lower() # Sync uvicorn log level with app
    )
