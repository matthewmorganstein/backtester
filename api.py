"""FastAPI-based API for backtesting and performance visualization."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta
from typing import Any, AsyncGenerator, Dict, List, Optional

from asyncpg.exceptions import PostgresError
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, field_validator, model_validator, ValidationInfo

from backtest_dao import BacktestDAO
from backtester import SignalBacktester
from dao import PostgresDAO
from settings import Settings
from utils import configure_logging, verify_api_key
from visualization import create_performance_chart

# Constants
DATE_FORMAT = "%Y-%m-%d"
ERROR_MSG_INVALID_DATE_FORMAT = f"Date must be a string in {DATE_FORMAT} format."
ERROR_MSG_END_DATE_BEFORE_START = "end_date must be after start_date."
ERROR_MSG_DATE_RANGE_EXCEEDED = "Date range exceeds the maximum allowed limit of {max_days} days."

# Application Setup
try:
    settings = Settings()
except Exception as e:
    print(f"CRITICAL: Failed to load application settings: {e}")
    raise SystemExit(f"CRITICAL: Failed to load application settings: {e}") from e

configure_logging(settings)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Initializing Backtest API DAOs...")
    try:
        await backtest_api_service.initialize_daos()
        logger.info("Backtest API DAOs initialized successfully.")
    except PostgresError as db_err:
        logger.exception(f"Failed to initialize DAOs: {db_err}")
        raise RuntimeError(f"DAO Initialization failed: {db_err}") from db_err
    yield
    logger.info("Shutting down Backtest API...")
    try:
        await backtest_api_service.close_dao_pools()
        logger.info("Backtest API DAO pools closed successfully.")
    except PostgresError as db_err:
        logger.exception(f"Failed to close DAO pools: {db_err}")

# Pydantic Models
class TradeResult(BaseModel):
    trade_id: str
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime]
    exit_price: Optional[float]
    profit: float
    success: bool
    failure: bool
    time_above_stop: float

class SignalEvent(BaseModel):
    trade_id: str
    timestamp: datetime
    event_type: str
    price: float
    details: Dict[str, Any]

class BacktestRequest(BaseModel):
    start_date: str
    end_date: str
    square_threshold: float = 350.0
    distance_threshold: float = 0.01

    @field_validator("start_date", "end_date", mode="before")
    @classmethod
    def validate_and_parse_date_format(cls, v: Any) -> str:
        if not isinstance(v, str):
            raise ValueError(ERROR_MSG_INVALID_DATE_FORMAT)
        try:
            datetime.strptime(v, DATE_FORMAT)
            return v
        except ValueError as e:
            logger.debug(f"Invalid date format provided: {v}")
            raise ValueError(ERROR_MSG_INVALID_DATE_FORMAT) from e

    @field_validator("square_threshold")
    @classmethod
    def validate_square_threshold(cls, v: float) -> float:
        if not -200 <= v <= 650:
            raise ValueError("square_threshold must be between -200 and 650")
        return v

    @field_validator("distance_threshold")
    @classmethod
    def validate_distance_threshold(cls, v: float) -> float:
        if not 0 < v < 1:
            raise ValueError("distance_threshold must be between 0 and 1")
        return v

    @model_validator(mode='after')
    def ensure_date_order_and_range(self) -> 'BacktestRequest':
        try:
            start_dt = datetime.strptime(self.start_date, DATE_FORMAT).date()
            end_dt = datetime.strptime(self.end_date, DATE_FORMAT).date()
        except ValueError:
            raise ValueError("Internal error: Invalid date format")
        if end_dt <= start_dt:
            raise ValueError(ERROR_MSG_END_DATE_BEFORE_START)
        if (end_dt - start_dt).days > settings.MAX_DATE_RANGE_DAYS:
            raise ValueError(ERROR_MSG_DATE_RANGE_EXCEEDED.format(max_days=settings.MAX_DATE_RANGE_DAYS))
        return self

# Service Class
class BacktestAPIService:
    def __init__(self, app_settings: Settings):
        self.settings = app_settings
        self.dao: Optional[PostgresDAO] = None
        self.backtest_dao: Optional[BacktestDAO] = None

    async def initialize_daos(self) -> None:
        self.dao = PostgresDAO(self.settings)
        self.backtest_dao = BacktestDAO(self.settings)
        await self.dao.setup()
        await self.backtest_dao.setup()

    async def close_dao_pools(self) -> None:
        if self.dao:
            await self.dao.close_pool()
        if self.backtest_dao:
            await self.backtest_dao.close_pool()

    async def run_backtest_logic(self, request: BacktestRequest) -> List[Dict[str, Any]]:
        logger.info(f"Received backtest request: {request.model_dump_json()}")
        try:
            backtester = SignalBacktester(
                start_date=request.start_date,
                end_date=request.end_date,
                square_threshold=request.square_threshold,
                distance_threshold=request.distance_threshold,
                settings=self.settings,
            )
            trade_results = await backtester.run_backtest()
            logger.info(f"Backtest completed. Number of trades: {len(trade_results)}")
            return trade_results
        except PostgresError as e:
            logger.exception("Database error during backtest")
            raise HTTPException(status_code=503, detail="Service unavailable due to database error") from e
        except ValueError as e:
            logger.warning(f"Validation error during backtest: {e}")
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            logger.exception("Unexpected error during backtest")
            raise HTTPException(status_code=500, detail="Unexpected error during backtest") from e

    async def _validate_date_query_params(self, start_date_str: str, end_date_str: str) -> Tuple[date, date]:
        try:
            start_dt = datetime.strptime(start_date_str, DATE_FORMAT).date()
            end_dt = datetime.strptime(end_date_str, DATE_FORMAT).date()
        except ValueError:
            raise HTTPException(status_code=400, detail=ERROR_MSG_INVALID_DATE_FORMAT)
        if end_dt <= start_dt:
            raise HTTPException(status_code=400, detail=ERROR_MSG_END_DATE_BEFORE_START)
        if (end_dt - start_dt).days > self.settings.MAX_DATE_RANGE_DAYS:
            raise HTTPException(
                status_code=400,
                detail=ERROR_MSG_DATE_RANGE_EXCEEDED.format(max_days=self.settings.MAX_DATE_RANGE_DAYS)
            )
        return start_dt, end_dt

    async def get_portfolio_plot_data(self, backtest_id: str, start_date_str: str, end_date_str: str) -> List[Dict[str, Any]]:
        await self._validate_date_query_params(start_date_str, end_date_str)
        logger.info(f"Fetching portfolio plot data for backtest {backtest_id}: {start_date_str} to {end_date_str}")
        if not self.backtest_dao:
            raise HTTPException(status_code=503, detail="Service not initialized")
        return await self.backtest_dao.get_plot_data(backtest_id, start_date_str, end_date_str)

    async def get_trade_signal_events(self, backtest_id: str, start_date_str: str, end_date_str: str) -> List[Dict[str, Any]]:
        await self._validate_date_query_params(start_date_str, end_date_str)
        logger.info(f"Fetching signal events for backtest {backtest_id}: {start_date_str} to {end_date_str}")
        if not self.backtest_dao:
            raise HTTPException(status_code=503, detail="Service not initialized")
        return await self.backtest_dao.get_signal_events(backtest_id, start_date_str, end_date_str)

    async def generate_performance_chart_figure(self, backtest_id: str, start_date_str: str, end_date_str: str) -> go.Figure:
        plot_data = await self.get_portfolio_plot_data(backtest_id, start_date_str, end_date_str)
        trade_data = await self.get_trade_signal_events(backtest_id, start_date_str, end_date_str)
        return create_performance_chart(plot_data, trade_data)

# Instantiate Service
backtest_api_service = BacktestAPIService(app_settings=settings)

# FastAPI App
app = FastAPI(
    title="Financial Backtesting API",
    description="API for running trading strategy backtests and visualizing performance.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=settings.CORS_ALLOWED_METHODS,
    allow_headers=settings.CORS_ALLOWED_HEADERS,
)

templates = Jinja2Templates(directory="templates")

# Endpoints
@app.get("/", summary="Root Endpoint", tags=["General"])
async def root() -> Dict[str, str]:
    return {"message": "Welcome to the Backtest API"}

@app.post("/backtest", summary="Run a Trading Strategy Backtest", response_model=List[TradeResult], dependencies=[Depends(verify_api_key)], tags=["Backtest"])
async def run_backtest_endpoint(request_data: BacktestRequest) -> List[Dict[str, Any]]:
    return await backtest_api_service.run_backtest_logic(request_data)

@app.get("/events", summary="Get Signal Events", response_model=List[SignalEvent], dependencies=[Depends(verify_api_key)], tags=["Backtest"])
async def get_events_endpoint(backtest_id: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
    return await backtest_api_service.get_trade_signal_events(backtest_id, start_date, end_date)

@app.get("/performance_chart_json", summary="Get Performance Chart Data (JSON)", response_class=JSONResponse, dependencies=[Depends(verify_api_key)], tags=["Visualization"])
async def get_performance_chart_json_endpoint(backtest_id: str, start_date: str, end_date: str) -> JSONResponse:
    try:
        fig = await backtest_api_service.generate_performance_chart_figure(backtest_id, start_date, end_date)
        return JSONResponse(content=fig.to_json())
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error generating performance chart JSON")
        raise HTTPException(status_code=500, detail="Failed to generate performance chart data")

@app.get("/performance", summary="View Performance Chart Page", response_class=HTMLResponse, dependencies=[Depends(verify_api_key)], tags=["Visualization"])
async def performance_page_endpoint(http_request: Request, backtest_id: str, start_date: str, end_date: str) -> HTMLResponse:
    logger.info(f"Performance page request: backtest_id={backtest_id}, start_date={start_date}, end_date={end_date}")
    try:
        fig = await backtest_api_service.generate_performance_chart_figure(backtest_id, start_date, end_date)
        plot_json_data = fig.to_json()
        return templates.TemplateResponse(
            "performance.html",
            {
                "request": http_request,
                "plot_json": plot_json_data,
                "start_date": start_date,
                "end_date": end_date,
                "page_title": "Backtest Performance",
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error rendering performance page")
        return templates.TemplateResponse(
            "error.html",
            {
                "request": http_request,
                "error_message": "Failed to render performance page. Please try again later."
            },
            status_code=500
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
        log_level=settings.LOG_LEVEL.lower(),
    )
