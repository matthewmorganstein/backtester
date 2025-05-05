"""FastAPI-based API for backtesting and performance visualization."""
from __future__ import annotations
import logging
from datetime import datetime
from typing import Any, AsyncGenerator

import asyncpg
import pandas as pd
from asyncpg.exceptions import PostgresError
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, validator

from backtest_dao import BacktestDAO
from backtester import SignalBacktester
from dao import PostgresDAO
from settings import settings
from utils import InvalidToken, logger, verify_api_key
from visualization import create_performance_chart

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage FastAPI application lifespan for DAO initialization and cleanup."""
    logger.info("Initializing Backtest API DAOs...")
    try:
        await backtest_api.setup()
        logger.info("Backtest API DAOs initialized successfully")
    except Exception as e:
        logger.exception("Failed to initialize DAOs")
        raise
    yield
    logger.info("Shutting down Backtest API...")
    try:
        await backtest_api.dao.close_pool()
        await backtest_api.backtest_dao.close_pool()
        logger.info("Backtest API DAO pools closed successfully")
    except Exception as e:
        logger.exception("Failed to close DAO pools")

backtest_api = BacktestAPI()
app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to demo frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
templates = Jinja2Templates(directory="templates")

class BacktestRequest(BaseModel):
    start_date: str
    end_date: str
    square_threshold: float = 350
    distance_threshold: float = 0.01
    MAX_DATE_RANGE_DAYS = 30

    @validator("start_date", "end_date")
    def validate_date(cls, v: str) -> str:
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError("Date must be in YYYY-MM-DD format")

    @validator("end_date")
    def ensure_date_order(cls, v: str, values: dict) -> str:
        if "start_date" in values:
            start = datetime.strptime(values["start_date"], "%Y-%m-%d")
            end = datetime.strptime(v, "%Y-%m-%d")
            if end <= start:
                raise ValueError("end_date must be after start_date")
            if (end - start).days > cls.MAX_DATE_RANGE_DAYS:
                raise ValueError(f"Date range must not exceed {cls.MAX_DATE_RANGE_DAYS} days")
        return v

class BacktestAPI:
    def __init__(self) -> None:
        self.dao = PostgresDAO()
        self.backtest_dao = BacktestDAO()

    async def setup(self) -> None:
        """Initialize DAOs."""
        await self.dao.setup()
        await self.backtest_dao.setup()

    async def run_backtest(self, request: BacktestRequest) -> dict[str, Any]:
        logger.info("Backtest request: %s", request)
        try:
            backtester = SignalBacktester(
                start_date=request.start_date,
                end_date=request.end_date,
                square_threshold=request.square_threshold,
                distance_threshold=request.distance_threshold,
            )
            result = await backtester.backtest()
            logger.info("Backtest completed: final_portfolio_value=%s, returns=%s", result['final_portfolio_value'], result['returns'])
            return result
        except PostgresError as e:
            logger.exception("Database error")
            raise HTTPException(status_code=500, detail="Database error during backtest") from e
        except InvalidToken as e:
            logger.exception("Decryption error")
            raise HTTPException(status_code=500, detail="Failed to decrypt indicators") from e

    async def get_plot_data(self, start_date: str, end_date: str) -> list[dict]:
        return await self.backtest_dao.get_plot_data(start_date, end_date)

    async def get_signal_events(self, start_date: str, end_date: str) -> list[dict]:
        return await self.backtest_dao.get_signal_events(start_date, end_date)

@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "Welcome to the Backtest API"}

@app.post("/backtest", dependencies=[Depends(verify_api_key)])
async def run_backtest_endpoint(request: BacktestRequest) -> dict[str, Any]:
    return await backtest_api.run_backtest(request)

@app.post("/backtest/demo", dependencies=[Depends(verify_api_key)])
async def backtest_demo(request: BacktestRequest, req: Request) -> templates.TemplateResponse:
    result = await backtest_api.run_backtest(request)
    plot_data = await backtest_api.get_plot_data(request.start_date, request.end_date)
    fig = create_performance_chart(plot_data)
    plot_json = fig.to_json()
    return templates.TemplateResponse(
        "performance.html",
        {
            "request": req,
            "plot_json": plot_json,
            "start_date": request.start_date,
            "end_date": request.end_date,
            "result": result,
        },
    )

@app.get("/events", dependencies=[Depends(verify_api_key)])
async def get_events(start_date: str, end_date: str) -> list[dict]:
    return await backtest_api.get_signal_events(start_date, end_date)

@app.get("/performance_chart")
async def get_performance_chart(start_date: str, end_date: str):
    backtest_dao = BacktestDAO()
    await backtest_dao.setup()
    plot_data = await backtest_dao.get_plot_data(start_date, end_date)
    trade_data = await backtest_dao.get_signal_events(start_date, end_date)
    fig = create_performance_chart(plot_data, trade_data)
    return fig.to_json()

@app.get("/performance", dependencies=[Depends(verify_api_key)])
async def performance_page(request: Request, start_date: str, end_date: str) -> templates.TemplateResponse:
    logger.info("Performance request: start_date=%s, end_date=%s", start_date, end_date)
    try:
        plot_data = await backtest_api.get_plot_data(start_date, end_date)
        fig = create_performance_chart(plot_data)
        plot_json = fig.to_json()
        return templates.TemplateResponse(
            "performance.html",
            {
                "request": request,
                "plot_json": plot_json,
                "start_date": start_date,
                "end_date": end_date,
            },
        )
    except Exception as e:
        logger.exception("Performance error")
        raise HTTPException(status_code=500, detail="Performance fetch failed") from e

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=settings.PORT)
