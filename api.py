from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from typing import Dict, Any, List
from contextlib import asynccontextmanager
from backtester import SignalBacktester
from backtest_dao import BacktestDAO
from dao import PostgresDAO
from visualization import create_performance_chart
from utils import verify_api_key, logger
from settings import settings
from datetime import datetime
from asyncpg.exceptions import PostgresError

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage FastAPI application lifespan for DAO initialization and cleanup."""
    logger.info("Initializing Backtest API DAOs...")
    try:
        await backtest_api.setup()
        logger.info("Backtest API DAOs initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize DAOs: {e}")
        raise
    yield
    logger.info("Shutting down Backtest API...")
    try:
        await backtest_api.dao.close_pool()
        await backtest_api.backtest_dao.close_pool()
        logger.info("Backtest API DAO pools closed successfully")
    except Exception as e:
        logger.error(f"Failed to close DAO pools: {e}")

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

    @validator("start_date", "end_date")
    def validate_date(cls, v):
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError("Date must be in YYYY-MM-DD format")

    @validator("end_date")
    def ensure_date_order(cls, v, values):
        if "start_date" in values:
            start = datetime.strptime(values["start_date"], "%Y-%m-%d")
            end = datetime.strptime(v, "%Y-%m-%d")
            if end <= start:
                raise ValueError("end_date must be after start_date")
            if (end - start).days > 30:
                raise ValueError("Date range must not exceed 30 days")
        return v

class BacktestAPI:
    def __init__(self):
        self.dao = PostgresDAO()
        self.backtest_dao = BacktestDAO()

    async def setup(self):
        """Initialize DAOs."""
        await self.dao.setup()
        await self.backtest_dao.setup()

    async def run_backtest(self, request: BacktestRequest) -> Dict[str, Any]:
        logger.info(f"Backtest request: {request}")
        try:
            backtester = SignalBacktester(
                start_date=request.start_date,
                end_date=request.end_date,
                square_threshold=request.square_threshold,
                distance_threshold=request.distance_threshold
            )
            result = await backtester.backtest()
            logger.info(f"Backtest completed: final_portfolio_value={result['final_portfolio_value']}, returns={result['returns']}")
            return result
        except PostgresError as e:
            logger.error(f"Database error: {str(e)}")
            raise HTTPException(status_code=500, detail="Database error during backtest")
        except InvalidToken as e:
            logger.error(f"Decryption error: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to decrypt indicators")
        except Exception as e:
            logger.error(f"Backtest error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Backtest failed: {str(e)}")

    async def get_plot_data(self, start_date: str, end_date: str) -> List[Dict]:
        return await self.backtest_dao.get_plot_data(start_date, end_date)

    async def get_signal_events(self, start_date: str, end_date: str) -> List[Dict]:
        return await self.backtest_dao.get_signal_events(start_date, end_date)

backtest_api = BacktestAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to the Backtest API"}

@app.post("/backtest", dependencies=[Depends(verify_api_key)])
async def run_backtest_endpoint(request: BacktestRequest) -> Dict[str, Any]:
    return await backtest_api.run_backtest(request)

@app.post("/backtest/demo", dependencies=[Depends(verify_api_key)])
async def backtest_demo(request: BacktestRequest, req: Request):
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
            "result": result
        }
    )

@app.get("/events", dependencies=[Depends(verify_api_key)])
async def get_events(start_date: str, end_date: str) -> List[Dict]:
    return await backtest_api.get_signal_events(start_date, end_date)

@app.get("/performance", dependencies=[Depends(verify_api_key)])
async def performance_page(request: Request, start_date: str, end_date: str):
    logger.info(f"Performance request: start_date={start_date}, end_date={end_date}")
    try:
        plot_data = await backtest_api.get_plot_data(start_date, end_date)
        fig = create_performance_chart(plot_data)
        plot_json = fig.to_json()
        return templates.TemplateResponse(
            "performance.html",
            {"request": request, "plot_json": plot_json, "start_date": start_date, "end_date": end_date}
        )
    except Exception as e:
        logger.error(f"Performance error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Performance fetch failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=settings.PORT)
