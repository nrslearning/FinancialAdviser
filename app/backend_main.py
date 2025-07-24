
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from app.stock_analysis import summarize_stock_insights
from app.rank_top_stocks import rank_top_stocks
from app.llm_prompt import get_final_stock_advice
from app.data_cache import load_all_data
from app.data_fetcher_yfinance import fetch_and_save_all_stocks
from app.config import settings
from fastapi.responses import JSONResponse

app = FastAPI()

class UserProfile(BaseModel):
    age: int
    monthly_income: float
    monthly_expenses: float
    monthly_investment: float
    annual_extra_investment: float
    current_savings: float
    risk_percent: int
    years: int
    expected_returns_percent: int
    num_dependents: int
    has_health_insurance: bool
    has_emergency_fund: bool
    investment_type: str
    interested_sectors: List[str]
    investor_knowledge: str

@app.on_event("startup")
def on_startup():
    print("App is starting... loading data cache")
    # Check if today's data exists
    any_missing = False
    for symbol in settings.INDEX_STOCKS[:1]:  # Just check a few to avoid long loops
        ohlc_path = settings.OHLC_DIR / f"{symbol}_OHLC_{settings.TODAY}.csv"
        news_path = settings.NEWS_DIR / f"{symbol}_news_{settings.TODAY}.csv"
        fund_path = settings.FUNDMENTALS_DIR / f"{symbol}_fundamentals_{settings.TODAY}.csv"
        if not (ohlc_path.exists() and news_path.exists() and fund_path.exists()):
            any_missing = True
            break

    if any_missing:
        print("Data for today not found. Fetching fresh stock data...")
        fetch_and_save_all_stocks()
    else:
        print("Today's data is already present. Skipping fetch.")

    print("Loading data into cache...")
    load_all_data()

@app.get("/")
def health():
    return JSONResponse({"status": "ok", "message": "FastAPI backend is running."})

@app.post("/recommend")
async def recommend(user: UserProfile):
    # Validation
    errors = []
    if user.monthly_expenses > user.monthly_income:
        errors.append("Monthly expenses cannot exceed income.")
    disposable_income = user.monthly_income - user.monthly_expenses
    if disposable_income <= 0:
        errors.append("Disposable income is zero or negative.")
    elif user.monthly_investment > disposable_income:
        errors.append("Investment exceeds disposable income.")

    if errors:
        raise HTTPException(status_code=400, detail=errors)

    user_dict = user.dict()
    user_dict["investment_amount"] = user.monthly_investment * 12 * user.years + user.annual_extra_investment * user.years
    top10 = rank_top_stocks(user_dict)[:settings.NUM_SCREENED_STOCKS]
    #summaries = {symbol.upper(): summarize_stock_insights(symbol) for symbol, _ in top10}
    # now that rank_top_stocks returns List[str], unpack only symbol
    summaries = {
        symbol.upper(): summarize_stock_insights(symbol)
        for symbol in top10
    }
    advice, updated_summaries = get_final_stock_advice(user_dict, summaries)

    return {
        "user_profile": user_dict,
        "top_stocks": top10,
        "advice": advice,
        "growth_data": updated_summaries
    }
