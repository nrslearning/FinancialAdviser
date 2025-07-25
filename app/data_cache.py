import os
import pandas as pd
from pathlib import Path
from app.config import settings
from prophet import Prophet
import logging
from transformers import pipeline

# Suppress Prophet warnings
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

# Global caches
ohlc_data = {}
news_data = {}
fundamentals_data = {}
forecast_data = {}
sentiment_score_data = {}  # key: symbol, value: sentiment_score


sentiment_model = pipeline("sentiment-analysis",
                           model="distilbert-base-uncased-finetuned-sst-2-english")

def get_sentiment_from_news(df: pd.DataFrame) -> tuple[float, list[str]]:
    #print("### get_sentiment_from_news")
    headlines = df["title"].dropna().tolist()[:3]
    if not headlines:
        return 0.0, []
    results = sentiment_model(headlines)
    score = sum(r["score"] if r["label"] == "POSITIVE" else -r["score"] for r in results)
    return round((score / len(results)), 2), headlines

def load_all_data():
    print("⏳ Loading all OHLC, news, fundamentals, and predictions...")
    for symbol in settings.INDEX_STOCKS:
        try:
            ohlc_path = settings.OHLC_DIR / f"{symbol}_OHLC_{settings.TODAY}.csv"
            news_path = settings.NEWS_DIR / f"{symbol}_news_{settings.TODAY}.csv"
            fundamentals_path = settings.FUNDMENTALS_DIR / f"{symbol}_fundamentals_{settings.TODAY}.csv"

            if ohlc_path.exists():
                ohlc_df = pd.read_csv(ohlc_path)
                ohlc_data[symbol] = ohlc_df
                #print("load_all_data1")
                forecast_data[symbol] = compute_forecast_growth(ohlc_df)
                #print(f"forecast_data[{symbol}]: {forecast_data[symbol]}")

            if news_path.exists():
                news_df = news_data[symbol] = pd.read_csv(news_path)
                try:
                    sentiment_score, _ = get_sentiment_from_news(news_df)
                    sentiment_score_data[symbol] = sentiment_score
                    print(f"\n{symbol}:{sentiment_score}")
                except Exception as e:
                    print(f"⚠️ Sentiment error for {symbol}: {e}")
                    sentiment_score_data[symbol] = 0.0
            
            if fundamentals_path.exists():
                fundamentals_data[symbol] = pd.read_csv(fundamentals_path)

        except Exception as e:
            print(f"⚠️ Error loading data for {symbol}: {e}")

    print(f"✅ Loaded data for {len(ohlc_data)} stocks.")


def compute_forecast_growth(df):
    #print("compute_forecast_growth 1")
    try:
        df = df.copy()
        #df["ds"] = pd.to_datetime(df["Date"])
        df["ds"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)

        df = df.rename(columns={"Close": "y"})

        logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
        logging.getLogger("prophet").setLevel(logging.ERROR)
        #print("compute_forecast_growth 2")
        if 0: #{"Open", "High", "Low", "Volume"}.issubset(df.columns):
            model = Prophet()
            for col in ["Open", "High", "Low", "Volume"]:
                df[col] = df[col].ffill()
                model.add_regressor(col)
            model.fit(df[["ds", "y", "Open", "High", "Low", "Volume"]])
            future = model.make_future_dataframe(periods=30)
            for col in ["Open", "High", "Low", "Volume"]:
                future[col] = df[col].iloc[-1]
        else:
            #print("compute_forecast_growth 3")
            model = Prophet()
            model.fit(df[["ds", "y"]])
            future = model.make_future_dataframe(periods=30)

        forecast = model.predict(future)
        y_current = round(df["y"].iloc[-1], 2)
        y_future = round(forecast["yhat"].iloc[-1], 2)
        growth = (y_future - y_current) / y_current
        if growth <= 0.03:
            growth = 0.04
        print(f"y_current:{y_current}, y_future: {y_future}, growth:{round(growth, 2)}")
        return round(growth, 4)
    except Exception as e:
        print(f"Forecast error: {e}")
        return 0.0


# Automatically load on module import
#load_all_data()
