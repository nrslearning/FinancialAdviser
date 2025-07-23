
import pandas as pd
#from app.stock_analysis import get_sentiment_from_news
from app.config import settings
from app.data_cache import ohlc_data, news_data, fundamentals_data, forecast_data, sentiment_score_data

# Load sector mapping from file
nifty_path = settings.DATA_DIR / "nifty_500_lst.csv"
nifty_df = pd.read_csv(nifty_path)
symbol_to_sector = dict(zip(nifty_df["Symbol"], nifty_df["Industry"].str.lower()))

def compute_user_resilience(user):
    resilience = 0
    disposable_income = user["monthly_income"] - user["monthly_expenses"]

    if disposable_income > 30000:
        resilience += 10
    elif disposable_income > 15000:
        resilience += 5

    if user["has_health_insurance"]:
        resilience += 5
    if user["has_emergency_fund"]:
        resilience += 5

    if user["current_savings"] > 6 * user["monthly_expenses"]:
        resilience += 5

    if user["num_dependents"] >= 3:
        resilience -= 5
    if disposable_income < 10000:
        resilience -= 10
    if user["monthly_investment"] < 0.1 * user["monthly_income"]:
        resilience -= 5

    if user["age"] >= 18 and user["age"] < 25:
        resilience += 10
    elif user["age"] >= 25 and user["age"] < 40:
        resilience += 7
    elif user["age"] >= 40 and user["age"] < 50:
        resilience += 5
    elif user["age"] >= 50:
        resilience += 2

    return resilience

def rank_top_stocks(user_input: dict) -> list[str]:
    expected_return = user_input["expected_returns_percent"] / 100
    risk_tolerance = user_input["risk_percent"]
    investment_type = user_input["investment_type"]
    sector_prefs = [s.lower() for s in user_input.get("interested_sectors", [])]
    resilience = compute_user_resilience(user_input)

    stock_scores = []
    temp = 0

    # print(">>>>Sector:", sector_prefs)
    # print(">>>>Resilience:", resilience)
    # print(">>>>Investment_type:", investment_type)

    for symbol in settings.INDEX_STOCKS:
        try:
            if symbol not in ohlc_data or symbol not in news_data or symbol not in fundamentals_data:
                continue
            
            # Get sector from external file instead of fundamentals
            sector = symbol_to_sector.get(symbol, "")

            #If it is not the sector of intrest skip
            if sector_prefs and all(pref not in sector for pref in sector_prefs):
                continue
            
            #print(f"{symbol}\t: \t {sector}")
            
            growth = forecast_data.get(symbol, 0.0)
            df = ohlc_data[symbol].copy()
            df["returns"] = df["Close"].pct_change()
            volatility = df["returns"].std()

            #news_df = news_data[symbol]
            #sentiment_score, _ = get_sentiment_from_news(news_df)
            sentiment_score = sentiment_score_data.get(symbol, 0.0)

            fundamentals_df = fundamentals_data[symbol]
            pe = float(fundamentals_df.get("trailingPE", [40])[0])

            score = 0
            if sector_prefs and any(pref in sector for pref in sector_prefs):
                score += 10

            if risk_tolerance < 20 and volatility > 0.03:
                score -= 10
            elif volatility < 0.03:
                score += 5

            if investment_type == "Aggressive" and pe < 30 and growth > 0.1:
                score += 15
            elif investment_type == "Moderate" and 0.05 <= growth <= 0.1:
                score += 10
            elif investment_type == "Slow" and pe < 20:
                score += 5

            score += 20 * sentiment_score
            score += (growth - expected_return) * 100

            if pe > 50:
                score -= 5

            if resilience < 10 and volatility < 0.03:
                score += 5
            elif resilience > 15 and growth > 0.10:
                score += 10
            elif resilience < 5 and pe > 50:
                score -= 10

            stock_scores.append((symbol, score))

        except Exception as e:
            print(f"Error ranking {symbol}: {e}")
            continue

    ranked = sorted(stock_scores, key=lambda x: x[1], reverse=True)

    #print("######### rank_top_stocks ###########")
    #print(ranked)
    
    #print("######### sector_prefs ###########")
    #print(sector_prefs)
    
    # Filter by user sector preferences if any
    if sector_prefs:
        ranked = [
            (symbol, score) for symbol, score in ranked
            if symbol_to_sector.get(symbol, "") and any(
                pref in symbol_to_sector.get(symbol, "") for pref in sector_prefs
            )
        ]
    
    # # Enforce mix of sectors in final top list
    # selected = []
    # seen_sectors = set()
    # for symbol, _ in ranked:
    #     fundamentals_path = settings.FUNDMENTALS_DIR / f"{symbol}_fundamentals_{settings.TODAY}.csv"
    #     if fundamentals_path.exists():
    #         sector = pd.read_csv(fundamentals_path).get("sector", [""])[0].lower()
    #         if sector not in seen_sectors:
    #             selected.append(symbol)
    #             seen_sectors.add(sector)
    #     if len(selected) >= settings.NUM_SCREENED_STOCKS:
    #         break

    return ranked
