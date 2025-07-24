
import pandas as pd
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

# 1. Helper functions for raw factor calculation
def compute_momentum(df, days=63):
    returns = df["Close"].pct_change().dropna()
    return returns.tail(days).mean()

def compute_risk(df, beta):
    returns = df["Close"].pct_change().dropna()
    vol = returns.std()
    return vol * beta  # or use max drawdown

def build_metrics_df(symbols, sector_map):
    rows = []
    missing = []
    for sym in symbols:
        if sym not in ohlc_data or sym not in fundamentals_data:
            missing.append(sym)
            continue
        
        ohlc = ohlc_data[sym]
        f = fundamentals_data[sym]
        rows.append({
            "symbol": sym,
            "sector": sector_map.get(sym, ""),
            "value_pe": -(f["forwardPE"][0] + f["trailingPE"][0]) / 2,
            "value_div": f.get("dividendYield", [0])[0],
            "growth_rev": f.get("revenueGrowth", [0])[0],
            "growth_eps": f.get("earningsGrowth", [0])[0],
            "quality_roe": f.get("returnOnEquity", [0])[0],
            "quality_margin": (f.get("grossMargins",[0])[0] + f.get("profitMargins",[0])[0]) / 2,
            "momentum": compute_momentum(ohlc),
            "risk_beta": f.get("beta", [1])[0],
            "risk_vol": ohlc["Close"].pct_change().std(),
            "sentiment": sentiment_score_data.get(sym, 0.0),
            "forecast": forecast_data.get(sym, 0.0)
        })
    
    if missing:
        # you can also use logging.warning() here if you prefer
        print(f"[build_metrics_df] skipped symbols with no data: {missing}")

    return pd.DataFrame(rows)

# 2. Main ranking
def rank_top_stocks(user_input: dict) -> list[str]:
    # Unpack user inputs
    exp_ret = user_input["expected_returns_percent"] / 100
    risk_tol = user_input["risk_percent"] / 100
    inv_type = user_input["investment_type"]
    prefs = [s.lower() for s in user_input.get("interested_sectors", [])]
    resilience = compute_user_resilience(user_input)  # reuse yours

    # Build raw metrics
    metrics = build_metrics_df(settings.INDEX_STOCKS, symbol_to_sector)

    # Factor construction
    metrics["value_score"]   = (metrics["value_pe"].rank() + metrics["value_div"].rank()) / 2
    metrics["growth_score"]  = metrics["growth_rev"] + metrics["growth_eps"]
    metrics["quality_score"] = metrics["quality_roe"] + metrics["quality_margin"]
    metrics["momentum_score"]= metrics["momentum"]
    metrics["risk_score"]    = metrics["risk_vol"] * metrics["risk_beta"]
    metrics["sentiment_score"]= metrics["sentiment"]

    # Standardize all factor scores
    factor_cols = [
        "value_score","growth_score","quality_score",
        "momentum_score","sentiment_score","risk_score"
    ]
    for col in factor_cols:
        metrics[col] = (metrics[col] - metrics[col].mean()) / metrics[col].std()

    # Determine weights based on risk & type
    # Example mapping (you can tune these):
    if inv_type == "Aggressive":
        weights = {
            "value_score": 0.1, "growth_score": 0.35,
            "quality_score": 0.1, "momentum_score": 0.25,
            "sentiment_score": 0.1, "risk_score": 0.1
        }
    elif inv_type == "Moderate":
        weights = {**dict.fromkeys(factor_cols, 1/6)}
    else:  # Slow/Conservative
        weights = {
            "value_score": 0.3, "growth_score": 0.1,
            "quality_score": 0.3, "momentum_score": 0.1,
            "sentiment_score": 0.1, "risk_score": 0.1
        }
    # Mix in user’s risk tolerance & resilience
    weights["risk_score"] *= (1 - risk_tol) * (resilience/20)

    # Composite score
    metrics["composite"] = sum(
        weights[f] * metrics[f] for f in factor_cols
    )

    # Sector tilt & diversification
    if prefs:
        # +0.2 bonus to preferred sectors
        metrics.loc[metrics["sector"].isin(prefs), "composite"] += 0.2
    # Limit max 2 per sector
    top = []
    for _, row in metrics.sort_values("composite", ascending=False).iterrows():
        sec = row["sector"]
        if sum(1 for t in top if symbol_to_sector[t] == sec) >= 2:
            continue
        top.append(row["symbol"])
        if len(top) >= settings.NUM_SCREENED_STOCKS:
            break

    return top
