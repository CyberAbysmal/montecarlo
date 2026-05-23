import montecarlo as mm

TICKERS = ["TSLA", "NVDA", "QQQ", "SPY", "PLTR"]

OPTIONS = {
    "TSLA": {"type": "call", "strike": 480.0, "expiry_days": 45, "contracts": 2},
    "NVDA": {"type": "call", "strike": 140.0, "expiry_days": 45, "contracts": 2},
    "QQQ":  {"type": "put",  "strike": 480.0, "expiry_days": 30, "contracts": 1},
    "SPY":  {"type": "put",  "strike": 540.0, "expiry_days": 30, "contracts": 1},
    "PLTR": {"type": "call", "strike": 95.0,  "expiry_days": 60, "contracts": 3},
}

prices = mm.fetch_prices(TICKERS, lookback_days=504)
mm.monte_carlo_options_sim(
    prices,
    OPTIONS,
    num_sim=20000,
    conf_level=0.95,
    risk_free=0.045,
    contract_size=100,
)