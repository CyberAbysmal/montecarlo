import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import re

# --- 1. Monte Carlo Simulation Function ---
def monte_carlo_portfolio_sim(prices_dict, weights, time_horizon=252, num_sim=20000, conf_level=0.95):

    tickers = list(prices_dict.keys())
    weights = np.array(weights)

    try:
        prices = np.array([prices_dict[ticker] for ticker in tickers])
    except ValueError as e:
        print("ERROR: Price arrays have different lengths.")
        print(f"Details: {e}")
        return

    n_assets, n_price_points = prices.shape

    if n_price_points < 2:
        raise ValueError(f"Insufficient price data for returns. Found only {n_price_points} points.")

    log_returns = np.log(prices[:, 1:] / prices[:, :-1])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mu = np.mean(log_returns, axis=1)

    cov = np.cov(log_returns)

    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        print("ERROR: Covariance matrix not positive definite.")
        return

    results = np.zeros((num_sim, time_horizon + 1))
    initial_prices = prices[:, -1]
    weighted_init = np.dot(weights, initial_prices)
    results[:, 0] = weighted_init

    for i in range(num_sim):
        curr_prices = initial_prices.copy()
        for t in range(1, time_horizon + 1):
            rand_normals = np.random.normal(size=n_assets)
            correlated_normals = np.dot(L, rand_normals)
            daily_returns = mu + correlated_normals
            curr_prices = curr_prices * np.exp(daily_returns)
            results[i, t] = np.dot(weights, curr_prices)

    final_vals = results[:, -1]
    val_at_risk_level = np.percentile(final_vals, (1 - conf_level) * 100)
    VaR = weighted_init - val_at_risk_level
    percentiles = np.percentile(final_vals, [10, 25, 50, 75, 90])

    print(f"\n--- Simulation Results ---")
    print(f"Initial Portfolio Value: ${weighted_init:,.2f}")
    print(f"Value at Risk ({int(conf_level*100)}%): ${VaR:,.2f}")
    print(f"Loss will not exceed this with {int(conf_level*100)}% confidence.\n")

    print("Final Value Percentiles:")
    print({f'P{p}': f'${val:,.2f}' for p, val in zip([10, 25, 50, 75, 90], percentiles)})

    plt.figure(figsize=(10, 6))
    plt.hist(final_vals, bins=60, alpha=0.7, density=True)
    plt.title(f"Portfolio Monte Carlo Distribution ({num_sim:,} runs)")
    plt.xlabel("Final Portfolio Value")
    plt.ylabel("Density")
    plt.axvline(val_at_risk_level, linestyle='dashed', linewidth=2)
    plt.axvline(weighted_init, linestyle='dotted', linewidth=2)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('portfolio_simulation.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'portfolio_simulation.png'")


# --- 2. Fake Price Generator ---
def generate_fake_price_data(start_price, n_days, mu, sigma):
    mu_adj = mu + np.random.normal(0, 0.00001)
    sigma_adj = sigma + np.random.normal(0, 0.0001)

    returns = np.random.normal(mu_adj, sigma_adj, n_days)
    prices_rev = start_price * np.exp(np.cumsum(-returns))
    prices_hist = prices_rev[::-1]
    return np.append(prices_hist, start_price)


# --- 3. Main Execution ---
if __name__ == "__main__":
    raw_tickers = "EOSE QS IREN RKLB TSLA IBIT GOOG SOFI BABA XIACY BIDU HDB TCEHY STRL SPY GLD ESGU URBN VXUS ACWI KO PEP VZ T TMUS UNH VTI O PLD"
    raw_prices = "$14.55 $13.80 $65.00 $66.15 $461.00 $63.00 $252.34 $28.61 $165.84 $29.68 $120.15 $36.91 $80.06 $349.38 $671.38 $386.33 $146.12 $66.75 $74.19 $139.00 $69.71 $151.55 $38.82 $25.14 $217.77 $364.10 $328.79 $60.00 $126.43"
    raw_holdings = "$5,621.00 $4,205.00 $20,800.00 $9,922.50 $11,525.00 $9,450.00 $11,722.95 $5,802.00 $3,494.00 $2,950.00 $2,455.20 $2,205.60 $2,054.00 $10,235.16 $47,407.50 $18,876.00 $10,376.10 $8,132.40 $5,996.00 $4,225.80 $12,199.25 $12,124.00 $9,705.00 $10,056.00 $5,444.25 $54,375.00 $13,348.40 $24,000.00 $25,286.00"

    print("--- Data Processing ---")

    def clean_currency_string(s):
        return s.replace('$', '').replace(',', '')

    try:
        tickers_list = raw_tickers.split(' ')
        prices_list = [float(clean_currency_string(p)) for p in raw_prices.split(' ')]
        holdings_list = [float(clean_currency_string(h)) for h in raw_holdings.split(' ')]

        if not (len(tickers_list) == len(prices_list) == len(holdings_list)):
            raise ValueError("Ticker/price/holding count mismatch")

        current_prices = dict(zip(tickers_list, prices_list))
        total_portfolio_value = sum(holdings_list)
        weights = [h / total_portfolio_value for h in holdings_list]

        print(f"Total Portfolio Value: ${total_portfolio_value:,.2f}")

        print("\n--- Generating Fake Data for Simulation ---")
        fake_prices_dict = {}
        n_history = 252

        for ticker, price in current_prices.items():
            fake_mu = np.random.uniform(0.0001, 0.0005)
            fake_sigma = np.random.uniform(0.01, 0.03)
            fake_prices_dict[ticker] = generate_fake_price_data(price, n_history, fake_mu, fake_sigma)

        monte_carlo_portfolio_sim(fake_prices_dict, weights, time_horizon=252, num_sim=10000)

    except Exception as e:
        print(f"Error: {e}")
