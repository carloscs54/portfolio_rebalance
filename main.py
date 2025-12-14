import yfinance as yf
import pandas as pd
import numpy as np
import cvxpy as cp
import datetime
from scipy.optimize import minimize


tickers = pd.read_csv("portfolio.csv")
tickers

stocks_targets = []

for i in tickers.index:
    stocks_targets.append((tickers.iloc[i]["Ticker"], int(tickers.iloc[i]["Target"])))

existing_holdings = {
    tickers.iloc[i]["Ticker"]: float(tickers.iloc[i]["Holding"])
    for i in tickers.index
}


def expected_return_overvalued(current_price, fair_value, annual_growth_rate, dividend_yield=0.0, years_to_revert=5):
    """
    Calculates expected annualized return for an overvalued stock.
    
    Parameters:
        current_price (float): Current market price of the stock
        fair_value (float): Estimated intrinsic value
        annual_growth_rate (float): Expected EPS or revenue growth rate (decimal, e.g., 0.08 for 8%)
        dividend_yield (float): Annual dividend yield (decimal, e.g., 0.02 for 2%)
        years_to_revert (int): Number of years until price is expected to revert to fair value

    Returns:
        float: Annualized expected return (decimal)
    """
    # Step 1: Base return from growth + dividends
    base_return = annual_growth_rate + dividend_yield
    
    # Step 2: Price reversion impact
    reversion_factor = (fair_value / current_price) ** (1 / years_to_revert) - 1
    
    # Step 3: Total expected return
    total_expected_return = base_return + reversion_factor
    return total_expected_return


def standard_deviation(weights, cov_matrix):
  variance = weights.T @ cov_matrix @ weights
  return np.sqrt(variance)

def get_stock_metrics(ticker, start_date, end_date, target_price):
    data = yf.download(ticker, start=start_date, end=end_date)
    data["Daily_Return"] = data["Close"].pct_change()

    avg_daily_return = data["Daily_Return"].mean()
    log_returns = np.log(data["Close"] / data["Close"].shift(1))
    log_returns = log_returns.dropna()
    daily_vol = log_returns.std()
    log_returns_mean = log_returns.mean()
    log_returns = log_returns_mean.values[0] * 252
    sharpe = (log_returns_mean.values[0] / daily_vol.values[0]) * np.sqrt(252) if daily_vol.values[0] > 0 else 0


    current_price = data["Close"].iloc[-1].values[0]

    expected_return = (target_price - current_price) / current_price
    if expected_return < 0:
        expected_return = expected_return_overvalued(current_price, target_price, 0.06, 0.00, 4)
    else:
        expected_return

    return {
        "Ticker": ticker,
        "CurrentPrice": current_price,
        "ExpectedReturn": expected_return,
        "SharpeRatio": sharpe,
        "DailyReturns": data["Daily_Return"].dropna()
    }

def rebalance_portfolio(stocks_targets, start_date, end_date,
                        total_portfolio_value, new_cash=0,
                        existing_holdings=None,
                        risk_free_rate=0.08,
                        risk_aversion=0.5,
                        upside_weight=1.0, max_weight=0.3):
    """
    Optimizer that takes covariance into account in the objective.

    Parameters:
      - stocks_targets: list of (ticker, target_price)
      - start_date, end_date: for history
      - total_portfolio_value: not strictly used for optimization, used to compute allocations
      - new_cash: extra cash to be allocated according to optimized weights
      - existing_holdings: dict ticker -> current value
      - risk_free_rate: used for portfolio Sharpe calculation
      - risk_aversion: lambda for vol penalty in objective (higher -> more conservative)
      - upside_weight: multiplier for weighted upside term in objective
    Returns:
      - df (DataFrame) with results and metrics, and portfolio-level metrics
    """

    metrics_list = []
    # collect metrics
    for ticker, tgt in stocks_targets:
        m = get_stock_metrics(ticker, start_date, end_date, tgt)
        metrics_list.append(m)

    df = pd.DataFrame(metrics_list)
    if df.empty:
        raise ValueError("No metrics collected. Check tickers and data availability.")

    # prepare arrays
    # Expected returns (annualized). We already used target-based expected returns in get_stock_metrics.
    mu = df["ExpectedReturn"].values.astype(float)        # shape (n,)
    upside = mu.copy()                                   # you can use a separate upside field if desired
    tickers_list = df["Ticker"].tolist()

    # Build aligned daily returns DataFrame for covariance
    returns_df = pd.concat([m["DailyReturns"].rename(m["Ticker"]) for m in metrics_list], axis=1).dropna()
    # If returns_df is empty (no overlapping days), try concatenating without dropna
    if returns_df.shape[0] == 0:
        returns_df = pd.concat([m["DailyReturns"].rename(m["Ticker"]) for m in metrics_list], axis=1)

    cov_daily = returns_df.cov()
    cov_annual = cov_daily * 252
    cov = cov_annual.values

    n = len(mu)
    bounds = [(0, max_weight) for _ in range(n)]
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

    def objective(w):
        port_return = np.dot(w, mu)
        port_vol = np.sqrt(w.T @ cov @ w)
        return risk_aversion * port_vol - port_return

    initial_guess = np.repeat(1/n, n)
    result = minimize(objective, initial_guess, bounds=bounds, constraints=constraints, method='SLSQP')

    if not result.success:
        raise Exception("Optimization failed: " + result.message)

    weights_opt = result.x
    df["Weight"] = weights_opt

    # Allocate only the new cash according to optimized weights
    df["NewCashAllocation"] = df["Weight"] * new_cash

    # Existing holdings and current weights
    if existing_holdings:
        df["CurrentHolding"] = df["Ticker"].apply(lambda t: existing_holdings.get(t, 0.0))
        total_existing = df["CurrentHolding"].sum()
        df["current_weight"] = df["CurrentHolding"] / total_existing if total_existing > 0 else 0.0
    else:
        df["CurrentHolding"] = 0.0
        df["current_weight"] = 0.0

    # New total allocation (existing + new cash allocation)
    df["NewTotalAllocation"] = df["CurrentHolding"] + df["NewCashAllocation"]
    total_after = df["NewTotalAllocation"].sum()
    df["FinalWeight"] = df["NewTotalAllocation"] / total_after if total_after > 0 else df["Weight"]

    # Portfolio-level metrics using covariance
    port_expected_return = float(np.dot(df["FinalWeight"].values, mu))
    port_vol = float(np.sqrt(df["FinalWeight"].values @ cov_annual @ df["FinalWeight"].values))
    port_sharpe = (port_expected_return - risk_free_rate) / port_vol if port_vol > 0 else np.nan

    # previous portfolio metrics (if existing holdings provided)
    prev_return = None; prev_vol = None; prev_sharpe = None
    if existing_holdings and df["CurrentHolding"].sum() > 0:
        prev_w = df["current_weight"].values
        prev_return = float(np.dot(prev_w, mu))
        prev_vol = float(np.sqrt(prev_w @ cov_annual @ prev_w))
        prev_sharpe = (prev_return - risk_free_rate) / prev_vol if prev_vol > 0 else np.nan
    else:
        prev_return = np.nan; prev_vol = np.nan; prev_sharpe = np.nan

    # Sort by allocation for clarity
    df = df.sort_values("NewCashAllocation", ascending=False).reset_index(drop=True)

    # Print summary
    print(f"\nPrev Return: {prev_return:.4f}, Prev Vol: {prev_vol:.4f}, Prev Sharpe: {prev_sharpe:.4f}")
    print(f"Post Return: {port_expected_return:.4f}, Post Vol: {port_vol:.4f}, Post Sharpe: {port_sharpe:.4f}")

    return df, prev_return, port_expected_return, prev_vol, port_vol, prev_sharpe, port_sharpe


# ==== Example Usage ====

start = "2015-01-01"
end = datetime.datetime.today().strftime('%Y-%m-%d')

df_rebalance, previous_return, port_return, previous_vol, port_vol, previous_sharpe, port_sharpe = rebalance_portfolio(
    stocks_targets,
    start,
    end,
    total_portfolio_value=sum(existing_holdings.values()),
    new_cash=50000,
    existing_holdings=existing_holdings
)

print(df_rebalance)
df_rebalance.to_csv("df_rebalance.csv")

""" performance = []

for i in range(0,20):
    data = rebalance_portfolio(
    stocks_targets,
    start,
    end,
    total_portfolio_value=sum(existing_holdings.values()),
    new_cash=1000000*i,
    existing_holdings=existing_holdings
    )
    performance.append(data)

perf = pd.DataFrame(performance)
perf.to_csv("performance.csv") """""