# Portfolio Rebalancing Optimizer (Quantitative)

This project implements a **mean–variance portfolio rebalancing optimizer** that combines **forward-looking expected returns**, **covariance-based risk estimation**, and **constrained optimization**. The design follows standard quantitative finance principles while making each calculation economically interpretable.

---

## Dependencies

```bash
pip install yfinance pandas numpy scipy cvxpy
```

---

## Input Data

The optimizer expects a CSV file named `portfolio.csv`:

| Column  | Description                   |
| ------- | ----------------------------- |
| Ticker  | Yahoo Finance ticker symbol   |
| Target  | Estimated fair / target price |
| Holding | Current dollar allocation     |

---

## Expected Return Model

Expected returns are **forward-looking** and derived from valuation assumptions rather than historical averages.

### Undervalued Assets

[
\mu_i = \frac{P_i^{\text{target}} - P_i}{P_i}
]

**What this does:**
This computes the percentage upside between the current market price and the estimated fair value. The model assumes the market will eventually close this valuation gap.

**Interpretation:**
If the stock converges to its target price, this is the implied return.

---

### Overvalued Assets (Mean Reversion)

[
\mu_i = g + d + \left( \frac{P_i^{\text{target}}}{P_i} \right)^{1/T} - 1
]

**What this does:**
When a stock trades above fair value, expected return is modeled via **gradual mean reversion** rather than simple negative upside.

* The reversion term annualizes the valuation correction over (T) years
* Growth (g) and dividends (d) partially offset valuation drag

**Interpretation:**
Even overpriced assets may deliver returns as fundamentals catch up over time.

---

## Risk Model

Risk is estimated from historical price behavior and asset co-movement.

### Log Returns

[
r_{i,t} = \ln\left(\frac{P_{i,t}}{P_{i,t-1}}\right)
]

**What this does:**
Log returns are time-additive and statistically well-behaved, making them suitable for covariance estimation.

---

### Covariance Matrix

[
\Sigma = 252 \cdot \text{Cov}(r)
]

**What this does:**
The covariance matrix captures how assets move relative to one another. Annualization assumes 252 trading days.

**Interpretation:**
Diversification benefits arise when assets exhibit low or negative covariance.

---

### Portfolio Volatility

[
\sigma_p = \sqrt{\mathbf{w}^T \Sigma \mathbf{w}}
]

**What this does:**
Aggregates individual asset risk and correlations into a single portfolio-level risk metric.

---

## Optimization Problem

The optimizer balances expected return against total portfolio risk.

[
\min_{\mathbf{w}} ; \lambda , \sigma_p - \mathbf{w}^T \boldsymbol{\mu}
]

**What this does:**

* Rewards higher expected return
* Penalizes volatility
* (\lambda) controls the aggressiveness of the portfolio

---

### Constraints

[
\sum_i w_i = 1
]

**Interpretation:**
The portfolio is fully invested with no leverage or idle cash.

[
0 \le w_i \le w_{\max}
]

**Interpretation:**
Long-only and position limits reduce concentration and estimation risk.

---

## Portfolio Metrics

### Expected Portfolio Return

[
E[R_p] = \mathbf{w}^T \boldsymbol{\mu}
]

**What this does:**
Computes the weighted average of asset-level expected returns.

---

### Sharpe Ratio

[
S = \frac{E[R_p] - r_f}{\sigma_p}
]

**What this does:**
Measures risk-adjusted performance by comparing excess return to total volatility.

---

## Output

The optimizer returns:

* Asset-level optimized weights
* New cash allocation per asset
* Final post-rebalance portfolio weights
* Pre- and post-rebalance return, volatility, and Sharpe ratio

Results are exported to:

```text
df_rebalance.csv
```

---

## Notes

* Expected returns are **valuation-driven**, not historical averages
* No short selling or transaction costs
* Single-period optimization

---

## Disclaimer

This project is for educational and research purposes only and does not constitute financial advice.
