# Portfolio Rebalancing Optimizer (Quantitative)

This project implements a **mean–variance portfolio rebalancing optimizer** that incorporates **target-price-based expected returns**, **covariance-based risk estimation**, and **constrained optimization** to allocate new capital and evaluate portfolio efficiency.

The model is designed for **quantitative portfolio analysis**, combining forward-looking return assumptions with historical risk estimates.

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
| Ticker  | Yahoo Finance ticker          |
| Target  | Estimated fair / target price |
| Holding | Current dollar allocation     |

---

## Expected Return Model

For each asset *i*, expected return ( \mu_i ) is defined as:

### Undervalued assets

[
\mu_i = \frac{P_i^{\text{target}} - P_i}{P_i}
]

### Overvalued assets (mean reversion)

If ( P_i > P_i^{\text{target}} ), expected return is modeled as:
[
\mu_i = g + d + \left( \frac{P_i^{\text{target}}}{P_i} \right)^{1/T} - 1
]

Where:

* ( g ) = long-term growth rate
* ( d ) = dividend yield
* ( T ) = years to price reversion

---

## Risk Model

* Daily log returns are computed:
  [
  r_{i,t} = \ln\left(\frac{P_{i,t}}{P_{i,t-1}}\right)
  ]

* The annualized covariance matrix is estimated as:
  [
  \Sigma = 252 \cdot \text{Cov}(r)
  ]

* Portfolio volatility:
  [
  \sigma_p = \sqrt{\mathbf{w}^T \Sigma \mathbf{w}}
  ]

---

## Optimization Problem

The optimizer solves:

[
\min_{\mathbf{w}} ; \lambda , \sigma_p - \mathbf{w}^T \boldsymbol{\mu}
]

Subject to:
[
\sum_i w_i = 1
]
[
0 \le w_i \le w_{\max}
]

Where:

* ( \lambda ) = risk aversion parameter
* ( w_{\max} ) = maximum asset weight (default 30%)

Optimization is performed using **SLSQP**.

---

## Portfolio Metrics

Expected portfolio return:
[
E[R_p] = \mathbf{w}^T \boldsymbol{\mu}
]

Sharpe ratio:
[
S = \frac{E[R_p] - r_f}{\sigma_p}
]

Where ( r_f ) is the risk-free rate.

---

## Output

The optimizer returns:

* Asset-level optimized weights
* New cash allocation per asset
* Final post-rebalance weights
* Pre- and post-rebalance return, volatility, and Sharpe ratio

Results are exported to:

```text
df_rebalance.csv
```

---

## Notes

* Expected returns are **forward-looking** and target-driven
* No short selling or transaction costs
* Single-period optimization

---

## Disclaimer

This project is for educational and research purposes only and does not constitute financial advice.
