# Relative-Return Forecasting for S&P 500 Stocks

A paper-inspired machine-learning study of whether historical stock returns relative to the S&P 500 contain weak out-of-sample predictive information.

The research design follows the feature, target, and walk-forward framework of Htun, Biehl, and Petkov (2024), but it is **not an exact replication**. This project uses a 100-tree Random Forest rather than the paper's 1,000-tree specification, replaces the paper's SVM/LSTM comparisons with a LightGBM extension, and evaluates a current-at-collection constituent universe rather than historical point-in-time S&P 500 membership.

The portfolio backtest is an additional research extension and was not part of the original paper.

## Research question

For each stock, can 13 historical relative-return features help predict whether the stock will outperform the S&P 500 by more than 2% at any point during the next 10 trading days?

The project evaluates this as a stock-specific binary-classification problem using strictly out-of-sample walk-forward predictions.

## Methodology

### Relative-return features

For stock \(s\), date \(t\), and lookback \(k\):

\[
R_s(t,k)=100\left(\frac{P_s(t)-P_s(t-k)}{P_s(t-k)}\right)
\]

\[
X_s(t,k)=R_s(t,k)-R_{\mathrm{S\&P500}}(t,k)
\]

Each observation contains exactly 13 relative-return features:

\[
k \in \{260,180,150,120,100,80,60,40,20,15,10,5,1\}
\]

All features use only the current and preceding closing prices.

### Classification target

For each date \(t\), the target is:

\[
y_t =
\begin{cases}
1, & \max_{k=1,\ldots,10} X_s(t+k,k) > 2\% \\
0, & \text{otherwise}
\end{cases}
\]

A positive label means that the stock exceeds the S&P 500 return by more than 2% on at least one of the following 10 trading days. It does not necessarily imply positive absolute performance or outperformance exactly on day 10.

### Walk-forward validation

Each stock and window is modeled independently:

| Component | Length |
|---|---:|
| Training window | 253 trading days |
| Purge gap | 10 trading days |
| Test window | 21 trading days |
| Window shift | 21 trading days |
| Number of windows | 23 |
| OOS predictions per stock/model | 483 |

The 10-day gap prevents training labels, which depend on the following 10 trading days, from overlapping the test period. A new model is fitted for every stock and every window.

Most common-calendar OOS observations span February 2019 through December 2020. A small number of stocks have later ticker-relative evaluation calendars because their available histories begin later.

## Models

| Model | Role | Main configuration |
|---|---|---|
| Random Forest | Paper-inspired baseline | 100 trees, maximum depth 5, maximum features 13 |
| LightGBM | Project extension | 100 estimators, learning rate 0.05, maximum depth 5 |

Both models use a fixed random seed. No additional features, global preprocessing, or cross-stock training are used.

The Random Forest is intentionally described as paper-inspired: Htun et al. use 1,000 trees, while Final-A uses 100.

## Universe accounting

| Stage | Symbols | Description |
|---|---:|---|
| Candidate manifest | 503 | Current-at-collection S&P 500 symbols returned by the data source |
| Initially validated | 471 | Sufficient aligned history for the initial protocol checks |
| Classification universe | 470 | Complete 23-window, 483-row-per-model OOS coverage |
| Backtest formation universe | 467 | Stocks sharing the common formation calendar in every portfolio period |

VICI was excluded at the final classification stage because it could not produce complete 23-window OOS coverage.

CVNA, INVH, and IR remain part of the 470-stock classification analysis but have ticker-relative windows beginning on later dates. They are excluded from the 467-stock common-calendar cross-sectional formation universe rather than ranked against predictions from a different date.

Final-A contains:

- 227,010 OOS predictions per model;
- 454,020 OOS predictions in total;
- 21,620 stock/model/window fits; and
- 23 windows x 21 observations for every included stock and model.

Checkpoint validation requires the exact expected models, window IDs 0-22, 21 rows per window, unique ticker/date/model/window records, and complete finite values.

## Classification results

The following are pooled metrics across 227,010 OOS predictions per model:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.5205 | 0.5273 | 0.5314 | 0.5293 | 0.5290 |
| LightGBM | 0.5213 | 0.5284 | 0.5280 | 0.5282 | 0.5315 |

Both models exhibit weak discrimination above a 0.50 ROC-AUC reference. Their differences are small and metric-dependent: LightGBM has slightly higher pooled accuracy, precision, and ROC-AUC, while Random Forest has higher recall and F1.

These results do **not** establish general LightGBM superiority. Per-stock comparisons are mixed, and this project does not claim a statistically or economically decisive difference between the models.

Pooled ROC-AUC should also be interpreted cautiously because it combines probability estimates produced by independently fitted stock- and window-specific models.

## Top-10 backtest extension

The economic extension uses the first common-calendar date of each 21-day test window, ranks the 467 eligible stocks by OOS predicted probability, selects an equal-weighted Top-10 portfolio, and measures its close-to-close return over the following 10 trading days. The same dates are used for the S&P 500 comparison. The 23 holding periods do not overlap.

### Important execution assumption

This is an **idealized same-close research backtest**.

The formation-day closing price contributes to the features, and that same close is used as the return measurement entry price. A live system ordinarily cannot observe the official close, compute the signal, and then guarantee execution at that price.

The results are useful for methodological analysis but are **not directly live-executable**. No next-session-entry version is substituted for the reported experiment.

### Backtest results

| Strategy | Cumulative return | Mean 10-day return | Annualized return* | Sharpe* | Max drawdown* | Cumulative difference vs S&P 500 |
|---|---:|---:|---:|---:|---:|---:|
| LightGBM Top-10 | +27.40% | +1.67% | +13.47% | 0.567 | -38.70% | -1.72 pp |
| Random Forest Top-10 | +27.16% | +1.58% | +13.36% | 0.571 | -36.30% | -1.96 pp |
| Matched-horizon S&P 500 | +29.12% | +1.34% | +14.26% | 0.713 | -23.39% | - |

\*Annualized statistics and drawdown are computed from 23 discrete, non-overlapping 10-day holding-period observations. They are illustrative research statistics rather than conventional daily live-strategy estimates.

Both model portfolios have positive arithmetic mean period excess returns, but their compounded returns finish below the benchmark. This is mathematically consistent: the portfolios experience greater volatility and more severe negative periods, creating volatility drag.

The backtest does **not** demonstrate investable alpha. Before transaction costs, both model portfolios trail the matched-horizon S&P 500 in cumulative return and Sharpe ratio and experience materially deeper drawdowns.

## Interpretation

The evidence supports a narrow conclusion:

> Historical relative returns contain a weak out-of-sample classification signal in this experimental design, but the signal does not translate into robust investable performance in the idealized Top-10 backtest.

The results do not support claims that:

- LightGBM conclusively outperforms Random Forest;
- the ML portfolios outperform the S&P 500;
- the experiment establishes tradable alpha;
- the strategy is ready for live deployment; or
- this repository exactly replicates Htun et al. (2024).

## Leakage and integrity controls

- Strictly backward-looking feature construction
- Separate target and feature columns
- Ten-trading-day purge gap
- Fresh stock-specific and window-specific models
- No global scaler or preprocessing fitted on test data
- OOS probabilities generated only after fitting the corresponding training window
- No realized future-return columns used in portfolio ranking
- Exact stock/benchmark date alignment
- Strict checkpoint completeness and duplicate checks
- Atomic checkpoint writes and explicit final-stage exclusions
- Common-calendar backtest formation checks

The 470 completed checkpoints can be reused without recomputing their models when they are available locally.

## Limitations

### Current-constituent and survivorship bias

The candidate list reflects S&P 500 membership when the data were collected, not historical point-in-time membership during the evaluation period. Later entrants may be included retrospectively, while firms removed before collection may be absent. The results therefore do not represent a point-in-time historical S&P 500 simulation. This bias is likely optimistic, although its magnitude is not estimated.

### Symbol and historical-data quality

Ticker changes, mergers, spin-offs, reorganizations, and data-provider symbol mapping may affect historical interpretation. Current tickers can be associated with predecessor histories, while certain share-class symbols may fail automated normalization. The project uses Yahoo-sourced closing-price data through FinanceDataReader, and historical vendor revisions may affect reproducibility.

### Corporate actions and dividends

Saved prices are continuous across major stock splits, but the study does not construct explicit dividend-inclusive total-return series. Stock and benchmark results should be interpreted as price-return results rather than complete investor total returns.

### Idealized execution and omitted costs

The same-close assumption is not directly executable after observing the official close. The backtest also omits commissions, bid-ask spreads, slippage, market impact, taxes, execution failures, and cash returns between holding periods. These omissions are directionally favorable to the reported strategies.

### Turnover and concentration

The Top-10 portfolios allocate 10% to each selected stock. Average one-way turnover is approximately 92% per monthly rebalance, so realistic trading costs could be material.

### Probability comparability

Portfolio ranking compares probabilities generated by independently fitted stock-specific models. These probabilities are not explicitly calibrated across stocks or windows.

### Metric interpretation

Headline classification metrics are pooled across all OOS observations. Pooled precision, recall, and ROC-AUC are not identical to averages of per-stock metrics. ROC-AUC for an individual single-class window is undefined and is not averaged into the pooled headline result.

### Backtest sample size and regime dependence

The backtest contains only 23 formation periods and is heavily influenced by the 2019-2020 market regime, including the COVID-19 shock. Annualized return, Sharpe ratio, and maximum drawdown therefore have substantial sampling uncertainty. Maximum drawdown is measured at holding-period endpoints and does not capture intraperiod paths.

### Model and parameter selection

The 2% threshold follows the paper's class-balance motivation, while LightGBM and the Top-10 portfolio are project extensions. Development and intermediate work create potential parameter-selection and data-snooping risk even though Final-A uses fixed parameters.

### No forward/live validation

All results are historical. The project contains no prospective paper-trading or live-trading validation of this research strategy.

## Repository structure

```text
research/
|-- settings.py
|-- data.py
|-- features.py
|-- models.py
|-- validation.py
|-- backtest.py
|-- run_final_a_full_universe.py
`-- output/
    |-- tickers_final_a.json
    |-- final_a_excluded.json
    |-- summary_metrics_final_a.json
    `-- backtest_final_a_periodic.csv

tests/
`-- test_research_audit_regressions.py
```

## Installation

Python 3.13 was used for the final audited environment.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Run the targeted audit regression checks from the repository root:

```bash
python tests/test_research_audit_regressions.py
```

The full-universe runner is computationally expensive and downloads historical market data. It is not required to inspect the included compact results.

## Reproducibility and artifact policy

This repository includes only the compact manifest, exclusion record, summary metrics, and periodic backtest results.

The complete prediction file is approximately 61 MB, and the checkpoint directory contains 470 per-ticker CSV files. These large raw artifacts are intentionally excluded from the public repository. They may instead be regenerated, stored as release assets, archived in an external research-data repository, or distributed on request with checksums and environment metadata.

## Reference

Htun, H. H., Biehl, M., & Petkov, N. (2024). Forecasting relative returns for S&P 500 stocks using machine learning. *Financial Innovation, 10*, 118. [https://doi.org/10.1186/s40854-024-00644-0](https://doi.org/10.1186/s40854-024-00644-0)

## License

Source code is released under the MIT License. The cited paper and external market data remain subject to their respective licenses and terms.

## Disclaimer

This repository is an academic research project. It does not provide investment advice, and its historical results should not be interpreted as evidence of future performance or suitability for live trading.
