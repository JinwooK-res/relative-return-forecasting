"""
Research backtesting module (Extension beyond Htun et al., 2024).

NOTE ON METHODOLOGY:
The original paper (Htun et al., 2024) evaluated machine learning models purely
on classification metrics (Accuracy, Precision, Recall) and did NOT implement a
financial portfolio backtest or compute financial risk-adjusted metrics.
All portfolio construction and backtesting logic implemented here is an
explicit research extension to evaluate the economic utility of model scores.

Design:
1. Signal Timing:
    - Portfolios are formed ONLY on the first trading day of each 21-day test window.
    - Non-overlapping: Holding period is 10 trading days, rebalancing occurs every 21 days.
    - Formation uses the modal common-calendar date across ticker-local windows; tickers
      whose same window_id begins on a different date are disclosed and excluded.
    - Execution is an idealized same-close research assumption: the formation-day close
      is used both in the signal features and as the return measurement entry price.
2. Portfolio Construction:
   - Rank stocks descending by out-of-sample predicted probability of outperformance (y_prob).
   - Select Top-K stocks with strictly equal weighting (1/K).
3. Holding Period & Realized Return:
   - 10 trading days holding.
   - Portfolio return = mean(fwd_ret_stock_10d) of selected Top-K.
   - Benchmark return = fwd_ret_bench_10d (S&P 500 over the exact same 10-day period).
4. Models:
   - Evaluates Random Forest Baseline and LightGBM Extension separately.
5. Metrics:
   - Cumulative return, mean 10-day return, annualized return, Sharpe ratio,
     maximum drawdown, and excess return vs S&P 500.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd

from research.settings import BACKTEST_TOP_K


IDEALIZED_SAME_CLOSE_EXECUTION = (
    "Idealized same-close research backtest; not a live-executable fill assumption."
)


def extract_formation_date_predictions(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract ticker-local first-day OOS predictions on the modal common-calendar date.

    A window_id is positional within each ticker and therefore does not by itself prove
    that all tickers share a calendar date. This function first finds each ticker's local
    formation date, then retains the date shared by the largest number of tickers. It
    records analyzed and eligible universe counts so later reporting cannot imply that
    every analyzed ticker was available at every portfolio formation.
    """
    formation_records = []

    local_first_indices = predictions_df.groupby(
        ['model', 'window_id', 'ticker']
    )['date'].idxmin()
    local_first = predictions_df.loc[local_first_indices].copy()

    for (model_name, window_id), group in local_first.groupby(['model', 'window_id']):
        date_counts = group['date'].value_counts()
        max_count = int(date_counts.max())
        common_date = sorted(date_counts[date_counts == max_count].index)[0]
        common_slice = group[group['date'] == common_date].copy()

        analyzed_tickers = sorted(group['ticker'].astype(str).unique())
        eligible_tickers = set(common_slice['ticker'].astype(str))
        excluded_tickers = [t for t in analyzed_tickers if t not in eligible_tickers]

        common_slice['analyzed_universe_size'] = len(analyzed_tickers)
        common_slice['formation_universe_size'] = len(eligible_tickers)
        common_slice['excluded_non_common_calendar_count'] = len(excluded_tickers)
        common_slice['excluded_non_common_calendar_tickers'] = [
            excluded_tickers.copy() for _ in range(len(common_slice))
        ]
        common_slice['execution_assumption'] = IDEALIZED_SAME_CLOSE_EXECUTION
        formation_records.append(common_slice)

    if not formation_records:
        return pd.DataFrame()

    formation_df = pd.concat(formation_records, ignore_index=True)
    return formation_df.sort_values(by=['model', 'window_id', 'date'])


def construct_top_k_portfolio(
    formation_df: pd.DataFrame,
    top_k: int = BACKTEST_TOP_K
) -> pd.DataFrame:
    """
    Select Top-K stocks for each formation date ranked by out-of-sample predicted probability (y_prob).
    Assigns strictly equal weight (1 / K) to each selected stock.
    """
    # Sort descending by y_prob within each (model, window_id)
    sorted_df = formation_df.sort_values(
        by=['model', 'window_id', 'y_prob'],
        ascending=[True, True, False]
    )

    # Take Top-K
    top_k_df = sorted_df.groupby(['model', 'window_id']).head(top_k).copy()
    top_k_df['weight'] = 1.0 / top_k
    return top_k_df


def calculate_periodic_portfolio_returns(top_k_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the realized 10-day portfolio return for each formation date.
    
    Returns:
      DataFrame with columns:
      ['model', 'window_id', 'date', 'selected_tickers', 'selected_probs',
       'selected_stock_returns', 'portfolio_return', 'benchmark_return', 'excess_return',
       'analyzed_universe_size', 'formation_universe_size',
       'excluded_non_common_calendar_tickers', 'execution_assumption']
    """
    periodic_rows = []

    for (model_name, window_id, date), group in top_k_df.groupby(['model', 'window_id', 'date']):
        tickers = group['ticker'].tolist()
        probs = group['y_prob'].round(4).tolist()
        stock_rets = group['fwd_ret_stock_10d'].round(4).tolist()

        # Equal-weighted portfolio return
        port_ret = float(np.mean(group['fwd_ret_stock_10d']))
        # Benchmark return (identical for all rows on the same date)
        bench_ret = float(group['fwd_ret_bench_10d'].iloc[0])
        excess_ret = port_ret - bench_ret

        analyzed_universe_size = int(group['analyzed_universe_size'].iloc[0])
        formation_universe_size = int(group['formation_universe_size'].iloc[0])
        excluded_tickers = group['excluded_non_common_calendar_tickers'].iloc[0]

        periodic_rows.append({
            'model': model_name,
            'window_id': window_id,
            'date': date,
            'selected_tickers': tickers,
            'selected_probs': probs,
            'selected_stock_returns': stock_rets,
            'portfolio_return': port_ret,
            'benchmark_return': bench_ret,
            'excess_return': excess_ret,
            'analyzed_universe_size': analyzed_universe_size,
            'formation_universe_size': formation_universe_size,
            'excluded_non_common_calendar_count': analyzed_universe_size - formation_universe_size,
            'excluded_non_common_calendar_tickers': excluded_tickers,
            'execution_assumption': IDEALIZED_SAME_CLOSE_EXECUTION
        })

    periodic_df = pd.DataFrame(periodic_rows)
    return periodic_df.sort_values(by=['model', 'window_id']).reset_index(drop=True)


def compute_financial_metrics(
    returns_percent: np.ndarray,
    periods_per_year: float = 12.0
) -> Dict[str, float]:
    """
    Compute standard financial performance and risk metrics on a series of periodic percentage returns.
    
    Parameters:
      returns_percent: Array of periodic returns in percent (e.g., [2.5, -1.0, 3.2])
      periods_per_year: 12.0 for monthly rebalanced windows (252 / 21 = 12 periods/year)
    """
    r = np.asarray(returns_percent, dtype=float) / 100.0
    n = len(r)

    if n == 0:
        return {
            'cumulative_return': 0.0,
            'mean_period_return': 0.0,
            'annualized_return': 0.0,
            'annualized_volatility': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0
        }

    # 1. Cumulative Return: prod(1 + r) - 1
    wealth = np.cumprod(1.0 + r)
    cum_ret = (wealth[-1] - 1.0) * 100.0

    # 2. Mean Period Return
    mean_period_ret = float(np.mean(returns_percent))

    # 3. Annualized Return: (1 + cum_ret) ^ (periods_per_year / n) - 1
    years = n / periods_per_year
    if years > 0 and wealth[-1] > 0:
        ann_ret = ((wealth[-1]) ** (1.0 / years) - 1.0) * 100.0
    else:
        ann_ret = np.nan

    # 4. Annualized Volatility
    period_vol = float(np.std(r, ddof=1)) if n > 1 else 0.0
    ann_vol = (period_vol * np.sqrt(periods_per_year)) * 100.0

    # 5. Annualized Sharpe Ratio (assuming risk-free rate = 0.0)
    if period_vol > 0:
        sharpe = (float(np.mean(r)) / period_vol) * np.sqrt(periods_per_year)
    else:
        sharpe = 0.0

    # 6. Maximum Drawdown
    wealth_series = np.insert(wealth, 0, 1.0)
    peaks = np.maximum.accumulate(wealth_series)
    drawdowns = (wealth_series - peaks) / peaks
    max_dd = float(np.min(drawdowns)) * 100.0

    return {
        'cumulative_return': cum_ret,
        'mean_period_return': mean_period_ret,
        'annualized_return': ann_ret,
        'annualized_volatility': ann_vol,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd
    }


def run_research_backtest(
    predictions_df: pd.DataFrame,
    top_k: int = BACKTEST_TOP_K
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute the research backtest across models present in predictions_df.
    
    Returns:
      (periodic_results_df, summary_metrics_df)
    """
    # 1. Extract first trading day of each test window
    formation_df = extract_formation_date_predictions(predictions_df)

    # 2. Form Top-K portfolios
    top_k_df = construct_top_k_portfolio(formation_df, top_k=top_k)

    # 3. Compute periodic returns
    periodic_df = calculate_periodic_portfolio_returns(top_k_df)

    # 4. Calculate summary performance metrics per model and for S&P 500 benchmark
    summary_rows = []
    models = periodic_df['model'].unique()

    for m in models:
        sub = periodic_df[periodic_df['model'] == m]
        port_metrics = compute_financial_metrics(sub['portfolio_return'].values)
        bench_metrics = compute_financial_metrics(sub['benchmark_return'].values)

        mean_excess = float(np.mean(sub['excess_return']))
        cum_excess = port_metrics['cumulative_return'] - bench_metrics['cumulative_return']

        summary_rows.append({
            'strategy': f"{m} (Top-{top_k})",
            'num_windows': len(sub),
            'analyzed_tickers_count': int(sub['analyzed_universe_size'].max()),
            'formation_universe_size_min': int(sub['formation_universe_size'].min()),
            'formation_universe_size_max': int(sub['formation_universe_size'].max()),
            'excluded_non_common_calendar_tickers': sorted({
                ticker
                for tickers in sub['excluded_non_common_calendar_tickers']
                for ticker in tickers
            }),
            'execution_assumption': IDEALIZED_SAME_CLOSE_EXECUTION,
            'cumulative_return': port_metrics['cumulative_return'],
            'mean_10d_return': port_metrics['mean_period_return'],
            'annualized_return': port_metrics['annualized_return'],
            'sharpe_ratio': port_metrics['sharpe_ratio'],
            'max_drawdown': port_metrics['max_drawdown'],
            'cumulative_excess_vs_sp500': cum_excess,
            'mean_excess_vs_sp500': mean_excess
        })

    # Add S&P 500 Benchmark as a reference row
    if len(models) > 0:
        first_model_sub = periodic_df[periodic_df['model'] == models[0]]
        bench_metrics = compute_financial_metrics(first_model_sub['benchmark_return'].values)
        summary_rows.append({
            'strategy': "Matched-Horizon S&P 500 Benchmark (^GSPC)",
            'num_windows': len(first_model_sub),
            'analyzed_tickers_count': int(first_model_sub['analyzed_universe_size'].max()),
            'formation_universe_size_min': int(first_model_sub['formation_universe_size'].min()),
            'formation_universe_size_max': int(first_model_sub['formation_universe_size'].max()),
            'excluded_non_common_calendar_tickers': sorted({
                ticker
                for tickers in first_model_sub['excluded_non_common_calendar_tickers']
                for ticker in tickers
            }),
            'execution_assumption': IDEALIZED_SAME_CLOSE_EXECUTION,
            'cumulative_return': bench_metrics['cumulative_return'],
            'mean_10d_return': bench_metrics['mean_period_return'],
            'annualized_return': bench_metrics['annualized_return'],
            'sharpe_ratio': bench_metrics['sharpe_ratio'],
            'max_drawdown': bench_metrics['max_drawdown'],
            'cumulative_excess_vs_sp500': 0.0,
            'mean_excess_vs_sp500': 0.0
        })

    summary_df = pd.DataFrame(summary_rows)
    return periodic_df, summary_df
