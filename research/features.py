"""
Feature engineering and target definition module.
Strictly implements the methodology from Htun et al. (2024), Eq. 1, 2, and 3.

Features:
- Exactly 13 relative-return features comparing stock return vs S&P 500 index return:
  k in {260, 180, 150, 120, 100, 80, 60, 40, 20, 15, 10, 5, 1}
  X(i, k) = R(i, k) - R_SP(i, k)

Target:
- Binary label y(i) = 1 if max_{k=1..10} {X(i+k, k)} > 2.0%, else 0.
"""

from typing import List, Tuple, Optional
import numpy as np
import pandas as pd

from research.settings import (
    LOOKBACK_LAGS,
    FORWARD_HORIZON,
    RETURN_THRESHOLD
)


def compute_relative_return_features(
    df_aligned: pd.DataFrame,
    lags: List[int] = LOOKBACK_LAGS
) -> pd.DataFrame:
    """
    Compute the paper's 13 relative-return features for each trading day.
    
    Formula:
      R_stock(i, k) = (Close_stock[i] - Close_stock[i-k]) / Close_stock[i-k] * 100
      R_bench(i, k) = (Close_bench[i] - Close_bench[i-k]) / Close_bench[i-k] * 100
      X(i, k) = R_stock(i, k) - R_bench(i, k)
      
    Guaranteed: Strictly backward-looking (uses only past rows i-k up to current row i).
    """
    features_df = pd.DataFrame(index=df_aligned.index)

    close_stock = df_aligned['Close_stock']
    close_bench = df_aligned['Close_bench']

    for k in lags:
        # Percentage returns over lag k
        r_stock = (close_stock - close_stock.shift(k)) / close_stock.shift(k) * 100.0
        r_bench = (close_bench - close_bench.shift(k)) / close_bench.shift(k) * 100.0
        
        col_name = f"X_{k}"
        features_df[col_name] = r_stock - r_bench

    return features_df


def compute_forward_relative_target(
    df_aligned: pd.DataFrame,
    horizon: int = FORWARD_HORIZON,
    threshold: float = RETURN_THRESHOLD
) -> Tuple[pd.Series, pd.Series]:
    """
    Compute the forward 10-day relative return target (Eq. 3 in Htun et al., 2024).
    
    For each forward day k in {1, 2, ..., horizon}:
      R_stock(i+k, k) = (Close_stock[i+k] - Close_stock[i]) / Close_stock[i] * 100
      R_bench(i+k, k) = (Close_bench[i+k] - Close_bench[i]) / Close_bench[i] * 100
      X(i+k, k) = R_stock(i+k, k) - R_bench(i+k, k)
      
    max_rel_ret = max_{k=1..horizon} { X(i+k, k) }
    target = 1 if max_rel_ret > threshold else 0
    
    Returns:
      (target_series, max_rel_return_series)
      
    Note on look-ahead prevention:
      For the last `horizon` rows of the dataset, full 10-day future trajectory
      is unobserved. The target for these rows is set to NaN to prevent incomplete
      future leakage.
    """
    close_stock = df_aligned['Close_stock']
    close_bench = df_aligned['Close_bench']

    forward_rel_returns = pd.DataFrame(index=df_aligned.index)

    for k in range(1, horizon + 1):
        # Forward return from day i to day i+k
        fwd_stock_ret = (close_stock.shift(-k) - close_stock) / close_stock * 100.0
        fwd_bench_ret = (close_bench.shift(-k) - close_bench) / close_bench * 100.0
        forward_rel_returns[f"rel_ret_fwd_{k}"] = fwd_stock_ret - fwd_bench_ret

    # Any row that does not have all 10 forward days should be marked NaN
    valid_mask = forward_rel_returns.notna().all(axis=1)

    max_rel_ret = forward_rel_returns.max(axis=1)
    max_rel_ret[~valid_mask] = np.nan

    target = pd.Series(np.nan, index=df_aligned.index, name="target")
    target[valid_mask] = (max_rel_ret[valid_mask] > threshold).astype(int)

    return target, max_rel_ret


def compute_realized_holding_returns(
    df_aligned: pd.DataFrame,
    horizon: int = FORWARD_HORIZON
) -> pd.DataFrame:
    """
    Compute exact realized holding returns over the forward horizon (e.g. 10 trading days).
    Used solely for research backtesting evaluation (kept separate from features).
    
    Returns:
      DataFrame with ['fwd_ret_stock_10d', 'fwd_ret_bench_10d']
    """
    close_stock = df_aligned['Close_stock']
    close_bench = df_aligned['Close_bench']

    df_ret = pd.DataFrame(index=df_aligned.index)
    df_ret[f"fwd_ret_stock_{horizon}d"] = (close_stock.shift(-horizon) - close_stock) / close_stock * 100.0
    df_ret[f"fwd_ret_bench_{horizon}d"] = (close_bench.shift(-horizon) - close_bench) / close_bench * 100.0
    return df_ret


def build_research_dataset(
    df_aligned: pd.DataFrame,
    lags: List[int] = LOOKBACK_LAGS,
    horizon: int = FORWARD_HORIZON,
    threshold: float = RETURN_THRESHOLD
) -> pd.DataFrame:
    """
    Construct the complete dataset for a single stock:
    - Close prices: Close_stock, Close_bench
    - Features: X_260, ..., X_1 (13 paper features)
    - Target: target (binary 0/1, NaN for last horizon rows)
    - Auxiliary metadata for backtesting: max_rel_ret_10d, fwd_ret_stock_10d, fwd_ret_bench_10d
    """
    features = compute_relative_return_features(df_aligned, lags=lags)
    target, max_rel_ret = compute_forward_relative_target(df_aligned, horizon=horizon, threshold=threshold)
    fwd_rets = compute_realized_holding_returns(df_aligned, horizon=horizon)

    dataset = pd.concat([
        df_aligned[['Close_stock', 'Close_bench']],
        features,
        target,
        max_rel_ret.rename("max_rel_ret_10d"),
        fwd_rets
    ], axis=1)

    return dataset
