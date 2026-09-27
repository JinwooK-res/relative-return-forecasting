"""
Data ingestion and alignment module for research experiments.
Reuses FinanceDataReader while ensuring zero live-brokerage dependencies.
"""

from typing import List, Optional, Tuple
import pandas as pd
import FinanceDataReader as fdr

from research.settings import (
    BENCHMARK_SYMBOL,
    START_DATE,
    END_DATE,
    LOOKBACK_LAGS,
    FORWARD_HORIZON
)

# Minimum required consecutive trading days: at least max lag + forward horizon
MIN_REQUIRED_DAYS: int = max(LOOKBACK_LAGS) + FORWARD_HORIZON + 1  # 260 + 10 + 1 = 271


def get_sp500_tickers(top_n: Optional[int] = None) -> List[str]:
    """
    Fetch S&P 500 tickers using FinanceDataReader.
    Replaces '.' with '-' for ticker compatibility (e.g., BRK.B -> BRK-B).
    """
    listing = fdr.StockListing('S&P500')
    tickers = listing['Symbol'].tolist()
    sanitized = [t.replace('.', '-') for t in tickers]
    if top_n is not None and top_n > 0:
        return sanitized[:top_n]
    return sanitized


def fetch_historical_prices(
    symbol: str,
    start_date: str = START_DATE,
    end_date: str = END_DATE
) -> pd.DataFrame:
    """
    Fetch daily historical OHLCV data for a given symbol.
    Ensures index is DatetimeIndex and rows are sorted chronologically.
    """
    df = fdr.DataReader(symbol, start_date, end_date)
    if df is None or df.empty:
        raise ValueError(f"No price data returned for symbol '{symbol}' between {start_date} and {end_date}.")
    
    # Ensure index is standard datetime
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    # Remove duplicate date entries if any
    df = df[~df.index.duplicated(keep='first')]
    return df


def load_aligned_pair(
    symbol: str,
    benchmark_symbol: str = BENCHMARK_SYMBOL,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    benchmark_df: Optional[pd.DataFrame] = None
) -> Optional[pd.DataFrame]:
    """
    Fetch and synchronize daily closing prices of a stock and the S&P 500 benchmark.
    Performs an inner join on Date index to guarantee 100% calendar alignment.
    
    Optionally accepts a pre-fetched benchmark_df to avoid redundant API downloads.
    
    Returns a DataFrame with columns ['Close_stock', 'Close_bench'] and sorted Date index,
    or None if the stock has insufficient history.
    """
    try:
        df_stock = fetch_historical_prices(symbol, start_date, end_date)
        if benchmark_df is None:
            df_bench = fetch_historical_prices(benchmark_symbol, start_date, end_date)
        else:
            df_bench = benchmark_df

        if 'Close' not in df_stock.columns or 'Close' not in df_bench.columns:
            return None

        # Synchronize dates via inner join
        aligned = pd.DataFrame({
            'Close_stock': df_stock['Close'],
            'Close_bench': df_bench['Close']
        }).dropna()

        aligned = aligned.sort_index()

        # Check for minimum required trading history
        if len(aligned) < MIN_REQUIRED_DAYS:
            print(f"[Warning] Symbol '{symbol}' has only {len(aligned)} aligned trading days "
                  f"(minimum required: {MIN_REQUIRED_DAYS}). Skipping.")
            return None

        return aligned

    except Exception as e:
        print(f"[Error] Failed to load aligned pair for '{symbol}': {e}")
        return None
