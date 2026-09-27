"""
Research configuration settings for S&P 500 relative return forecasting project.
Inspired by Htun et al. (2024), Financial Innovation.

All research-specific parameters are isolated here to keep live-trading config.py untouched.
"""

from typing import List

# Universe & Benchmark
BENCHMARK_SYMBOL: str = "^GSPC"  # S&P 500 Index
UNIVERSE_NAME: str = "S&P500"

# Quick development tickers for sanity checks and fast iteration
DEV_TICKERS: List[str] = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
DEV_TOP_N: int = 10
FULL_TOP_N: int = 494

# Historical Data Range (Matches paper: 2017 to 2022)
# 2017 is used to compute initial 260-day lag features for 2018 training start
START_DATE: str = "2017-01-01"
END_DATE: str = "2022-01-01"

# Feature Definition (Exact 13 lookback periods from paper Eq. 1 & 2)
# k = 260, 180, 150, 120, 100, 80, 60, 40, 20, 15, 10, 5, 1
LOOKBACK_LAGS: List[int] = [260, 180, 150, 120, 100, 80, 60, 40, 20, 15, 10, 5, 1]

# Prediction Target Definition (Paper Eq. 3)
# Predict whether max relative return over the next 10 trading days exceeds 2.0%
FORWARD_HORIZON: int = 10       # 10 trading days
RETURN_THRESHOLD: float = 2.0   # +2.0% outperformance vs S&P 500

# Sliding-Window Walk-Forward Validation Protocol (Paper Fig. 1)
TRAIN_WINDOW: int = 253         # ~12 calendar months (253 trading days)
GAP_WINDOW: int = 10            # 10 trading days gap to prevent look-ahead leakage
TEST_WINDOW: int = 21           # ~1 calendar month (21 trading days)
STEP_SIZE: int = 21             # Shift window by 1 month (21 trading days)
NUM_WINDOWS: int = 23           # 23 sliding windows as in paper

# Model Hyperparameters
# Paper RF baseline: 1000 trees, max_depth=5, max_features=13
RF_PAPER_TREES: int = 1000
RF_DEV_TREES: int = 100         # For rapid development / verification
RF_MAX_DEPTH: int = 5
RF_MAX_FEATURES: int = 13
RANDOM_STATE: int = 42

# LightGBM Extension Model Parameters
LGBM_DEV_ESTIMATORS: int = 100
LGBM_LEARNING_RATE: float = 0.05
LGBM_MAX_DEPTH: int = 5

# Backtest Extension Settings
# Final-A's pre-selected portfolio size. Shared by the backtest helpers and runner
# so their defaults cannot silently diverge from the reported methodology.
BACKTEST_TOP_K: int = 10        # Equal-weight top K stocks by probability
