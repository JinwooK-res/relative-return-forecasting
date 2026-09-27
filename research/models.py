"""
Machine Learning models module for S&P 500 relative return forecasting.

Models:
1. Random Forest Baseline:
   Directly reproduces the paper's baseline model specification (Htun et al., 2024):
   - n_estimators: configurable (100 for fast dev, 1000 for final paper setting)
   - max_depth: 5
   - max_features: 13 (or all 13 features)
   - bootstrap: True
   - random_state: fixed for reproducibility

2. LightGBM Extension:
   Our model extension for comparison on the exact same dataset & protocol:
   - n_estimators: 100
   - max_depth: 5
   - learning_rate: 0.05
   - random_state: fixed for reproducibility
   - verbose: -1
"""

from typing import Optional
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

from research.settings import (
    RF_DEV_TREES,
    RF_PAPER_TREES,
    RF_MAX_DEPTH,
    RF_MAX_FEATURES,
    LGBM_DEV_ESTIMATORS,
    LGBM_LEARNING_RATE,
    LGBM_MAX_DEPTH,
    RANDOM_STATE
)


def get_random_forest_baseline(
    n_estimators: Optional[int] = None,
    max_depth: int = RF_MAX_DEPTH,
    max_features: int = RF_MAX_FEATURES,
    random_state: int = RANDOM_STATE
) -> RandomForestClassifier:
    """
    Factory function for the paper-inspired Random Forest baseline classifier.
    
    Defaults to RF_DEV_TREES (100) for development/testing if n_estimators is not specified.
    For full paper replication, pass n_estimators=RF_PAPER_TREES (1000).
    """
    if n_estimators is None:
        n_estimators = RF_DEV_TREES

    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        bootstrap=True,
        random_state=random_state,
        n_jobs=-1  # Multi-threaded execution for speed
    )


def get_lightgbm_extension(
    n_estimators: int = LGBM_DEV_ESTIMATORS,
    max_depth: int = LGBM_MAX_DEPTH,
    learning_rate: float = LGBM_LEARNING_RATE,
    random_state: int = RANDOM_STATE
) -> lgb.LGBMClassifier:
    """
    Factory function for the LightGBM model extension.
    Keeps hyperparameters simple and fixed for transparent comparison against the RF baseline.
    """
    return lgb.LGBMClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        verbose=-1,
        n_jobs=-1
    )
