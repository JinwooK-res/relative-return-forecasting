"""
Sliding-window walk-forward validation and evaluation module.
Implements the exact sliding-window protocol from Htun et al. (2024), Fig. 1:
- Train window: 253 consecutive trading days (~12 months)
- Gap window: 10 consecutive trading days (strictly excluded to prevent leakage)
- Test window: 21 consecutive trading days (~1 month)
- Step size: 21 consecutive trading days (shifts by 1 month)
- Num windows: 23 windows (or maximum available)

Ensures strictly identical features, targets, and windows across all models.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from research.settings import (
    TRAIN_WINDOW,
    GAP_WINDOW,
    TEST_WINDOW,
    STEP_SIZE,
    NUM_WINDOWS,
    LOOKBACK_LAGS
)
from research.models import get_random_forest_baseline, get_lightgbm_extension

MODEL_NAMES = ('RandomForest_Baseline', 'LightGBM_Extension')


def generate_sliding_windows(
    total_valid_days: int,
    train_size: int = TRAIN_WINDOW,
    gap_size: int = GAP_WINDOW,
    test_size: int = TEST_WINDOW,
    step_size: int = STEP_SIZE,
    max_windows: Optional[int] = NUM_WINDOWS
) -> List[Dict[str, int]]:
    """
    Generate the index boundaries for sliding windows.
    
    Each window contains:
      - Train: [t_start, t_end) with length train_size (253)
      - Gap:   [g_start, g_end) with length gap_size (10)  <- EXCLUDED
      - Test:  [te_start, te_end) with length test_size (21)
      
    Shifts forward by step_size (21) trading days per window.
    """
    windows = []
    w_idx = 0

    while True:
        t_start = w_idx * step_size
        t_end = t_start + train_size
        g_start = t_end
        g_end = g_start + gap_size
        te_start = g_end
        te_end = te_start + test_size

        if te_end > total_valid_days:
            break

        windows.append({
            'window_id': w_idx,
            'train_start': t_start,
            'train_end': t_end,
            'gap_start': g_start,
            'gap_end': g_end,
            'test_start': te_start,
            'test_end': te_end
        })

        w_idx += 1
        if max_windows is not None and w_idx >= max_windows:
            break

    return windows


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Calculate ML classification performance metrics.
    
    Paper's primary metrics:
      - Accuracy
      - Precision
      - Recall
      
    Additional extension metrics (clearly labeled):
      - F1 Score
      - ROC-AUC (safe handling when single class present in test window)
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    
    # Extension metrics beyond paper's main table:
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Safely compute ROC-AUC (undefined if only one class exists in y_true)
    auc = np.nan
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = np.nan

    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_extension': f1,
        'roc_auc_extension': auc,
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn),
        'total_samples': len(y_true)
    }


def evaluate_stock_walk_forward(
    ticker: str,
    dataset: pd.DataFrame,
    model_name: str,
    model_instance_fn,
    max_windows: Optional[int] = NUM_WINDOWS
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run sliding-window walk-forward evaluation for a single stock with a specific model.
    
    Returns:
      (predictions_df, window_metrics_df)
    """
    feature_cols = [f"X_{k}" for k in LOOKBACK_LAGS]

    # Filter rows that have valid features (drops the initial 260 NaN lag days)
    valid_df = dataset.dropna(subset=feature_cols).copy()
    
    # Generate sliding window indices
    windows = generate_sliding_windows(
        total_valid_days=len(valid_df),
        train_size=TRAIN_WINDOW,
        gap_size=GAP_WINDOW,
        test_size=TEST_WINDOW,
        step_size=STEP_SIZE,
        max_windows=max_windows
    )

    pred_records = []
    window_metrics_list = []

    for win in windows:
        w_id = win['window_id']

        # Slice train, gap, test blocks
        train_block = valid_df.iloc[win['train_start']:win['train_end']]
        test_block = valid_df.iloc[win['test_start']:win['test_end']]

        # Drop any row inside train/test missing target (e.g. at the extreme tail)
        train_clean = train_block.dropna(subset=['target'])
        test_clean = test_block.dropna(subset=['target'])

        if len(train_clean) < TRAIN_WINDOW - 20 or len(test_clean) == 0:
            # Skip if insufficient clean labels
            continue

        X_train = train_clean[feature_cols].values
        y_train = train_clean['target'].values.astype(int)

        X_test = test_clean[feature_cols].values
        y_test = test_clean['target'].values.astype(int)

        # Instantiate fresh model for this window (stock-specific & window-specific)
        model = model_instance_fn()
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Calculate per-window metrics
        win_metrics = compute_classification_metrics(y_test, y_pred, y_prob)
        win_metrics['ticker'] = ticker
        win_metrics['model'] = model_name
        win_metrics['window_id'] = w_id
        win_metrics['train_start_date'] = train_block.index[0].strftime('%Y-%m-%d')
        win_metrics['train_end_date'] = train_block.index[-1].strftime('%Y-%m-%d')
        win_metrics['gap_start_date'] = valid_df.index[win['gap_start']].strftime('%Y-%m-%d')
        win_metrics['gap_end_date'] = valid_df.index[win['gap_end'] - 1].strftime('%Y-%m-%d')
        win_metrics['test_start_date'] = test_block.index[0].strftime('%Y-%m-%d')
        win_metrics['test_end_date'] = test_block.index[-1].strftime('%Y-%m-%d')
        window_metrics_list.append(win_metrics)

        # Record out-of-sample predictions
        for dt, yt, yp, prob, c_stock, c_bench, ret_s, ret_b in zip(
            test_clean.index,
            y_test,
            y_pred,
            y_prob,
            test_clean['Close_stock'],
            test_clean['Close_bench'],
            test_clean.get('fwd_ret_stock_10d', [np.nan] * len(test_clean)),
            test_clean.get('fwd_ret_bench_10d', [np.nan] * len(test_clean))
        ):
            pred_records.append({
                'date': dt.strftime('%Y-%m-%d'),
                'ticker': ticker,
                'window_id': w_id,
                'model': model_name,
                'y_true': int(yt),
                'y_pred': int(yp),
                'y_prob': float(prob),
                'close_stock': float(c_stock),
                'close_bench': float(c_bench),
                'fwd_ret_stock_10d': float(ret_s) if pd.notna(ret_s) else np.nan,
                'fwd_ret_bench_10d': float(ret_b) if pd.notna(ret_b) else np.nan
            })

    preds_df = pd.DataFrame(pred_records)
    window_df = pd.DataFrame(window_metrics_list)
    return preds_df, window_df


def _evaluate_single_ticker_all_models(
    ticker: str,
    ds: pd.DataFrame,
    rf_trees: int,
    max_windows: Optional[int]
) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    """Helper to evaluate all models for a single stock."""
    preds_list = []
    wins_list = []
    
    models = {
        'RandomForest_Baseline': lambda: get_random_forest_baseline(n_estimators=rf_trees),
        'LightGBM_Extension': lambda: get_lightgbm_extension()
    }
    
    for m_name, m_fn in models.items():
        preds_df, win_df = evaluate_stock_walk_forward(
            ticker=ticker,
            dataset=ds,
            model_name=m_name,
            model_instance_fn=m_fn,
            max_windows=max_windows
        )
        preds_list.append(preds_df)
        wins_list.append(win_df)
        
    return preds_list, wins_list


def run_experiment_pipeline(
    datasets_by_ticker: Dict[str, pd.DataFrame],
    rf_trees: int = 100,
    max_windows: Optional[int] = NUM_WINDOWS,
    n_jobs: int = 1
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run both Experiment 1 (Random Forest Baseline) and Experiment 2 (LightGBM Extension)
    across all provided stock datasets on identical sliding windows.
    
    Supports parallel multi-core processing across tickers via joblib if n_jobs != 1.
    
    Returns:
      (all_predictions_df, window_metrics_df, aggregate_metrics_df)
    """
    all_preds_list = []
    all_window_metrics_list = []

    if n_jobs != 1:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(_evaluate_single_ticker_all_models)(
                ticker, ds, rf_trees, max_windows
            ) for ticker, ds in datasets_by_ticker.items()
        )
        for p_list, w_list in results:
            all_preds_list.extend(p_list)
            all_window_metrics_list.extend(w_list)
    else:
        for ticker, ds in datasets_by_ticker.items():
            p_list, w_list = _evaluate_single_ticker_all_models(ticker, ds, rf_trees, max_windows)
            all_preds_list.extend(p_list)
            all_window_metrics_list.extend(w_list)

    all_preds = pd.concat(all_preds_list, ignore_index=True)
    all_window_metrics = pd.concat(all_window_metrics_list, ignore_index=True)

    # Compute aggregate out-of-sample metrics for each model
    agg_list = []
    for m_name in MODEL_NAMES:
        sub = all_preds[all_preds['model'] == m_name]
        if len(sub) == 0:
            continue
        agg = compute_classification_metrics(sub['y_true'].values, sub['y_pred'].values, sub['y_prob'].values)
        agg['model'] = m_name
        agg['num_tickers'] = len(sub['ticker'].unique())
        agg['total_predictions'] = len(sub)
        agg_list.append(agg)

    agg_df = pd.DataFrame(agg_list)

    # Reorder columns logically
    first_cols = ['model', 'num_tickers', 'total_predictions', 'accuracy', 'precision', 'recall', 'f1_extension', 'roc_auc_extension']
    other_cols = [c for c in agg_df.columns if c not in first_cols]
    agg_df = agg_df[first_cols + other_cols]

    return all_preds, all_window_metrics, agg_df
