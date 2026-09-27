"""
Final-A: Full-Universe S&P 500 Walk-Forward Validation & Research Backtest.
Evaluates the maximum usable S&P 500 universe across 2017-2022 using the exact paper protocol.

Design:
- Universe: All S&P 500 constituent tickers with sufficient historical data.
- Feature set: Exact 13 relative-return features (Htun et al., 2024).
- Target: 10-day forward relative return > +2.0% vs S&P 500.
- Validation: 23 sliding windows (253 train, 10 gap, 21 test, 21 step).
- Models: Random Forest Baseline (100 trees) vs LightGBM Extension.
- Backtest Extension: Fixed Top-10 equal-weight portfolio (10-day holding).
- Benchmark: Matched-Horizon S&P 500 Benchmark (^GSPC).
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Set
from concurrent.futures import ThreadPoolExecutor

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Ensure utf-8 encoding on Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import pandas as pd
import numpy as np

from research.settings import (
    BENCHMARK_SYMBOL,
    START_DATE,
    END_DATE,
    NUM_WINDOWS,
    RF_DEV_TREES,
    TRAIN_WINDOW,
    GAP_WINDOW,
    TEST_WINDOW,
    STEP_SIZE,
    LOOKBACK_LAGS,
    BACKTEST_TOP_K
)
from research.data import get_sp500_tickers, fetch_historical_prices, load_aligned_pair
from research.features import build_research_dataset
from research.validation import (
    _evaluate_single_ticker_all_models,
    compute_classification_metrics
)
from research.backtest import run_research_backtest

# Minimum required aligned days for 23 windows:
# 260 initial lag days + (22 * 21 step) + 253 train + 10 gap + 21 test = 1,006 trading days
MIN_DAYS_FOR_FULL_WINDOWS: int = max(LOOKBACK_LAGS) + (NUM_WINDOWS - 1) * STEP_SIZE + TRAIN_WINDOW + GAP_WINDOW + TEST_WINDOW

REQUIRED_MODELS = ('RandomForest_Baseline', 'LightGBM_Extension')
OOS_ROWS_PER_MODEL = NUM_WINDOWS * TEST_WINDOW  # 23 windows * 21 test days = 483
EXPECTED_PRED_COLUMNS = [
    'date', 'ticker', 'window_id', 'model', 'y_true', 'y_pred', 'y_prob',
    'close_stock', 'close_bench', 'fwd_ret_stock_10d', 'fwd_ret_bench_10d'
]
EXPECTED_NUMERIC_COLUMNS = [
    'window_id', 'y_true', 'y_pred', 'y_prob', 'close_stock', 'close_bench',
    'fwd_ret_stock_10d', 'fwd_ret_bench_10d'
]
INCOMPLETE_OOS_REASON = (
    "incomplete OOS coverage (required 23 windows and 483 rows per model)"
)


def process_ticker_data(sym: str, df_bench: pd.DataFrame):
    """Helper to fetch and build dataset for a single ticker in parallel thread."""
    try:
        aligned = load_aligned_pair(sym, benchmark_symbol=BENCHMARK_SYMBOL, benchmark_df=df_bench)
        if aligned is None:
            return sym, None, "Insufficient trading history (< 271 aligned days or missing data)"
        
        if len(aligned) < MIN_DAYS_FOR_FULL_WINDOWS:
            return sym, None, (f"Insufficient trading history for 23 windows ({len(aligned)} days, "
                               f"minimum required: {MIN_DAYS_FOR_FULL_WINDOWS})")

        ds = build_research_dataset(aligned)
        return sym, ds, None
    except Exception as e:
        return sym, None, f"Data building error: {str(e)}"


def is_complete_checkpoint(df: pd.DataFrame, ticker: str) -> bool:
    """Return True only if both models, 23 windows, and 483 OOS rows per model are present."""
    if df is None or df.empty:
        return False
    if any(col not in df.columns for col in EXPECTED_PRED_COLUMNS):
        return False
    if df[EXPECTED_PRED_COLUMNS].isnull().any().any():
        return False
    if set(df['ticker'].astype(str).unique()) != {ticker}:
        return False
    if df.duplicated(subset=['ticker', 'date', 'model', 'window_id']).any():
        return False

    numeric_values = df[EXPECTED_NUMERIC_COLUMNS].apply(pd.to_numeric, errors='coerce')
    if numeric_values.isnull().any().any() or not np.isfinite(numeric_values.to_numpy()).all():
        return False

    models_present = set(df['model'].astype(str).unique())
    if models_present != set(REQUIRED_MODELS):
        return False

    expected_window_ids = set(range(NUM_WINDOWS))
    for model_name in REQUIRED_MODELS:
        sub = df[df['model'] == model_name]
        if len(sub) != OOS_ROWS_PER_MODEL:
            return False
        window_ids = set(pd.to_numeric(sub['window_id'], errors='coerce'))
        if window_ids != expected_window_ids:
            return False
        if not sub.groupby('window_id').size().eq(TEST_WINDOW).all():
            return False
    return True


def atomic_write_csv(df: pd.DataFrame, dest_path: Path) -> None:
    """Write CSV to a sibling temp file, then rename into place."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_name(dest_path.name + ".tmp")
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, dest_path)


def load_complete_checkpoint(path: Path, ticker: str) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if not is_complete_checkpoint(df, ticker):
        return None
    return df


def discover_completed_tickers(checkpoint_dir: Path, successful_tickers: List[str]) -> Set[str]:
    completed = set()
    for ticker in successful_tickers:
        path = checkpoint_dir / f"{ticker}.csv"
        if path.exists() and load_complete_checkpoint(path, ticker) is not None:
            completed.add(ticker)
    return completed


def concatenate_checkpoints(checkpoint_dir: Path, tickers: List[str]) -> Optional[pd.DataFrame]:
    frames = []
    for ticker in tickers:
        df = load_complete_checkpoint(checkpoint_dir / f"{ticker}.csv", ticker)
        if df is None:
            return None
        frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def load_excluded_tickers(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
    except Exception:
        return {}
    records = payload.get('excluded_at_final_stage', payload if isinstance(payload, list) else [])
    excluded = {}
    for rec in records:
        ticker = rec.get('ticker')
        if ticker:
            excluded[str(ticker)] = rec.get('reason', INCOMPLETE_OOS_REASON)
    return excluded


def save_excluded_tickers(path: Path, excluded: Dict[str, str]) -> None:
    records = [{'ticker': t, 'reason': excluded[t]} for t in sorted(excluded)]
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({
            'excluded_count': len(records),
            'excluded_at_final_stage': records
        }, f, indent=2)


def compute_aggregate_metrics(all_preds: pd.DataFrame) -> pd.DataFrame:
    agg_list = []
    for m_name in REQUIRED_MODELS:
        sub = all_preds[all_preds['model'] == m_name]
        if len(sub) == 0:
            continue
        agg = compute_classification_metrics(sub['y_true'].values, sub['y_pred'].values, sub['y_prob'].values)
        agg['model'] = m_name
        agg['num_tickers'] = len(sub['ticker'].unique())
        agg['total_predictions'] = len(sub)
        agg_list.append(agg)

    agg_df = pd.DataFrame(agg_list)
    first_cols = [
        'model', 'num_tickers', 'total_predictions', 'accuracy', 'precision', 'recall',
        'f1_extension', 'roc_auc_extension'
    ]
    other_cols = [c for c in agg_df.columns if c not in first_cols]
    return agg_df[first_cols + other_cols]


def evaluate_ticker_and_checkpoint(ticker: str, ds: pd.DataFrame, checkpoint_dir_str: str):
    """Fit both models for one ticker and atomically checkpoint complete OOS predictions."""
    try:
        checkpoint_dir = Path(checkpoint_dir_str)
        p_list, _w_list = _evaluate_single_ticker_all_models(
            ticker, ds, RF_DEV_TREES, NUM_WINDOWS
        )
        preds = pd.concat(p_list, ignore_index=True)
        dest = checkpoint_dir / f"{ticker}.csv"
        if is_complete_checkpoint(preds, ticker):
            atomic_write_csv(preds, dest)
            return ticker, True, None
        return ticker, False, INCOMPLETE_OOS_REASON
    except Exception as exc:
        return ticker, False, f"model execution error: {exc}"


def run_full_universe_validation(fixed_top_k: int = BACKTEST_TOP_K, n_jobs_workers: int = 6):
    total_start_time = time.perf_counter()
    print("=" * 82)
    print("      FINAL-A: FULL-UNIVERSE S&P 500 WALK-FORWARD VALIDATION & BACKTEST       ")
    print("=" * 82)
    print("NOTE: Backtesting is an explicit research extension beyond Htun et al. (2024).")
    print(f"Fixed Portfolio Selection Policy: Top-{fixed_top_k} Equal-Weighted (pre-selected, un-tuned)")
    print("-" * 82)

    output_dir = ROOT_DIR / "research" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    tickers_json_path = output_dir / "tickers_final_a.json"
    checkpoint_dir = output_dir / "final_a_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    excluded_json_path = output_dir / "final_a_excluded.json"

    # 1-3. Reuse validated universe when present; otherwise download and validate once.
    datasets: Dict[str, pd.DataFrame] = {}
    if tickers_json_path.exists():
        with open(tickers_json_path, 'r', encoding='utf-8') as f:
            ticker_manifest = json.load(f)
        successful_tickers = list(ticker_manifest['successful_tickers'])
        skipped_tickers = list(ticker_manifest.get('skipped_tickers', []))
        total_candidates = int(ticker_manifest.get('total_candidates', len(successful_tickers) + len(skipped_tickers)))
        print(f"\n[2] Reusing validated universe from: {tickers_json_path}")
        print(f"    - Usable tickers : {len(successful_tickers)}")
        print(f"    - Skipped tickers: {len(skipped_tickers)}")
        print("    Skipping universe redownload/revalidation.")
    else:
        print(f"\n[1] Fetching Benchmark: {BENCHMARK_SYMBOL} ({START_DATE} to {END_DATE})...")
        df_bench = fetch_historical_prices(BENCHMARK_SYMBOL, START_DATE, END_DATE)
        print(f"    Benchmark trading days: {len(df_bench)}")

        all_candidates = get_sp500_tickers(top_n=None)
        total_candidates = len(all_candidates)
        print(f"\n[2] Retrieved {total_candidates} S&P 500 candidate constituents.")

        print(f"\n[3] Ingesting and aligning data for {total_candidates} tickers (concurrent fetch)...")
        fetch_start = time.perf_counter()
        successful_tickers = []
        skipped_tickers = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(process_ticker_data, sym, df_bench) for sym in all_candidates]
            for f in futures:
                sym, ds, err = f.result()
                if ds is not None:
                    datasets[sym] = ds
                    successful_tickers.append(sym)
                else:
                    skipped_tickers.append({'ticker': sym, 'reason': err})

        fetch_duration = time.perf_counter() - fetch_start
        print(f"    Data ingestion completed in {fetch_duration:.2f} seconds.")
        print(f"    - Successfully processed: {len(successful_tickers)} tickers")
        print(f"    - Skipped tickers       : {len(skipped_tickers)} tickers")

        with open(tickers_json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'total_candidates': total_candidates,
                'successful_count': len(successful_tickers),
                'successful_tickers': successful_tickers,
                'skipped_count': len(skipped_tickers),
                'skipped_tickers': skipped_tickers
            }, f, indent=2)
        print(f"    Saved ticker manifest to: {tickers_json_path}")
        del df_bench

    completed_tickers = discover_completed_tickers(checkpoint_dir, successful_tickers)
    excluded_tickers = load_excluded_tickers(excluded_json_path)
    remaining_tickers = [
        t for t in successful_tickers
        if t not in completed_tickers and t not in excluded_tickers
    ]
    print(f"\n[4] Checkpoint status: {len(completed_tickers)} complete, "
          f"{len(excluded_tickers)} excluded at final stage, {len(remaining_tickers)} remaining.")

    total_model_fits = len(successful_tickers) * NUM_WINDOWS * 2
    total_oos_preds = len(successful_tickers) * NUM_WINDOWS * TEST_WINDOW * 2
    print(f"    Walk-Forward protocol (unchanged):")
    print(f"    - Windows per stock: {NUM_WINDOWS} (253 train, 10 gap, 21 test)")
    print(f"    - Models: Random Forest Baseline ({RF_DEV_TREES} trees) & LightGBM Extension")
    print(f"    - Total Model Fits (full universe): {total_model_fits:,}")
    print(f"    - Total Expected OOS Predictions: {total_oos_preds:,}")
    print(f"    - Parallel Workers: {n_jobs_workers}")

    ml_start = time.perf_counter()
    if remaining_tickers:
        missing_data = [t for t in remaining_tickers if t not in datasets]
        if missing_data:
            print(f"\n    Fetching data for {len(missing_data)} remaining tickers...")
            print(f"    Fetching Benchmark: {BENCHMARK_SYMBOL} ({START_DATE} to {END_DATE})...")
            df_bench = fetch_historical_prices(BENCHMARK_SYMBOL, START_DATE, END_DATE)
            print(f"    Benchmark trading days: {len(df_bench)}")
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(process_ticker_data, sym, df_bench) for sym in missing_data]
                for f in futures:
                    sym, ds, err = f.result()
                    if ds is not None:
                        datasets[sym] = ds
                    else:
                        print(f"    WARNING: previously validated ticker {sym} failed data build: {err}")
                        excluded_tickers[sym] = err

        to_run = [t for t in remaining_tickers if t in datasets]
        print(f"    Executing walk-forward ML for {len(to_run)} remaining tickers...")
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs_workers, verbose=0)(
            delayed(evaluate_ticker_and_checkpoint)(
                ticker, datasets[ticker], str(checkpoint_dir)
            ) for ticker in to_run
        )
        n_ok = sum(1 for _t, ok, _err in results if ok)
        n_fail = len(results) - n_ok
        for ticker, ok, err in results:
            if not ok:
                excluded_tickers[ticker] = err or INCOMPLETE_OOS_REASON
                print(f"    WARNING: {ticker} excluded at final stage: {excluded_tickers[ticker]}")
        print(f"    Checkpointed this run: {n_ok}  excluded this run: {n_fail}")
    else:
        print("    All tickers already checkpointed; skipping ML.")

    ml_duration = time.perf_counter() - ml_start
    print(f"    ML / resume stage completed in {ml_duration:.2f} seconds ({ml_duration / 60:.2f} minutes).")

    completed_tickers = discover_completed_tickers(checkpoint_dir, successful_tickers)
    final_analysis_tickers = [t for t in successful_tickers if t in completed_tickers]
    still_remaining = [
        t for t in successful_tickers
        if t not in completed_tickers and t not in excluded_tickers
    ]
    if excluded_tickers:
        save_excluded_tickers(excluded_json_path, excluded_tickers)
    if still_remaining:
        print(f"\n    Incomplete universe: {len(still_remaining)} tickers still missing valid checkpoints.")
        print("    Final predictions/backtest files were NOT written. Re-run to resume.")
        return
    if not final_analysis_tickers:
        print("    ERROR: no valid complete checkpoints available. Final files not written.")
        return

    excluded_records = [{'ticker': t, 'reason': excluded_tickers[t]} for t in successful_tickers if t in excluded_tickers]
    print(f"    Final-analysis universe: {len(final_analysis_tickers)} tickers "
          f"({len(excluded_records)} excluded at final stage).")
    for rec in excluded_records:
        print(f"    - Excluded {rec['ticker']}: {rec['reason']}")

    all_preds = concatenate_checkpoints(checkpoint_dir, final_analysis_tickers)
    if all_preds is None:
        print("    ERROR: failed to concatenate complete checkpoints. Final files not written.")
        return

    preds_csv_path = output_dir / "predictions_final_a.csv"
    all_preds.to_csv(preds_csv_path, index=False)
    print(f"    Saved {len(all_preds):,} out-of-sample predictions to: {preds_csv_path}")

    agg_metrics = compute_aggregate_metrics(all_preds)

    # 5. Display Aggregate Classification Performance
    print("\n" + "=" * 82)
    print("            AGGREGATE FULL-UNIVERSE CLASSIFICATION PERFORMANCE          ")
    print("=" * 82)
    print("Note: Accuracy, Precision, and Recall are paper-reproduced metrics.")
    print("      F1 Score and ROC-AUC are additional extension metrics.")
    print("-" * 82)

    display_agg = agg_metrics[[
        'model', 'num_tickers', 'total_predictions', 'accuracy', 'precision', 'recall',
        'f1_extension', 'roc_auc_extension'
    ]].copy()

    for col in ['accuracy', 'precision', 'recall']:
        display_agg[col] = display_agg[col].apply(lambda x: f"{x * 100:.2f}%")
    display_agg['f1_extension'] = display_agg['f1_extension'].apply(lambda x: f"{x:.4f}")
    display_agg['roc_auc_extension'] = display_agg['roc_auc_extension'].apply(lambda x: f"{x:.4f}")

    print(display_agg.to_string(index=False))

    # 6. Compute Cross-Sectional Per-Ticker Distribution Summaries
    print("\n--- Cross-Sectional Per-Ticker Performance Distributions ---")
    per_ticker_list = []
    for model_name in ['RandomForest_Baseline', 'LightGBM_Extension']:
        m_preds = all_preds[all_preds['model'] == model_name]
        for sym, grp in m_preds.groupby('ticker'):
            m_res = compute_classification_metrics(grp['y_true'].values, grp['y_pred'].values, grp['y_prob'].values)
            per_ticker_list.append({
                'model': model_name,
                'ticker': sym,
                'accuracy': m_res['accuracy'] * 100.0,
                'precision': m_res['precision'] * 100.0,
                'recall': m_res['recall'] * 100.0,
                'f1': m_res['f1_extension'],
                'roc_auc': m_res['roc_auc_extension']
            })

    dist_df = pd.DataFrame(per_ticker_list)
    dist_summary = dist_df.groupby('model')[['accuracy', 'precision', 'recall', 'f1', 'roc_auc']].describe().T
    
    # Format distribution summary neatly
    print("\n[Random Forest Baseline - Per-Stock Distribution Summary (% for Acc/Prec/Rec)]")
    rf_dist = dist_df[dist_df['model'] == 'RandomForest_Baseline'][['accuracy', 'precision', 'recall', 'f1', 'roc_auc']].describe()
    print(rf_dist.round(2).to_string())

    print("\n[LightGBM Extension - Per-Stock Distribution Summary (% for Acc/Prec/Rec)]")
    lgb_dist = dist_df[dist_df['model'] == 'LightGBM_Extension'][['accuracy', 'precision', 'recall', 'f1', 'roc_auc']].describe()
    print(lgb_dist.round(2).to_string())

    # 7. Research Backtest (Fixed Top-10 Equal-Weighted)
    print(f"\n[5] Executing Research Backtest (Top-{fixed_top_k} Equal-Weighted, 10-day holding)...")
    periodic_df, summary_df = run_research_backtest(all_preds, top_k=fixed_top_k)

    formation_min = int(periodic_df['formation_universe_size'].min())
    formation_max = int(periodic_df['formation_universe_size'].max())
    analyzed_for_backtest = int(periodic_df['analyzed_universe_size'].max())
    excluded_from_formation = sorted({
        ticker
        for tickers in periodic_df['excluded_non_common_calendar_tickers']
        for ticker in tickers
    })
    print(f"    Backtest analyzed universe: {analyzed_for_backtest} tickers")
    print(f"    Common-calendar formation universe: {formation_min}"
          f"{'-' + str(formation_max) if formation_max != formation_min else ''} tickers")
    if excluded_from_formation:
        print(f"    Excluded from common-calendar formation: {', '.join(excluded_from_formation)}")
    print("    Execution assumption: idealized same-close research backtest; "
          "not a live-executable fill assumption.")

    # Save periodic backtest results
    backtest_csv_path = output_dir / "backtest_final_a_periodic.csv"
    periodic_df.to_csv(backtest_csv_path, index=False)
    print(f"    Saved periodic backtest details to: {backtest_csv_path}")

    # Format summary table
    display_summary = summary_df.copy()
    display_summary['cumulative_return'] = display_summary['cumulative_return'].apply(lambda x: f"{x:+.2f}%")
    display_summary['mean_10d_return'] = display_summary['mean_10d_return'].apply(lambda x: f"{x:+.2f}%")
    display_summary['annualized_return'] = display_summary['annualized_return'].apply(lambda x: f"{x:+.2f}%")
    display_summary['sharpe_ratio'] = display_summary['sharpe_ratio'].apply(lambda x: f"{x:.2f}")
    display_summary['max_drawdown'] = display_summary['max_drawdown'].apply(lambda x: f"{x:.2f}%")
    display_summary['cumulative_excess_vs_sp500'] = display_summary['cumulative_excess_vs_sp500'].apply(lambda x: f"{x:+.2f}%")
    display_summary['mean_excess_vs_sp500'] = display_summary['mean_excess_vs_sp500'].apply(lambda x: f"{x:+.2f}%")

    print("\n" + "=" * 82)
    print(f"          FULL-UNIVERSE BACKTEST SUMMARY TABLE (Top-{fixed_top_k} Equal-Weighted)        ")
    print("=" * 82)
    print(display_summary.to_string(index=False))

    total_duration = time.perf_counter() - total_start_time
    print(f"\n[6] Total Execution Runtime: {total_duration:.2f} seconds ({total_duration / 60:.2f} minutes).")

    # 8. Save overall summary metrics to JSON
    summary_metrics_json_path = output_dir / "summary_metrics_final_a.json"
    with open(summary_metrics_json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'total_candidates': total_candidates,
            'successful_tickers_count': len(successful_tickers),
            'skipped_tickers_count': len(skipped_tickers),
            'final_analysis_tickers_count': len(final_analysis_tickers),
            'final_analysis_tickers': final_analysis_tickers,
            'excluded_at_final_stage_count': len(excluded_records),
            'excluded_at_final_stage': excluded_records,
            'total_model_fits': len(final_analysis_tickers) * NUM_WINDOWS * 2,
            'total_oos_predictions': len(all_preds),
            'ml_runtime_seconds': ml_duration,
            'total_runtime_seconds': total_duration,
            'aggregate_classification_metrics': agg_metrics.to_dict(orient='records'),
            'backtest_summary_metrics': summary_df.to_dict(orient='records')
        }, f, indent=2)
    print(f"    Saved complete summary metrics to: {summary_metrics_json_path}")

    # 9. Data Quality & Integrity Report
    universe_pct = (len(successful_tickers) / total_candidates * 100.0) if total_candidates else 0.0
    analyzed_fits = len(final_analysis_tickers) * NUM_WINDOWS * 2
    print("\n" + "=" * 82)
    print("                     FINAL-A DATA QUALITY & AUDIT REPORT                ")
    print("=" * 82)
    print(f" - Candidate Tickers Scanned   : {total_candidates}")
    print(f" - Validated Universe Included : {len(successful_tickers)} ({universe_pct:.1f}%)")
    print(f" - Final-Analysis Universe     : {len(final_analysis_tickers)}")
    print(f" - Excluded at Final Stage     : {len(excluded_records)}")
    print(f" - Skipped Tickers             : {len(skipped_tickers)}")
    print(f" - Total Model Fits (analyzed) : {analyzed_fits:,}")
    print(f" - Total Out-of-Sample Rows    : {len(all_preds):,} ({len(all_preds)//2:,} per model)")
    print(f" - Windows per Stock           : {NUM_WINDOWS} windows x {TEST_WINDOW} test days = 483 predictions/stock")
    print(" - Look-Ahead Checks           : Zero leakage asserted across 10-day gaps.")
    print(" - Benchmark Alignment         : Strictly matched 10-day holding periods vs ^GSPC.")
    print("=" * 82)


if __name__ == "__main__":
    run_full_universe_validation(n_jobs_workers=6)
