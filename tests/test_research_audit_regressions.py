import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import research.validation as validation
from research.backtest import (
    IDEALIZED_SAME_CLOSE_EXECUTION,
    extract_formation_date_predictions,
    run_research_backtest,
)
from research.run_final_a_full_universe import (
    EXPECTED_PRED_COLUMNS,
    REQUIRED_MODELS,
    is_complete_checkpoint,
)
from research.settings import NUM_WINDOWS, TEST_WINDOW


def _complete_checkpoint(ticker: str = "TEST") -> pd.DataFrame:
    rows = []
    for model in REQUIRED_MODELS:
        for window_id in range(NUM_WINDOWS):
            for row_id in range(TEST_WINDOW):
                date = pd.Timestamp('2020-01-01') + pd.Timedelta(
                    days=window_id * TEST_WINDOW + row_id
                )
                rows.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'ticker': ticker,
                    'window_id': window_id,
                    'model': model,
                    'y_true': row_id % 2,
                    'y_pred': row_id % 2,
                    'y_prob': 0.75 if row_id % 2 else 0.25,
                    'close_stock': 100.0 + row_id,
                    'close_bench': 3000.0 + row_id,
                    'fwd_ret_stock_10d': 1.0,
                    'fwd_ret_bench_10d': 0.5,
                })
    return pd.DataFrame(rows, columns=EXPECTED_PRED_COLUMNS)


def test_checkpoint_requires_exact_complete_finite_unique_coverage():
    complete = _complete_checkpoint()
    assert is_complete_checkpoint(complete, "TEST")

    extra_model = complete.copy()
    extra_model.loc[len(extra_model)] = extra_model.iloc[0].to_dict() | {'model': 'UnexpectedModel'}
    assert not is_complete_checkpoint(extra_model, "TEST")

    wrong_window_count = complete.copy()
    model_mask = wrong_window_count['model'].eq(REQUIRED_MODELS[0])
    source_index = wrong_window_count[model_mask & wrong_window_count['window_id'].eq(0)].index[0]
    wrong_window_count.loc[source_index, 'window_id'] = 1
    assert not is_complete_checkpoint(wrong_window_count, "TEST")

    duplicate = complete.copy()
    duplicate.iloc[1] = duplicate.iloc[0]
    assert not is_complete_checkpoint(duplicate, "TEST")

    with_null = complete.copy()
    with_null.loc[0, 'y_prob'] = np.nan
    assert not is_complete_checkpoint(with_null, "TEST")

    with_infinity = complete.copy()
    with_infinity.loc[0, 'close_stock'] = np.inf
    assert not is_complete_checkpoint(with_infinity, "TEST")


def test_backtest_discloses_common_calendar_universe_without_changing_arithmetic():
    rows = []
    models = ['RandomForest_Baseline', 'LightGBM_Extension']
    common_tickers = ['A', 'B', 'C', 'D']
    for model in models:
        for ticker_id, ticker in enumerate(common_tickers):
            for date in ['2020-01-02', '2020-01-03']:
                rows.append({
                    'date': date,
                    'ticker': ticker,
                    'window_id': 0,
                    'model': model,
                    'y_prob': 0.9 - ticker_id * 0.1,
                    'fwd_ret_stock_10d': 4.0 - ticker_id,
                    'fwd_ret_bench_10d': 1.0,
                })
        for date in ['2020-02-03', '2020-02-04']:
            rows.append({
                'date': date,
                'ticker': 'LATE',
                'window_id': 0,
                'model': model,
                'y_prob': 0.99,
                'fwd_ret_stock_10d': 99.0,
                'fwd_ret_bench_10d': 2.0,
            })

    predictions = pd.DataFrame(rows)
    formation = extract_formation_date_predictions(predictions)

    assert set(formation['ticker']) == set(common_tickers)
    assert formation['analyzed_universe_size'].eq(5).all()
    assert formation['formation_universe_size'].eq(4).all()
    assert formation['excluded_non_common_calendar_tickers'].apply(
        lambda tickers: tickers == ['LATE']
    ).all()

    periodic, summary = run_research_backtest(predictions, top_k=2)
    assert periodic['portfolio_return'].eq(3.5).all()
    assert periodic['analyzed_universe_size'].eq(5).all()
    assert periodic['formation_universe_size'].eq(4).all()
    assert summary['formation_universe_size_min'].eq(4).all()
    assert summary['formation_universe_size_max'].eq(4).all()
    assert summary['execution_assumption'].eq(IDEALIZED_SAME_CLOSE_EXECUTION).all()


def test_run_experiment_pipeline_uses_defined_model_names():
    def fake_evaluator(ticker, _dataset, _rf_trees, _max_windows):
        predictions = []
        window_metrics = []
        for model in validation.MODEL_NAMES:
            predictions.append(pd.DataFrame([{
                'ticker': ticker,
                'model': model,
                'y_true': 1,
                'y_pred': 1,
                'y_prob': 0.75,
            }]))
            window_metrics.append(pd.DataFrame([{
                'ticker': ticker,
                'model': model,
                'window_id': 0,
            }]))
        return predictions, window_metrics

    original_evaluator = validation._evaluate_single_ticker_all_models
    validation._evaluate_single_ticker_all_models = fake_evaluator
    try:
        _, _, aggregate = validation.run_experiment_pipeline(
            {'TEST': pd.DataFrame()}, n_jobs=1
        )
    finally:
        validation._evaluate_single_ticker_all_models = original_evaluator

    assert set(aggregate['model']) == set(validation.MODEL_NAMES)


if __name__ == '__main__':
    test_checkpoint_requires_exact_complete_finite_unique_coverage()
    test_backtest_discloses_common_calendar_universe_without_changing_arithmetic()
    test_run_experiment_pipeline_uses_defined_model_names()
    print('3 targeted audit regression tests passed')
