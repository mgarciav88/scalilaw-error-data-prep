from scalilaw_data.config import TOKENS_PER_STEP
from scalilaw_data.extract import _rows_to_frame
from scalilaw_data.schema import COLUMNS


def test_columns_and_order(sample_history_rows):
    df = _rows_to_frame(sample_history_rows)
    assert list(df.columns) == list(COLUMNS)


def test_tokens_seen_matches_formula(sample_history_rows):
    df = _rows_to_frame(sample_history_rows)
    assert (df["tokens_seen"] == df["step"] * TOKENS_PER_STEP).all()


def test_sparse_validation_loss_is_nan_elsewhere(sample_history_rows):
    df = _rows_to_frame(sample_history_rows)
    assert df["validation_lm_loss"].isna().sum() == len(df) - 1
    assert df.iloc[-1]["validation_lm_loss"] == 3.58689


def test_missing_metric_column_filled_with_nan():
    rows = [{"_step": 1, "train/lm_loss": 5.0}]  # no validation/lm_loss logged at all
    df = _rows_to_frame(rows)
    assert df["validation_lm_loss"].isna().all()


def test_duplicate_step_keeps_last():
    rows = [
        {"_step": 1, "train/lm_loss": 9.0},
        {"_step": 1, "train/lm_loss": 8.0},  # resumed/retried log for the same step
    ]
    df = _rows_to_frame(rows)
    assert len(df) == 1
    assert df.iloc[0]["train_lm_loss"] == 8.0


def test_empty_rows_returns_empty_typed_frame():
    df = _rows_to_frame([])
    assert df.empty
    assert list(df.columns) == list(COLUMNS)


def test_sorted_by_step():
    rows = [{"_step": 3, "train/lm_loss": 1.0}, {"_step": 1, "train/lm_loss": 2.0}]
    df = _rows_to_frame(rows)
    assert df["step"].tolist() == [1, 3]
