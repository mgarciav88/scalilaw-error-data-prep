import pandas as pd
import pytest

from scalilaw_data.schema import RegistryEntry, ResolutionStatus, Variant
from scalilaw_data.validate import validate_run


def _entry(dedup: bool | None = False) -> RegistryEntry:
    return RegistryEntry(
        model_size="14m",
        seed=1,
        variant=Variant.JOINT,
        dedup=dedup,
        wandb_project="pythia-extra-seeds",
        wandb_group="g",
        wandb_run_id="r",
        wandb_run_name="ip-1-2-3-4-0",
        resolution_status=ResolutionStatus.CLEAN,
        notes="",
    )


def _frame(steps: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "step": steps,
            "tokens_seen": [s * 2_097_152 for s in steps],
            "train_lm_loss": [1.0] * len(steps),
            "validation_lm_loss": [None] * len(steps),
        }
    )


def test_empty_frame_raises():
    df = pd.DataFrame(columns=["step", "tokens_seen", "train_lm_loss", "validation_lm_loss"])
    with pytest.raises(ValueError, match="zero rows"):
        validate_run(df, _entry())


def test_unconfirmed_dedup_raises():
    with pytest.raises(ValueError, match="unconfirmed dedup"):
        validate_run(_frame([1, 2]), _entry(dedup=None))


def test_gapless_coverage():
    stats = validate_run(_frame([1, 2, 3]), _entry())
    assert stats.step_min == 1
    assert stats.step_max == 3
    assert stats.n_rows == 3
    assert stats.n_gaps == 0


def test_gap_detected():
    stats = validate_run(_frame([1, 2, 5]), _entry())
    assert stats.n_gaps == 1


def test_multiple_gaps_counted():
    stats = validate_run(_frame([1, 5, 6, 10]), _entry())
    assert stats.n_gaps == 2
