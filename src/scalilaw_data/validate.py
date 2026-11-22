"""Coverage and hygiene checks on an extracted run -- fail loud, never assume.

Run once per extracted frame, before anything is written to disk. Catastrophic
problems (zero rows, unconfirmed dedup stratum) raise immediately and block the
write. Partial coverage (gaps, a step_min above 1) is recorded, not rejected -- per
the execution plan's "always verify actual step min/max, don't assume completeness"
guidance: a partial run (like base Pythia-2.8B's missing-first-third case) is still
useful data as long as its gaps are visible to whoever consumes it next, not silently
smoothed over.
"""

from dataclasses import dataclass

import pandas as pd

from scalilaw_data.schema import STEP, RegistryEntry


@dataclass
class CoverageStats:
    step_min: int
    step_max: int
    n_rows: int
    n_gaps: int  # count of step-to-step jumps > 1


def validate_run(df: pd.DataFrame, entry: RegistryEntry) -> CoverageStats:
    """Check an extracted frame is safe to write, and report its actual coverage.

    Raises:
        ValueError: on zero rows, or an unconfirmed dedup status -- both mean the
            data isn't safe to fold into the dataset yet, so this must stop the
            pipeline rather than write something misleading.
    """
    label = (
        f"{entry.model_size} seed={entry.seed} [{entry.variant.value}] "
        f"(run {entry.wandb_run_id})"
    )
    if df.empty:
        raise ValueError(f"{label} extracted zero rows -- refusing to write.")
    if entry.dedup is None:
        raise ValueError(
            f"{label} has unconfirmed dedup status -- refusing to write until this "
            "is resolved. Never assume: duped vs deduped Pile differ ~5% in loss, "
            "and a real subset of PolyPythias runs trained on deduped data."
        )

    steps = df[STEP]
    step_min, step_max = int(steps.min()), int(steps.max())
    n_gaps = int((steps.diff().dropna() > 1).sum())
    return CoverageStats(step_min=step_min, step_max=step_max, n_rows=len(df), n_gaps=n_gaps)
