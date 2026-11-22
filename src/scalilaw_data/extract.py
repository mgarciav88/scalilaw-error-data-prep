"""Pull one WandB run's history into the canonical tidy per-step DataFrame.

Scope is deliberately narrow: WandB rows in, a schema-conformant DataFrame out. Gap
detection, coverage checks, and writing to disk are validate.py/manifest.py's job
(next stage) -- keeping this module to just the transform makes it independently
testable against a handful of recorded fixture rows, no live API needed.
"""

from typing import Any

import pandas as pd

from scalilaw_data.config import TOKENS_PER_STEP, WANDB_ENTITY
from scalilaw_data.schema import COLUMNS, STEP, TOKENS_SEEN, WANDB_KEYS, RegistryEntry
from scalilaw_data.wandb_client import get_run, scan_run_history


def extract_run(entry: RegistryEntry) -> pd.DataFrame:
    """Pull and tidy one run's history per the RegistryEntry's resolved run id.

    Raises:
        ValueError: if entry has no resolved wandb_run_id. AMBIGUOUS entries still
            carry a (best-guess) run id and are extractable; only UNRESOLVED entries
            lack one, and must not be silently skipped into an empty frame -- callers
            should see the failure and decide, not get a quiet no-op.
    """
    if entry.wandb_run_id is None:
        raise ValueError(
            f"{entry.model_size} seed={entry.seed} [{entry.variant.value}] has no "
            f"resolved wandb_run_id (status={entry.resolution_status.value}) -- "
            "cannot extract."
        )
    run = get_run(WANDB_ENTITY, entry.wandb_project, entry.wandb_run_id)
    rows = list(scan_run_history(run, keys=list(WANDB_KEYS.values())))
    return _rows_to_frame(rows)


def _rows_to_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=list(COLUMNS)).astype(COLUMNS)

    df = pd.DataFrame(rows)
    df = df.rename(columns={wandb_key: col for col, wandb_key in WANDB_KEYS.items()})

    # A column is entirely absent (not just sparse) if no scanned row logged that
    # metric at all -- e.g. a run that never logged validation/lm_loss. Fill it in
    # as all-missing rather than letting the later astype() KeyError. Uses float
    # nan, not pd.NA -- pd.NA doesn't cast cleanly to float64 via astype() below.
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = float("nan")

    df[STEP] = df[STEP].astype("int64")
    df[TOKENS_SEEN] = df[STEP] * TOKENS_PER_STEP

    # Guard against a step logged twice (seen on resumed/retried jobs elsewhere in
    # this workspace) -- keep the most recently scanned value for that step.
    df = df.sort_values(STEP).drop_duplicates(subset=STEP, keep="last").reset_index(drop=True)

    return df[list(COLUMNS)].astype(COLUMNS)
