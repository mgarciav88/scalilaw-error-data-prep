"""Canonical schema for extracted per-run loss data and its provenance metadata.

Two deliberately separate shapes:

- The *tidy frame* (see COLUMNS) is a clean numeric time series: step, tokens, loss.
  It's what gets written to parquet and handed to the stats package. No strings, no
  identifiers -- keeping it numeric-only means downstream analysis code never has to
  filter metadata columns back out.
- ``RunMetadata`` is everything about *provenance*: which model/seed/variant this is,
  where it came from in WandB, and what the extractor observed about its quality. This
  travels in the manifest (see manifest.py), keyed by output filename, not in the
  parquet itself.
"""

from dataclasses import dataclass, field
from enum import StrEnum


class Variant(StrEnum):
    """How a run's seed was assigned.

    Standard PolyPythias runs vary init AND data order jointly under one seed
    (JOINT). Only the 160M decoupled study separates them -- DATA_SEED holds init
    fixed and varies data order, WEIGHT_SEED holds data order fixed and varies init.
    Getting this wrong silently conflates the two noise sources Phase 2 of the
    execution plan tries to decompose, so it's a required field, not inferred.
    """

    JOINT = "joint"
    DATA_SEED = "data_seed"
    WEIGHT_SEED = "weight_seed"


class ResolutionStatus(StrEnum):
    """How confidently discovery identified the canonical run for a (model, seed, variant).

    Used in the registry so ambiguous cases (e.g. 70m's duplicate groups per seed) are
    flagged for human review rather than silently guessed at.
    """

    CLEAN = "clean"
    AMBIGUOUS = "ambiguous"
    UNRESOLVED = "unresolved"


# --- Tidy per-step frame --------------------------------------------------------
# Column name -> pandas dtype, in on-disk column order. extract.py must produce
# exactly these columns (see validate.py for the corresponding schema check).

STEP = "step"
TOKENS_SEEN = "tokens_seen"
TRAIN_LM_LOSS = "train_lm_loss"
VALIDATION_LM_LOSS = "validation_lm_loss"

COLUMNS: dict[str, str] = {
    STEP: "int64",
    TOKENS_SEEN: "int64",
    TRAIN_LM_LOSS: "float64",
    VALIDATION_LM_LOSS: "float64",  # sparse; NaN where not logged
}

# Source WandB history keys these columns are extracted from.
WANDB_KEYS: dict[str, str] = {
    STEP: "_step",
    TRAIN_LM_LOSS: "train/lm_loss",
    VALIDATION_LM_LOSS: "validation/lm_loss",
}


@dataclass
class RegistryEntry:
    """Identifies which WandB run is canonical for one (model, seed, variant).

    Produced by ``discover`` *before* any data is pulled -- at this point nothing is
    known about the run's actual content, only which run the registry believes is the
    right one to extract. ``wandb_run_id`` is None for UNRESOLVED entries (e.g. 31m's
    zero-finished-groups case) where no candidate could be identified at all.
    """

    model_size: str  # e.g. "14m", "410m" -- matches PolyPythias/HF naming
    seed: int
    variant: Variant
    dedup: bool | None  # None = could not be confirmed from config -- needs manual check,
    # never assume; a real subset of PolyPythias attempts (all so far in 'failed' groups)
    # ran on /mnt/ssd-2/pile_deduped/... while most run on non-deduped Pile.
    wandb_project: str
    wandb_group: str
    wandb_run_id: str | None
    wandb_run_name: str | None  # the sub-run's display name in the WandB UI (usually the
    # training node's hostname, e.g. "ip-26-0-144-150-0") -- NOT the run id. This is what a
    # human actually sees first when browsing a group in the UI, so it's kept here for
    # direct cross-referencing; None for UNRESOLVED entries with no shard at all.
    resolution_status: ResolutionStatus
    notes: str = field(default="")


@dataclass
class RunMetadata:
    """Provenance and data-quality record for one extracted (model, seed, variant).

    Wraps the RegistryEntry that identified the run with the coverage stats only
    known once extraction has actually happened. One instance per output parquet
    file; serialized into manifest.json keyed by output filename (see manifest.py).
    """

    registry_entry: RegistryEntry
    step_min: int
    step_max: int
    n_rows: int
    n_gaps: int  # count of non-contiguous step jumps detected by validate.py
    extracted_at: str  # ISO 8601 UTC timestamp
