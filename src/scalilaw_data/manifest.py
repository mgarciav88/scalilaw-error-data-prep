"""Provenance + data-quality manifest for extracted parquet files (data/manifest.json).

One JSON entry per output file, keyed by filename, so provenance (which WandB run,
resolution confidence) and coverage stats (step range, gaps) travel alongside the
parquet without needing to re-derive anything from the registry or re-scan the data.
StrEnum fields (Variant, ResolutionStatus) serialize as plain strings for free, since
StrEnum members are real str instances.
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from scalilaw_data.config import DATA_DIR
from scalilaw_data.schema import RegistryEntry, RunMetadata, Variant

MANIFEST_PATH = DATA_DIR / "manifest.json"


def output_filename(entry: RegistryEntry) -> str:
    """<model>_seed<N>[_<variant>].parquet -- the on-disk name for one registry entry."""
    suffix = "" if entry.variant == Variant.JOINT else f"_{entry.variant.value}"
    return f"{entry.model_size}_seed{entry.seed}{suffix}.parquet"


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        result: dict[str, Any] = json.load(f)
        return result


def record(meta: RunMetadata, path: Path = MANIFEST_PATH) -> None:
    """Merge one run's metadata into the manifest, keyed by its output filename.

    Read-modify-write against the existing manifest so extracting one model doesn't
    clobber entries already recorded for others.
    """
    manifest = load_manifest(path)
    manifest[output_filename(meta.registry_entry)] = asdict(meta)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
