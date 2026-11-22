"""Read/write the PolyPythias registry (registry/polypythias.yaml).

The registry is a reviewable, checked-in artifact -- the fragile WandB-group-to-model
mapping that ``discover`` resolves, made inspectable so humans can review diffs in
version control and correct AMBIGUOUS/UNRESOLVED entries by hand rather than trusting
a black-box script silently.
"""

from pathlib import Path
from typing import Any

import yaml

from scalilaw_data.config import REGISTRY_DIR
from scalilaw_data.schema import RegistryEntry, ResolutionStatus, Variant

POLYPYTHIAS_REGISTRY_PATH = REGISTRY_DIR / "polypythias.yaml"


def entry_to_dict(entry: RegistryEntry) -> dict[str, Any]:
    return {
        "model_size": entry.model_size,
        "seed": entry.seed,
        "variant": entry.variant.value,
        "dedup": entry.dedup,
        "wandb_project": entry.wandb_project,
        "wandb_group": entry.wandb_group,
        "wandb_run_id": entry.wandb_run_id,
        "wandb_run_name": entry.wandb_run_name,
        "resolution_status": entry.resolution_status.value,
        "notes": entry.notes,
    }


def dict_to_entry(d: dict[str, Any]) -> RegistryEntry:
    return RegistryEntry(
        model_size=d["model_size"],
        seed=d["seed"],
        variant=Variant(d["variant"]),
        dedup=d["dedup"],
        wandb_project=d["wandb_project"],
        wandb_group=d["wandb_group"],
        wandb_run_id=d["wandb_run_id"],
        wandb_run_name=d["wandb_run_name"],
        resolution_status=ResolutionStatus(d["resolution_status"]),
        notes=d.get("notes", ""),
    )


def write_registry(entries: list[RegistryEntry], path: Path = POLYPYTHIAS_REGISTRY_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sorted_entries = sorted(entries, key=lambda e: (e.model_size, e.seed, e.variant.value))
    with path.open("w") as f:
        yaml.safe_dump(
            [entry_to_dict(e) for e in sorted_entries],
            f,
            sort_keys=False,
            default_flow_style=False,
        )


def read_registry(path: Path = POLYPYTHIAS_REGISTRY_PATH) -> list[RegistryEntry]:
    with path.open() as f:
        raw = yaml.safe_load(f) or []
    return [dict_to_entry(d) for d in raw]
