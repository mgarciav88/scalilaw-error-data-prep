"""Minimal live smoke tests against the real WandB/HF APIs.

Skipped by default (pyproject's `addopts = "-m 'not live'"`); run explicitly with
`uv run pytest -m live`. Requires WANDB_API_KEY in .env and a populated
registry/polypythias.yaml (from `scalilaw-data discover`).
"""

import pytest

from scalilaw_data import registry
from scalilaw_data.config import POLYPYTHIAS_PROJECT, WANDB_ENTITY
from scalilaw_data.extract import extract_run
from scalilaw_data.wandb_client import list_runs


@pytest.mark.live
def test_list_runs_returns_data():
    first = next(iter(list_runs(WANDB_ENTITY, POLYPYTHIAS_PROJECT)))
    assert first.id


@pytest.mark.live
def test_extract_known_clean_entry():
    entries = registry.read_registry()
    entry = next(e for e in entries if e.model_size == "14m" and e.seed == 1)
    df = extract_run(entry)
    assert len(df) > 0
    assert df["step"].is_monotonic_increasing
