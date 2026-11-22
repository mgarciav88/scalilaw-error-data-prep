"""Thin wrapper over the wandb API: auth, run listing, and safe history scanning.

Centralizes the two lessons from the extraction notes (see
pythia_data_extraction_notes.md §3): every read must go through ``scan_history``,
never ``history()`` (which silently downsamples to ~500 rows by default -- this
produced a misleading result early on, where a run's train/lm_loss appeared to start
at step 47001 purely because of history()'s pagination limit, before scan_history()
gave the true picture); and the wandb API requires a key even for public projects
(anonymous access is deprecated), so auth happens explicitly here and fails loudly
rather than falling through to a confusing 401 deep in a library call.
"""

from collections.abc import Iterator
from functools import lru_cache
from typing import Any

import wandb
from wandb.apis.public import Run

from scalilaw_data.config import load_wandb_api_key


@lru_cache(maxsize=1)
def get_api() -> wandb.Api:
    """Return an authenticated wandb.Api, memoized for the process lifetime."""
    key = load_wandb_api_key()
    return wandb.Api(api_key=key)


def list_runs(
    entity: str, project: str, filters: dict[str, Any] | None = None
) -> Iterator[Run]:
    """Yield runs in a project, optionally filtered (mongo-style, e.g. {"state": "finished"}).

    Both eleutherai/pythia and eleutherai/pythia-extra-seeds are shared workspaces
    with runs unrelated to Pythia -- callers must additionally filter by group-name
    prefix (see discovery.py) rather than assume everything returned here is relevant.
    """
    api = get_api()
    yield from api.runs(f"{entity}/{project}", filters=filters)


def get_run(entity: str, project: str, run_id: str) -> Run:
    """Fetch a single run by id, e.g. for re-extracting a previously-resolved run."""
    api = get_api()
    # wandb ships no type stubs, so mypy can't see through api.run()'s return type.
    return api.run(f"{entity}/{project}/{run_id}")  # type: ignore[no-any-return,no-untyped-call]


def scan_run_history(run: Run, keys: list[str] | None = None) -> Iterator[dict[str, Any]]:
    """Yield every logged row for a run via scan_history -- never history().

    run.history() defaults to a downsampled ~500 rows regardless of how many steps
    were actually logged, which silently produces incomplete data with no error.
    scan_history() streams the full, ungapped log. This function exists so that trap
    can only be hit once, here, instead of at every call site.
    """
    yield from run.scan_history(keys=keys)
