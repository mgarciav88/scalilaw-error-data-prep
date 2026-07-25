"""Thin wrapper over the Hugging Face Hub API: repo creation timestamp lookup.

Used by discovery.py to disambiguate PolyPythias seeds with more than one
independently finished WandB run (the 70m case): a released HF checkpoint can only
have been produced by a run that finished BEFORE the HF repo was created, so
cross-referencing narrows candidates that WandB metadata alone can't distinguish.
"""

from datetime import datetime
from functools import cache

from huggingface_hub import HfApi

_api = HfApi()


@cache
def get_repo_created_at(repo_id: str) -> datetime:
    """Return the HF repo's creation timestamp (tz-aware UTC).

    Raises whatever huggingface_hub raises for a missing/inaccessible repo (e.g.
    RepositoryNotFoundError) -- callers decide whether that's fatal or just means
    this particular cross-reference can't be attempted.
    """
    info = _api.model_info(repo_id)
    created = info.created_at
    if created is None:
        raise ValueError(f"{repo_id}: HF API response had no created_at field")
    return created
