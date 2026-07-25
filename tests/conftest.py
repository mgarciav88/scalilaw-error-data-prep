from dataclasses import dataclass, field
from typing import Any

import pytest


@dataclass
class FakeRun:
    id: str
    state: str = "finished"
    group: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    heartbeatAt: str | None = None
    created_at: str | None = None


@pytest.fixture
def sample_history_rows() -> list[dict[str, Any]]:
    """Rows shaped like real scan_history() output (see the 14m seed=1 pull)."""
    return [
        {"_step": 1, "train/lm_loss": 11.0138989},
        {"_step": 2, "train/lm_loss": 11.0166872},
        {"_step": 3, "train/lm_loss": 11.0147257},
        {"_step": 143000, "train/lm_loss": 3.586955, "validation/lm_loss": 3.58689},
    ]
