from datetime import UTC, datetime

import pytest
from conftest import FakeRun

from scalilaw_data import discovery, hf_client
from scalilaw_data.schema import Variant


@pytest.mark.parametrize(
    "name,expected_size",
    [
        ("pythia-70m_ks5ej680", "70m"),
        ("pythia-70m_34jd67yh_gjzj89ao", "70m"),  # double-hash: still a valid candidate
        ("pythia-410m_xhekvdjj", "410m"),
    ],
)
def test_group_re_matches_canonical_forms(name, expected_size):
    match = discovery._GROUP_RE.match(name)
    assert match is not None
    assert match.group("size") == expected_size


@pytest.mark.parametrize(
    "name",
    [
        "pythia-14m-warmup01_6kcnj0ed",  # suffixed ablation, not the standard suite
        "3tNhfe2MJxifx4S9urj5mf_12wslca5",  # unrelated workspace noise
    ],
)
def test_group_re_excludes_non_canonical_forms(name):
    assert discovery._GROUP_RE.match(name) is None


def test_find_populated_shard_returns_the_logged_one():
    empty = FakeRun(id="empty", summary={})
    populated = FakeRun(id="populated", summary={"train/lm_loss": 3.5})
    assert discovery.find_populated_shard([empty, populated]) is populated


def test_find_populated_shard_none_when_all_empty():
    runs = [FakeRun(id="a", summary={}), FakeRun(id="b", summary={})]
    assert discovery.find_populated_shard(runs) is None


def test_infer_dedup_true_for_deduped_path():
    run = FakeRun(
        id="r",
        config={"train_data_paths": ["/mnt/ssd-2/pile_deduped/pile_20B_tokenizer_text_document"]},
    )
    assert discovery.infer_dedup(run) is True


def test_infer_dedup_false_for_standard_path():
    run = FakeRun(
        id="r", config={"train_data_paths": ["/fsx/pile/pile_20B_tokenizer_text_document"]}
    )
    assert discovery.infer_dedup(run) is False


def test_infer_dedup_none_when_path_missing():
    assert discovery.infer_dedup(FakeRun(id="r", config={})) is None


def test_hf_resolution_picks_closest_preceding_candidate(monkeypatch):
    monkeypatch.setattr(
        hf_client,
        "get_repo_created_at",
        lambda repo_id: datetime(2023, 11, 8, 13, 25, 14, tzinfo=UTC),
    )
    early = discovery.GroupInfo(name="early", size="70m", seed=1, runs=[])
    late = discovery.GroupInfo(name="late", size="70m", seed=1, runs=[])
    early_run = FakeRun(id="early_run", heartbeatAt="2023-10-31T10:34:28Z")
    late_run = FakeRun(id="late_run", heartbeatAt="2023-11-08T03:06:06Z")

    result = discovery._try_hf_resolution(
        "70m", 1, Variant.JOINT, [(early, early_run), (late, late_run)]
    )

    assert result is not None
    chosen, shard, note = result
    assert chosen.name == "late"
    assert shard is late_run


def test_hf_resolution_returns_none_if_all_candidates_after_creation(monkeypatch):
    monkeypatch.setattr(
        hf_client, "get_repo_created_at", lambda repo_id: datetime(2023, 1, 1, tzinfo=UTC)
    )
    g = discovery.GroupInfo(name="g", size="70m", seed=1, runs=[])
    run = FakeRun(id="r", heartbeatAt="2024-01-01T00:00:00Z")
    assert discovery._try_hf_resolution("70m", 1, Variant.JOINT, [(g, run)]) is None


def test_hf_resolution_returns_none_for_decoupled_variant():
    assert discovery._try_hf_resolution("160m", 1, Variant.DATA_SEED, []) is None


def test_hf_resolution_returns_none_on_lookup_failure(monkeypatch):
    def boom(repo_id: str) -> datetime:
        raise RuntimeError("404")

    monkeypatch.setattr(hf_client, "get_repo_created_at", boom)
    g = discovery.GroupInfo(name="g", size="70m", seed=1, runs=[])
    run = FakeRun(id="r", heartbeatAt="2023-01-01T00:00:00Z")
    assert discovery._try_hf_resolution("70m", 1, Variant.JOINT, [(g, run)]) is None
