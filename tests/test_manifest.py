from datetime import UTC, datetime

from scalilaw_data import manifest
from scalilaw_data.schema import RegistryEntry, ResolutionStatus, RunMetadata, Variant


def _entry(variant: Variant = Variant.JOINT) -> RegistryEntry:
    return RegistryEntry(
        model_size="14m",
        seed=1,
        variant=variant,
        dedup=False,
        wandb_project="pythia-extra-seeds",
        wandb_group="g",
        wandb_run_id="r",
        wandb_run_name="ip-1-2-3-4-0",
        resolution_status=ResolutionStatus.CLEAN,
        notes="",
    )


def test_output_filename_joint():
    assert manifest.output_filename(_entry()) == "14m_seed1.parquet"


def test_output_filename_variant_suffix():
    assert manifest.output_filename(_entry(Variant.DATA_SEED)) == "14m_seed1_data_seed.parquet"


def test_record_and_load_round_trip(tmp_path):
    path = tmp_path / "manifest.json"
    meta = RunMetadata(
        registry_entry=_entry(),
        step_min=1,
        step_max=143000,
        n_rows=143000,
        n_gaps=0,
        extracted_at=datetime.now(UTC).isoformat(),
    )
    manifest.record(meta, path=path)
    loaded = manifest.load_manifest(path=path)
    assert "14m_seed1.parquet" in loaded
    assert loaded["14m_seed1.parquet"]["n_rows"] == 143000
    assert loaded["14m_seed1.parquet"]["registry_entry"]["model_size"] == "14m"


def test_record_merges_without_clobbering_other_entries(tmp_path):
    path = tmp_path / "manifest.json"
    meta1 = RunMetadata(
        registry_entry=_entry(), step_min=1, step_max=100, n_rows=100, n_gaps=0, extracted_at="t1"
    )
    meta2 = RunMetadata(
        registry_entry=_entry(Variant.WEIGHT_SEED),
        step_min=1,
        step_max=200,
        n_rows=200,
        n_gaps=0,
        extracted_at="t2",
    )
    manifest.record(meta1, path=path)
    manifest.record(meta2, path=path)
    loaded = manifest.load_manifest(path=path)
    assert set(loaded.keys()) == {"14m_seed1.parquet", "14m_seed1_weight_seed.parquet"}


def test_load_manifest_missing_file_returns_empty_dict(tmp_path):
    assert manifest.load_manifest(path=tmp_path / "nonexistent.json") == {}
