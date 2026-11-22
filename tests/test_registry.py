from scalilaw_data import registry
from scalilaw_data.schema import RegistryEntry, ResolutionStatus, Variant


def _entries() -> list[RegistryEntry]:
    return [
        RegistryEntry(
            model_size="14m",
            seed=1,
            variant=Variant.JOINT,
            dedup=False,
            wandb_project="pythia-extra-seeds",
            wandb_group="g1",
            wandb_run_id="r1",
            wandb_run_name="ip-26-0-144-150-0",
            resolution_status=ResolutionStatus.CLEAN,
            notes="",
        ),
        RegistryEntry(
            model_size="31m",
            seed=1,
            variant=Variant.JOINT,
            dedup=None,
            wandb_project="pythia-extra-seeds",
            wandb_group="",
            wandb_run_id=None,
            wandb_run_name=None,
            resolution_status=ResolutionStatus.UNRESOLVED,
            notes="No finished group found.",
        ),
    ]


def _sort_key(e: RegistryEntry) -> tuple[str, int, str]:
    return (e.model_size, e.seed, e.variant.value)


def test_round_trip(tmp_path):
    path = tmp_path / "registry.yaml"
    original = _entries()
    registry.write_registry(original, path=path)
    loaded = registry.read_registry(path=path)
    assert sorted(loaded, key=_sort_key) == sorted(original, key=_sort_key)


def test_write_sorts_by_size_seed_variant(tmp_path):
    path = tmp_path / "registry.yaml"
    registry.write_registry(list(reversed(_entries())), path=path)
    loaded = registry.read_registry(path=path)
    assert [e.model_size for e in loaded] == ["14m", "31m"]


def test_none_dedup_and_run_id_round_trip(tmp_path):
    path = tmp_path / "registry.yaml"
    registry.write_registry(_entries(), path=path)
    loaded = registry.read_registry(path=path)
    unresolved = next(e for e in loaded if e.resolution_status == ResolutionStatus.UNRESOLVED)
    assert unresolved.dedup is None
    assert unresolved.wandb_run_id is None
    assert unresolved.wandb_run_name is None


def test_wandb_run_name_round_trips(tmp_path):
    path = tmp_path / "registry.yaml"
    registry.write_registry(_entries(), path=path)
    loaded = registry.read_registry(path=path)
    clean = next(e for e in loaded if e.resolution_status == ResolutionStatus.CLEAN)
    assert clean.wandb_run_name == "ip-26-0-144-150-0"
