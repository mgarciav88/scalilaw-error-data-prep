"""Command-line entry points: ``scalilaw-data discover|extract|extract-all``."""

import logging
from datetime import UTC, datetime
from pathlib import Path

import typer

from scalilaw_data import discovery, manifest, registry
from scalilaw_data.config import DATA_DIR
from scalilaw_data.extract import extract_run
from scalilaw_data.schema import RegistryEntry, ResolutionStatus, RunMetadata, Variant
from scalilaw_data.validate import validate_run

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    """scalilaw-data: WandB extraction pipeline for Pythia/PolyPythias loss data.

    An explicit callback (even a no-op) keeps subcommand names required even while
    only one command (``discover``) is registered -- Typer otherwise collapses a
    single-command app into a flat, subcommand-less CLI. Also configures progress
    logging here, once, before any command runs -- some commands (discover) talk to
    WandB/HF for minutes with nothing else to show for it otherwise.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")


@app.command()
def discover() -> None:
    """Resolve the PolyPythias WandB group registry and write registry/polypythias.yaml.

    Safe to re-run any time -- always talks to WandB fresh and overwrites the
    registry file. AMBIGUOUS/UNRESOLVED entries are expected (see the module
    docstrings in discovery.py for the confirmed archaeology); this command flags
    them loudly rather than blocking.
    """
    entries = discovery.build_registry()
    registry.write_registry(entries)
    typer.echo(f"Wrote {registry.POLYPYTHIAS_REGISTRY_PATH}")
    typer.echo(discovery.format_report(entries))


@app.command()
def extract(
    model: str = typer.Option(..., "--model", help="Model size, e.g. 14m, 70m, 160m, 410m"),
    seed: int = typer.Option(..., "--seed", help="Seed number"),
    variant: str = typer.Option(
        "joint", "--variant", help="joint | data_seed | weight_seed"
    ),
) -> None:
    """Pull one registry entry's loss history, validate it, and write parquet + manifest.

    Requires ``discover`` to have been run first -- looks the (model, seed, variant)
    up in registry/polypythias.yaml rather than talking to WandB group names directly.
    """
    entry = _find_entry(model, seed, variant)
    path = _extract_one(entry)
    typer.echo(f"Wrote {path}")


@app.command(name="extract-all")
def extract_all(
    include_ambiguous: bool = typer.Option(
        False, help="Also extract AMBIGUOUS entries (uses discovery's best-guess run)."
    ),
) -> None:
    """Extract every CLEAN registry entry (optionally AMBIGUOUS too); UNRESOLVED is always skipped.

    Continues past per-entry failures and reports a pass/fail summary at the end,
    rather than aborting the whole batch on the first problem.
    """
    entries = registry.read_registry()
    wanted_statuses = {ResolutionStatus.CLEAN}
    if include_ambiguous:
        wanted_statuses.add(ResolutionStatus.AMBIGUOUS)
    targets = [e for e in entries if e.resolution_status in wanted_statuses]

    succeeded: list[str] = []
    failed: list[tuple[str, str]] = []
    for entry in targets:
        label = f"{entry.model_size} seed={entry.seed} [{entry.variant.value}]"
        try:
            path = _extract_one(entry)
            typer.echo(f"OK   {label} -> {path}")
            succeeded.append(label)
        except Exception as exc:  # noqa: BLE001 -- batch job: record and keep going
            typer.echo(f"FAIL {label}: {exc}", err=True)
            failed.append((label, str(exc)))

    typer.echo(f"\n{len(succeeded)} succeeded, {len(failed)} failed out of {len(targets)}.")
    if failed:
        raise typer.Exit(code=1)


def _find_entry(model: str, seed: int, variant: str) -> RegistryEntry:
    entries = registry.read_registry()
    variant_enum = Variant(variant)
    matches = [
        e for e in entries if e.model_size == model and e.seed == seed and e.variant == variant_enum
    ]
    if not matches:
        typer.echo(
            f"No registry entry for {model} seed={seed} [{variant}]. Run `discover` first?",
            err=True,
        )
        raise typer.Exit(code=1)
    entry = matches[0]
    if entry.resolution_status == ResolutionStatus.UNRESOLVED:
        typer.echo(f"{model} seed={seed} [{variant}] is UNRESOLVED: {entry.notes}", err=True)
        raise typer.Exit(code=1)
    if entry.resolution_status == ResolutionStatus.AMBIGUOUS:
        typer.echo(f"WARNING: {model} seed={seed} [{variant}] is AMBIGUOUS -- {entry.notes}")
    return entry


def _extract_one(entry: RegistryEntry) -> Path:
    df = extract_run(entry)
    stats = validate_run(df, entry)

    out_path = DATA_DIR / manifest.output_filename(entry)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    meta = RunMetadata(
        registry_entry=entry,
        step_min=stats.step_min,
        step_max=stats.step_max,
        n_rows=stats.n_rows,
        n_gaps=stats.n_gaps,
        extracted_at=datetime.now(UTC).isoformat(),
    )
    manifest.record(meta)
    return out_path


if __name__ == "__main__":
    app()