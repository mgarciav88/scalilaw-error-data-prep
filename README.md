# scalilaw-error-data-prep

WandB/HF extraction pipeline that produces clean, versioned, per-step training-loss
datasets for Pythia and PolyPythias.

**Scope.** This repo's only job is turning EleutherAI's public WandB logs into a tidy,
provenance-tracked dataset. It contains code and a run registry only; the data itself
is not included and is produced locally by the commands below.

## Setup

```bash
uv sync
cp .env.example .env   # fill in WANDB_API_KEY (needs read access to the eleutherai entity)
```

## CLI

```bash
uv run scalilaw-data discover      # resolve the WandB group registry -> registry/polypythias.yaml
uv run scalilaw-data extract --model 14m --seed 1 [--variant joint|data_seed|weight_seed]
uv run scalilaw-data extract-all [--include-ambiguous]   # extract every CLEAN (+ optionally AMBIGUOUS) entry
```

`discover` talks to WandB (and, to disambiguate a handful of PolyPythias seeds, the
HF Hub) and can take several minutes — it logs progress as it goes. It's always safe
to re-run; it fully overwrites `registry/polypythias.yaml` from scratch.

## How it fits together

```
discover  ->  registry/polypythias.yaml   (which WandB run is canonical per model/seed/variant)
extract   ->  data/<model>_seed<N>[_<variant>].parquet   (the tidy per-step series)
          ->  data/manifest.json          (provenance + coverage stats per parquet file)
```

- **Registry** (`registry/polypythias.yaml`, committed): one entry per
  (model_size, seed, variant), each tagged `clean`, `ambiguous`, or `unresolved` — see
  `src/scalilaw_data/discovery.py`'s module docstring for exactly how each case was
  resolved (or why it couldn't be). Ambiguous/unresolved entries are expected and
  flagged loudly rather than guessed at; nothing here is silently dropped.
- **Parquet schema**: `step`, `tokens_seen`, `train_lm_loss`, `validation_lm_loss` —
  one row per optimizer step, numeric only (see `schema.py`). Provenance (which
  WandB run, dedup status, resolution confidence) deliberately lives in the manifest,
  not the parquet, so the parquet stays a clean time series.
- **Manifest** (`data/manifest.json`, gitignored): one JSON entry per parquet file,
  keyed by filename — the registry entry that produced it plus coverage stats
  (step range, gap count) observed at extraction time.
- `data/raw/` (gitignored): cached raw API dumps kept around for offline
  re-analysis when the WandB naming archaeology needs re-checking.

## Current status (PolyPythias)

Run `uv run scalilaw-data discover` for the live picture; as of the last run: **33/51
clean**, 0 ambiguous, 18 unresolved. The unresolved cases are genuine gaps, not bugs:

- **31m**: zero WandB groups found under any naming pattern, any state, for any seed.
- **160m**: seeds 1-3 (joint) and all 6 decoupled `data_seed`/`weight_seed` variants
  have no matching WandB groups under any convention tried.

Both are pending a reply from EleutherAI. Everything else (14m ×9, 70m ×9, 160m ×6,
410m ×9) is clean and already extracted.

## Testing

```bash
uv run pytest              # offline unit tests (fixtures + mocks, no network)
uv run pytest -m live      # + live smoke tests against the real WandB/HF APIs
```

## Repo layout

```
registry/polypythias.yaml   # committed — the reviewable WandB-group-to-model mapping
src/scalilaw_data/
  config.py       # paths, entity/project names, token math, .env loading
  wandb_client.py # thin authenticated wandb wrapper (scan_history, never history())
  hf_client.py    # HF Hub repo-creation-timestamp lookup (for ambiguity resolution)
  discovery.py    # enumerate WandB groups, resolve canonical run per (model, seed, variant)
  registry.py     # read/write registry/polypythias.yaml
  schema.py       # canonical parquet columns + RegistryEntry/RunMetadata dataclasses
  extract.py      # WandB history -> tidy DataFrame
  validate.py     # coverage/hygiene checks, fail loud
  manifest.py     # data/manifest.json read/write
  cli.py          # discover | extract | extract-all
tests/            # unit tests (offline) + live-marked smoke tests
data/             # gitignored — parquet output + manifest.json + raw/ dumps
```
