"""Enumerate PolyPythias WandB groups and resolve the canonical run per (model, seed).

Every pattern encoded here was confirmed against a full census of
eleutherai/pythia-extra-seeds (1,161 runs, 276 groups), not assumed from the notes:

- Canonical PolyPythias group names are exactly ``pythia-<size>m_<opaque-hash>``,
  where ``<opaque-hash>`` may itself contain underscores -- a subset of 70m groups
  have a double-hash form (e.g. "pythia-70m_ks5ej680" vs
  "pythia-70m_34jd67yh_gjzj89ao"), and both are valid, independent finished
  candidates, not a naming bug. ``_GROUP_RE`` captures everything after the size as
  one opaque group, so this needs no special-casing.
- This pattern deliberately excludes two other group shapes sharing the project and
  the "pythia-" prefix: suffixed ablations, e.g. "pythia-14m-warmup01_<hash>" (a
  warmup-schedule experiment, unrelated to the standard 9-seed suite -- the hyphen
  breaks the `\\d+m_` match, so it's excluded for free); and ~80 unrelated groups in
  the same shared workspace with non-"pythia-"-prefixed hash names.
- Only one of a group's ~4 sub-runs actually has logged history (the rest are
  0-row shards from the same distributed job). ``run.summary`` -- already fetched
  with the run object, no extra API call -- reliably flags which one: the populated
  shard has the target metric key in its summary, the others don't.
- ``run.config['seed']`` is real and populated. A `data_seed` config key exists on
  many runs but was always None in the census -- it's an unrelated framework field,
  NOT the 160M decoupled-seed marker. No decoupled variant (data_seed/weight_seed)
  runs were found under any naming pattern as of this census.
- 70m has 2-3 independently 'finished' candidate groups for every one of its 9 seeds
  -- confirmed genuine via full census, not a partial-duplicate artifact. Not
  resolvable from WandB metadata alone -- resolved instead by cross-referencing each
  candidate's last WandB activity (``heartbeatAt``) against the HF-released
  checkpoint's own creation timestamp (see ``_try_hf_resolution``): a checkpoint can
  only have been produced by a run that finished BEFORE its HF repo was created, so
  the viable candidate whose activity most closely precedes that timestamp is taken
  as canonical. Held up across all 9 seeds checked (8 with a clear multi-hour-to-day
  margin; seed 9 was the closest call, ~7 hours). If no candidate precedes the HF
  timestamp (or the HF lookup itself fails), falls back to AMBIGUOUS rather than
  guessing.
- 31m has zero groups of any state under any related naming pattern. 160m seeds 1-3
  are the same story. Both are marked UNRESOLVED, not guessed at.
- dedup status is NOT uniform across the project: most runs train on non-deduped Pile
  (".../pile/pile_20B_tokenizer_text_document"), but a real subset (so far, entirely
  within 'failed' groups) trains on "/mnt/ssd-2/pile_deduped/...". Never hardcode
  dedup -- always read it off the resolved run's own config.
"""

import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

from wandb.apis.public import Run

from scalilaw_data import hf_client
from scalilaw_data.config import HF_SEED_REPO_TEMPLATE, POLYPYTHIAS_PROJECT, WANDB_ENTITY
from scalilaw_data.schema import RegistryEntry, ResolutionStatus, Variant
from scalilaw_data.wandb_client import list_runs

logger = logging.getLogger(__name__)

_GROUP_RE = re.compile(r"^pythia-(?P<size>\d+m)_(?P<hash>[0-9a-zA-Z_]+)$")
_TARGET_METRIC = "train/lm_loss"
_PROGRESS_EVERY = 200  # log a liveness marker every N runs scanned in enumerate_groups

# The standard (non-decoupled) PolyPythias grid: every size x every seed 1-9.
EXPECTED_SEEDS = tuple(range(1, 10))
EXPECTED_SIZES = ("14m", "31m", "70m", "160m", "410m")

# 160M decoupled study, per the HF collection: 3 data-seed + 3 weight-seed runs.
DECOUPLED_SEEDS = (1, 2, 3)


@dataclass
class GroupInfo:
    """One WandB group, parsed and seed-tagged, with its raw sub-runs attached."""

    name: str
    size: str
    seed: int | None
    runs: list[Run]

    @property
    def states(self) -> set[str]:
        return {r.state for r in self.runs}


def enumerate_groups(
    entity: str = WANDB_ENTITY, project: str = POLYPYTHIAS_PROJECT
) -> list[GroupInfo]:
    """List every canonical PolyPythias group in the project, parsed and seed-tagged.

    Non-canonical groups (warmup ablations, unrelated workspace noise) don't match
    ``_GROUP_RE`` and are silently skipped -- they're a different naming pattern by
    construction, not an edge case of this one.
    """
    logger.info(
        "Enumerating runs in %s/%s (this reads every run's config; takes a few minutes)...",
        entity,
        project,
    )
    by_group: dict[str, list[Run]] = defaultdict(list)
    n_scanned = 0
    for run in list_runs(entity, project):
        n_scanned += 1
        if n_scanned % _PROGRESS_EVERY == 0:
            logger.info(
                "  ...%d runs scanned, %d canonical groups so far", n_scanned, len(by_group)
            )
        if run.group and _GROUP_RE.match(run.group):
            by_group[run.group].append(run)
    logger.info(
        "Scanned %d runs total -> %d canonical PolyPythias groups", n_scanned, len(by_group)
    )

    infos = []
    for name, runs in by_group.items():
        match = _GROUP_RE.match(name)
        assert match is not None  # guaranteed by the filter above
        seeds = {r.config.get("seed") for r in runs if r.config.get("seed") is not None}
        seed = seeds.pop() if len(seeds) == 1 else None
        infos.append(GroupInfo(name=name, size=match.group("size"), seed=seed, runs=runs))
    return infos


def find_populated_shard(runs: list[Run], metric_key: str = _TARGET_METRIC) -> Run | None:
    """Return the one sub-run in a group whose summary shows it actually logged data.

    Checked via ``run.summary`` (already fetched with the run object) rather than
    ``scan_history``, since discovery only needs to know WHICH run to extract later,
    not its full content -- keeps discovery cheap across hundreds of groups.
    """
    for run in runs:
        if metric_key in run.summary:
            return run
    return None


def infer_dedup(run: Run) -> bool | None:
    """Read dedup status off the run's own training data path -- never hardcode it.

    Returns None (unconfirmed) if the run's config doesn't carry a data path at all,
    which happens for some sub-runs; callers should treat None as "needs manual
    check," not as a default of False.
    """
    paths = run.config.get("train_data_paths")
    if not paths:
        return None
    return any("dedup" in str(p).lower() for p in paths)


def build_registry(
    entity: str = WANDB_ENTITY, project: str = POLYPYTHIAS_PROJECT
) -> list[RegistryEntry]:
    """Resolve one RegistryEntry per (model_size, seed) in the full PolyPythias grid.

    For each expected (size, seed) in the standard suite, candidates are first
    filtered to groups with an actual populated shard (see ``find_populated_shard``)
    -- WandB's 'finished' state does NOT guarantee any sub-run logged data; a whole
    group can be marked finished with every sub-run's summary empty. Counting those
    as real candidates would silently pick a data-less run. Among the viable
    candidates:
      - 0 -> UNRESOLVED, no wandb_run_id.
      - 1 -> CLEAN, resolved to its populated shard.
      - >1 -> AMBIGUOUS, resolved to the first as a placeholder, with every candidate
        listed in ``notes`` for human review.
    Also emits UNRESOLVED placeholders for the six 160M decoupled data-seed/
    weight-seed runs, since no matching groups exist under any naming pattern found
    so far.
    """
    groups = enumerate_groups(entity, project)
    finished_by_key: dict[tuple[str, int], list[GroupInfo]] = defaultdict(list)
    for g in groups:
        if g.seed is not None and "finished" in g.states:
            finished_by_key[(g.size, g.seed)].append(g)

    logger.info(
        "Resolving %d (size, seed) combinations...", len(EXPECTED_SIZES) * len(EXPECTED_SEEDS)
    )
    entries: list[RegistryEntry] = []
    for size in EXPECTED_SIZES:
        for seed in EXPECTED_SEEDS:
            candidates = finished_by_key.get((size, seed), [])
            entries.append(_resolve_entry(size, seed, Variant.JOINT, project, candidates))

    for seed in DECOUPLED_SEEDS:
        for variant in (Variant.DATA_SEED, Variant.WEIGHT_SEED):
            entries.append(
                _unresolved_entry(
                    "160m",
                    seed,
                    variant,
                    project,
                    "No group found under any known naming pattern in this census.",
                )
            )
    return entries


def _resolve_entry(
    size: str, seed: int, variant: Variant, project: str, candidates: list[GroupInfo]
) -> RegistryEntry:
    logger.info("%s seed=%d [%s]: %d finished group(s)", size, seed, variant.value, len(candidates))
    viable: list[tuple[GroupInfo, Run]] = []
    for g in candidates:
        shard = find_populated_shard(g.runs)
        if shard is not None:
            viable.append((g, shard))

    if not viable:
        note = (
            "No finished group with a populated shard found "
            f"({len(candidates)} finished group(s) checked, all data-less)."
            if candidates
            else "No finished group found."
        )
        return _unresolved_entry(size, seed, variant, project, note)

    if len(viable) == 1:
        chosen, shard = viable[0]
        return RegistryEntry(
            model_size=size,
            seed=seed,
            variant=variant,
            dedup=infer_dedup(shard),
            wandb_project=project,
            wandb_group=chosen.name,
            wandb_run_id=shard.id,
            wandb_run_name=shard.name,
            resolution_status=ResolutionStatus.CLEAN,
            notes="",
        )

    resolved = _try_hf_resolution(size, seed, variant, viable)
    if resolved is not None:
        chosen, shard, notes = resolved
        return RegistryEntry(
            model_size=size,
            seed=seed,
            variant=variant,
            dedup=infer_dedup(shard),
            wandb_project=project,
            wandb_group=chosen.name,
            wandb_run_id=shard.id,
            wandb_run_name=shard.name,
            resolution_status=ResolutionStatus.CLEAN,
            notes=notes,
        )

    chosen, shard = viable[0]
    others = ", ".join(g.name for g, _ in viable)
    notes = (
        f"{len(viable)} independently finished groups with real data for this "
        f"seed: {others}. HF cross-reference did not resolve this (repo missing, "
        f"lookup failed, or no candidate precedes the HF repo's creation). "
        f"Defaulted to {chosen.name}; needs manual review before use."
    )
    return RegistryEntry(
        model_size=size,
        seed=seed,
        variant=variant,
        dedup=infer_dedup(shard),
        wandb_project=project,
        wandb_group=chosen.name,
        wandb_run_id=shard.id,
        wandb_run_name=shard.name,
        resolution_status=ResolutionStatus.AMBIGUOUS,
        notes=notes,
    )


def _try_hf_resolution(
    size: str, seed: int, variant: Variant, viable: list[tuple[GroupInfo, Run]]
) -> tuple[GroupInfo, Run, str] | None:
    """Disambiguate via the released HF checkpoint's creation timestamp.

    A checkpoint upload can only follow the run that produced it, so among the
    viable candidates, the one whose last WandB activity (``heartbeatAt``) most
    closely PRECEDES the HF repo's creation is taken as canonical. Returns None
    (never guesses) if: no HF repo naming is known for this variant, the HF lookup
    fails (repo missing, network error), or every candidate's activity is AFTER the
    HF repo's creation (rules out this heuristic entirely, doesn't just weaken it).
    """
    if variant != Variant.JOINT:
        return None  # no confirmed HF repo naming convention for decoupled variants

    repo_id = HF_SEED_REPO_TEMPLATE.format(size=size, seed=seed)
    logger.info(
        "%s seed=%d: %d viable candidates, cross-referencing HF repo %s...",
        size,
        seed,
        len(viable),
        repo_id,
    )
    try:
        hf_created = hf_client.get_repo_created_at(repo_id)
    except Exception:  # noqa: BLE001 -- best-effort external lookup, never fatal
        logger.info("  HF lookup failed for %s; falling back to AMBIGUOUS", repo_id)
        return None
    logger.info("  %s created %s", repo_id, hf_created.isoformat())

    dated: list[tuple[GroupInfo, Run, datetime]] = []
    for g, r in viable:
        heartbeat = getattr(r, "heartbeatAt", None)
        if heartbeat:
            dated.append((g, r, datetime.fromisoformat(heartbeat)))

    preceding = [(g, r, end) for g, r, end in dated if end < hf_created]
    if not preceding:
        logger.info("  no candidate precedes %s; falling back to AMBIGUOUS", repo_id)
        return None

    chosen, shard, end = max(preceding, key=lambda item: item[2])
    gap = hf_created - end
    rejected = ", ".join(g.name for g, _ in viable if g.name != chosen.name)
    logger.info("  resolved: %s (%s before HF creation)", chosen.name, gap)
    note = (
        f"Resolved via HF cross-reference against {repo_id}: chosen run's last "
        f"WandB activity was {end.isoformat()}, {gap} before the HF repo's creation "
        f"({hf_created.isoformat()}). Rejected candidate(s): {rejected}."
    )
    return chosen, shard, note


def _unresolved_entry(
    size: str, seed: int, variant: Variant, project: str, note: str
) -> RegistryEntry:
    return RegistryEntry(
        model_size=size,
        seed=seed,
        variant=variant,
        dedup=None,
        wandb_project=project,
        wandb_group="",
        wandb_run_id=None,
        wandb_run_name=None,
        resolution_status=ResolutionStatus.UNRESOLVED,
        notes=note,
    )


def format_report(entries: list[RegistryEntry]) -> str:
    """Human-readable summary of what discovery resolved, flagging gaps loudly.

    Printed by ``scalilaw-data discover`` after writing the registry -- the registry
    file itself is the source of truth, this is just a scan-friendly digest so
    AMBIGUOUS/UNRESOLVED cases aren't buried in a 51-row YAML file.
    """
    by_status: dict[ResolutionStatus, list[RegistryEntry]] = defaultdict(list)
    for e in entries:
        by_status[e.resolution_status].append(e)

    lines = [
        f"Resolved {len(entries)} registry entries "
        f"({len(by_status[ResolutionStatus.CLEAN])} clean, "
        f"{len(by_status[ResolutionStatus.AMBIGUOUS])} ambiguous, "
        f"{len(by_status[ResolutionStatus.UNRESOLVED])} unresolved).",
    ]
    for status in (ResolutionStatus.AMBIGUOUS, ResolutionStatus.UNRESOLVED):
        flagged = by_status[status]
        if not flagged:
            continue
        lines.append(f"\n{status.value.upper()} ({len(flagged)}):")
        for e in flagged:
            lines.append(f"  {e.model_size} seed={e.seed} [{e.variant.value}]: {e.notes}")
    return "\n".join(lines)
