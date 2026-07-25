import os
from pathlib import Path

from dotenv import load_dotenv

# --- Paths -------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_DIR = REPO_ROOT / "registry"
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"  # unprocessed API dumps kept for offline re-analysis; gitignored

# --- WandB entities/projects ---------------------------------------------------
# pythia includes data for losses for the original pythia runs.
# pythia-extra-seeds includes data for losses from polypythia, additional runs with varying
# controlled seeds

WANDB_ENTITY = "eleutherai"
POLYPYTHIAS_PROJECT = "pythia-extra-seeds"
BASE_PYTHIA_PROJECT = "pythia"

# --- Hugging Face Hub -----------------------------------------------------------
# Confirmed repo naming for the per-seed PolyPythias releases (see the collection at
# huggingface.co/collections/EleutherAI/polypythias-67bed6916110c8933e1ea561), used by
# discovery.py to cross-reference WandB candidates against the HF repo's creation
# timestamp when a seed has more than one independently finished run.

HF_SEED_REPO_TEMPLATE = "EleutherAI/pythia-{size}-seed{seed}"

# --- Token math (Pythia family only) -------------------------------------------
# batch_size=1024 sequences * seq_len=2048 tokens/sequence, constant for the whole
# Pythia/PolyPythias training run. Verified against ColPret's own data: the first
# recorded checkpoint's tokens_seen was literally 2.097152e+06 (= 1 * TOKENS_PER_STEP).
# This linear conversion holds ONLY because batch size and sequence length never change
# during training -- do not reuse this constant for a model family with a batch-size
# warmup schedule.

TOKENS_PER_STEP = 1024 * 2048  # 2_097_152


def load_wandb_api_key() -> str:
    """Load WANDB_API_KEY from .env (or the environment) and fail loudly if absent.

    Raises:
        RuntimeError: if no key is set anywhere -- callers should not silently proceed
            with an unauthenticated wandb client, since that produces confusing
            downstream 401s instead of a clear error at the point of use.
    """
    load_dotenv(REPO_ROOT / ".env")
    key = os.environ.get("WANDB_API_KEY")
    if not key:
        raise RuntimeError(
            "WANDB_API_KEY is not set. Copy .env.example to .env and fill in a key "
            "with read access to the eleutherai WandB entity (https://wandb.ai/authorize)."
        )
    return key
