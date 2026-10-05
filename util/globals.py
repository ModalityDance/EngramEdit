"""Repository-relative paths; runtime data and results are not distributed."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
HPARAMS_DIR = PROJECT_ROOT / "hparams"
REMOTE_ROOT_URL = "https://memit.baulab.info"
