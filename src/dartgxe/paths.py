"""Filesystem locations. Everything heavy lives under DARTGXE_ROOT (on the 4090: <workstation>/dart-gxe)."""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("DARTGXE_ROOT", REPO))
DATA = Path(os.environ.get("DARTGXE_DATA", ROOT / "data"))
RESULTS = Path(os.environ.get("DARTGXE_RESULTS", ROOT / "results"))


def processed(dataset: str) -> Path:
    return DATA / "processed" / dataset


def splits_dir(dataset: str) -> Path:
    return processed(dataset) / "splits"
