"""Datasets, configuration and result paths."""

import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CONFIG = ROOT / "configs" / "priors.yaml"
DESIGNS = ("p134", "p800")
HYPOTHESES = ("H0", "H1")


def load_config(path=CONFIG):
    return yaml.safe_load(Path(path).read_text())


def dataset_path(design, hypothesis, replicate, data_dir=DATA):
    return Path(data_dir) / design / hypothesis / f"replicate_{int(replicate):03d}.npz"


def list_datasets(designs=DESIGNS, hypotheses=HYPOTHESES, data_dir=DATA):
    return [path for design in designs for hypothesis in hypotheses
            for path in sorted((Path(data_dir) / design / hypothesis).glob("replicate_*.npz"))]


def load_dataset(path):
    """Return (arrays, metadata): arrays hold time, conc, X (N x p) and, at p=800, raw genotypes."""
    path = Path(path)
    with np.load(path) as archive:
        arrays = {key: archive[key] for key in archive.files}
    metadata = json.loads(path.with_suffix(".json").read_text())
    return arrays, metadata


def fit_directory(results_dir, design, hypothesis, prior, replicate):
    return Path(results_dir) / design / hypothesis / prior / f"replicate_{int(replicate):03d}"
