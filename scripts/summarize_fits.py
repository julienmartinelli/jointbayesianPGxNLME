"""Derive compact per-fit results from the saved posterior samples.

For every completed fit this writes, next to samples.npz:
  derived.npz   inclusion probabilities over the epsilon grid (or the
                spike-and-slab PIPs), thinned PK draws and draws of the
                reference SNP effect (rs3745274 / chr19:41006936:G:T);
  derived.json  SNP, LD-component and Cartesian-block selection at
                epsilon = 0.01 and cutoff 0.5, and R-hat values.
Figures only read these files, so samples.npz can be deleted afterwards.

Example:
    python scripts/summarize_fits.py --workers 8    # all fits not yet summarized
"""

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pgxnlme.data import load_dataset  # noqa: E402
from pgxnlme.fit import PK_SITES  # noqa: E402
from pgxnlme.selection import (EPSILON_GRID, block_scores, cartesian_blocks, inclusion_events,  # noqa: E402
                               ld_components, pip, selection_counts)

SPIKE_SLABS = ("point-mass-spike-slab", "spike-slab")
EPSILON, CUTOFF, THIN = 0.01, 0.5, 10


def pip_curves(beta):
    """p(|beta_j| > eps | D) for every eps in EPSILON_GRID, shape (n_eps, p)."""
    magnitude = np.sort(np.abs(beta.reshape(-1, beta.shape[-1]).astype(float)), axis=0)
    n = magnitude.shape[0]
    return np.stack([1.0 - np.searchsorted(column, EPSILON_GRID, side="right") / n for column in magnitude.T], axis=1)


def summarize(fit_dir):
    design, hypothesis, prior, replicate = fit_dir.parts[-4], fit_dir.parts[-3], fit_dir.parts[-2], int(fit_dir.name[-3:])
    arrays, metadata = load_dataset(ROOT / "data" / design / hypothesis / f"replicate_{replicate:03d}.npz")
    with np.load(fit_dir / "samples.npz") as archive:
        samples = {key: archive[key] for key in archive.files}
    reference = int(metadata["reference_index"])
    causal = metadata["causal_indices"][0] if metadata["causal_indices"] else None

    # Inclusion probabilities.
    pips = pip(samples, prior, EPSILON)
    curves = pips[None, :] if prior in SPIKE_SLABS else pip_curves(samples["beta"])

    # Selection of SNPs, fixed LD components and Cartesian blocks at epsilon = 0.01.
    seed = int.from_bytes(hashlib.sha256(f"2605/{fit_dir.relative_to(fit_dir.parents[3])}".encode()).digest()[:4], "little")
    events, weights = inclusion_events(samples, prior, EPSILON, seed=seed)
    p = events.shape[-1]
    events, weights = events.reshape(-1, p), None if weights is None else weights.reshape(-1, p)
    ld = ld_components(arrays["X"])
    blocks, retained = cartesian_blocks(events)
    levels = {
        "individual": (pips, causal, [1] * p),
        "ld": (np.array([pips[list(g)].max() for g in ld]),
               None if causal is None else next(i for i, g in enumerate(ld) if causal in g), [len(g) for g in ld]),
        "griffin": (block_scores(blocks, retained, weights),
                    None if causal is None else next(i for i, b in enumerate(blocks) if causal in b.indices),
                    [len(b.indices) for b in blocks]),
    }
    selection = []
    for level, (scores, true_group, sizes) in levels.items():
        tp, fp, fn, f1 = selection_counts(scores, true_group, CUTOFF)
        selection.append(dict(level=level, tp=tp, fp=fp, fn=fn, f1=f1,
                              causal_group_size=None if true_group is None else int(sizes[true_group])))

    # R-hat of the PK parameters and of the reference SNP effect.
    table = pd.read_csv(fit_dir / "summary.csv", index_col=0)
    rhat = {site: float(table.loc[site, "r_hat"]) for site in PK_SITES}
    rhat["beta_reference"] = float(table.loc[f"beta[{reference}]", "r_hat"])

    np.savez_compressed(
        fit_dir / "derived.npz", epsilon=EPSILON_GRID if prior not in SPIKE_SLABS else np.array([np.nan]),
        pip_curves=curves.astype(np.float32), reference_index=reference,
        beta_reference=samples["beta"][:, ::THIN, reference].ravel(),
        **{site: samples[site][:, ::THIN].ravel() for site in PK_SITES})
    record = dict(design=design, hypothesis=hypothesis, prior=prior, replicate=replicate,
                  selection=selection, rhat=rhat)
    (fit_dir / "derived.json").write_text(json.dumps(record, indent=1) + "\n")
    return fit_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=str(ROOT / "results"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=1, help="fits summarized in parallel")
    args = parser.parse_args()
    fits = sorted(path.parent for path in Path(args.results).glob("*/*/*/replicate_*/diagnostics.json"))
    todo = [f for f in fits if args.overwrite or not (f / "derived.json").exists()]
    missing = [f for f in todo if not (f / "samples.npz").exists()]
    todo = [f for f in todo if f not in missing]
    print(f"{len(fits)} fits found, {len(todo)} to summarize"
          + (f", {len(missing)} without samples.npz skipped" if missing else ""))
    with ProcessPoolExecutor(args.workers) as pool:
        for i, fit_dir in enumerate(pool.map(summarize, todo), 1):
            print(f"[{i}/{len(todo)}] {fit_dir.relative_to(Path(args.results))}", flush=True)


if __name__ == "__main__":
    main()
