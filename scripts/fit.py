"""Fit one simulated dataset with one prior.

Example:
    python scripts/fit.py --design p134 --hypothesis H1 --prior horseshoe --replicate 1
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pgxnlme.data import dataset_path, fit_directory, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--design", choices=("p134", "p800"), required=True)
    parser.add_argument("--hypothesis", choices=("H0", "H1"), required=True)
    parser.add_argument("--prior", required=True)
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument("--results", default=str(ROOT / "results"), help="output root (default: results/)")
    parser.add_argument("--overwrite", action="store_true", help="refit even if the output exists")
    parser.add_argument("--num-warmup", type=int, help="override the configured value (e.g. for a quick test)")
    parser.add_argument("--num-samples", type=int, help="override the configured value")
    parser.add_argument("--num-chains", type=int, help="override the configured value")
    args = parser.parse_args()

    config = load_config()
    sampler = dict(config["sampler"], **config.get("sampler_overrides", {}).get(args.prior, {}))
    for key in ("num_warmup", "num_samples", "num_chains"):
        if getattr(args, key) is not None:
            sampler[key] = getattr(args, key)
    output = fit_directory(args.results, args.design, args.hypothesis, args.prior, args.replicate)
    if (output / "diagnostics.json").exists() and not args.overwrite:
        print(f"exists, skipped: {output}")
        return
    # One CPU device per chain; must be set before JAX is imported.
    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={int(sampler['num_chains'])}"

    from pgxnlme.fit import fit_seed, run_fit

    diagnostics = run_fit(
        dataset_path(args.design, args.hypothesis, args.replicate), args.prior,
        config["designs"][args.design][args.prior], sampler, output,
        fit_seed(args.design, args.hypothesis, args.prior, args.replicate))
    print(f"done in {diagnostics['seconds'] / 60:.1f} min: {output} "
          f"(divergences {diagnostics['divergences']}, max PK R-hat {diagnostics['max_pk_rhat']:.3f})")


if __name__ == "__main__":
    main()
