"""Monte Carlo prior checks used to calibrate the sparsity priors.

For each design (p=134 and p=800, N=400, reference omega_CL=0.3) this draws
200,000 SNP effects and implied model sizes from every prior and writes, to
results/calibration/:
  {design}_prior_draws.npz      the draws (used by Figure S1);
  {design}_model_size.csv       mean, SD, median and central 95% interval of the
                                model size (Supplementary Table S1);
  {design}_effect_regimes.csv   prior probabilities of effect-size regimes.

With --closed-form it instead prints closed-form starting values for a new
(N, p, target model size), to be checked with the same Monte Carlo summaries.

Examples:
    python scripts/prior_calibration.py
    python scripts/prior_calibration.py --closed-form --n 400 --p 2000 --target 5 --design p800
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pgxnlme.calibration import (beta_binomial_summary, closed_form_calibration, effect_samples,  # noqa: E402
                                 model_size_samples)
from pgxnlme.data import load_config  # noqa: E402

# Seeds follow this order: effect draws use 2605 + i, model sizes 4605 + i.
ORDER = ("spike-slab", "hierarchical-lasso", "horseshoe", "r2d2", "l1-ball", "point-mass-spike-slab")
N, OMEGA_REF, DRAWS = 400, 0.3, 200_000
REGIMES = {
    "P(beta = 0)": lambda v: np.mean(v == 0),
    "P(|beta| < 0.01)": lambda v: np.mean(np.abs(v) < 0.01),
    "P(0.01 < |beta| < 0.1)": lambda v: np.mean((np.abs(v) > 0.01) & (np.abs(v) < 0.1)),
    "P(|beta| > 0.13)": lambda v: np.mean(np.abs(v) > 0.13),
}


def calibrate(design, priors, out):
    p = int(design[1:])
    effects = {m: effect_samples(m, priors[m], p=p, omega_ref=OMEGA_REF, draws=DRAWS, seed=2605 + i)
               for i, m in enumerate(ORDER)}
    sizes = {m: model_size_samples(m, priors[m], n=N, p=p, draws=DRAWS, seed=4605 + i) for i, m in enumerate(ORDER)}
    np.savez_compressed(out / f"{design}_prior_draws.npz", **{f"{m}__beta": v for m, v in effects.items()},
                        **{f"{m}__size": v for m, v in sizes.items()})
    rows = []
    for m in ORDER:
        if m in ("spike-slab", "point-mass-spike-slab"):
            exact = beta_binomial_summary(p, priors[m]["theta_a"], priors[m]["theta_b"])
            rows.append(dict(prior=m, size="K_z (exact)", mean=exact["mean"], sd=exact["sd"], median=exact["median"],
                             q025=exact["q025"], q975=exact["q975"]))
        else:
            v = sizes[m]
            rows.append(dict(prior=m, size="K" if m == "l1-ball" else "m_eff", mean=v.mean(), sd=v.std(ddof=1),
                             median=np.median(v), q025=np.quantile(v, 0.025), q975=np.quantile(v, 0.975)))
    table = pd.DataFrame(rows)
    table.to_csv(out / f"{design}_model_size.csv", index=False, float_format="%.4g")
    regimes = pd.DataFrame({m: {k: f(effects[m]) for k, f in REGIMES.items()} for m in ORDER})
    regimes.to_csv(out / f"{design}_effect_regimes.csv", float_format="%.4f")
    print(f"\n{design}: implied prior model size\n{table.round(3).to_string(index=False)}")
    print(f"\n{design}: effect-size regimes (omega_CL = {OMEGA_REF})\n{regimes.round(4).to_string()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--design", nargs="+", choices=("p134", "p800"), default=["p134", "p800"])
    parser.add_argument("--out", default=str(ROOT / "results" / "calibration"))
    parser.add_argument("--closed-form", action="store_true")
    parser.add_argument("--n", type=int, default=N)
    parser.add_argument("--p", type=int)
    parser.add_argument("--target", type=float, default=5.0, help="target prior model size")
    args = parser.parse_args()
    config = load_config()
    if args.closed_form:
        if args.p is None:
            parser.error('--closed-form requires --p')
        start = config["designs"][args.design[0]]
        for name, values in closed_form_calibration(start, args.n, args.p, args.target).items():
            print(name, {k: round(float(v), 6) for k, v in values.items()})
        return
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for design in args.design:
        calibrate(design, config["designs"][design], out)


if __name__ == "__main__":
    main()
