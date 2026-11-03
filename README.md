# jointbayesianPGxNLME

Code for the simulation study of

> J. Martinelli, I. Rebai, D. W. Haas, J. Bertrand. *Joint Bayesian Inference of Genetic Effect Sizes and PK
> Parameters in Nonlinear Mixed-Effects Models.* Submitted to Statistics in Medicine.

A one-compartment steady-state PK model with random effects on $k_a$, $CL$ and $V_1$ is fitted jointly with SNP
effects on log-clearance, under six sparsity priors: point-mass spike-and-slab, continuous spike-and-slab, $\ell_1$-ball,
hierarchical lasso, regularized horseshoe and R2–D2. Models are written in [NumPyro](https://num.pyro.ai) and sampled
with NUTS (DiscreteHMCGibbs for the point-mass spike-and-slab).

## Contents

```
configs/priors.yaml          sampler settings and calibrated prior hyperparameters (p=134 and p=800)
pgxnlme/model.py             PK model and likelihood
pgxnlme/priors.py            the six sparsity priors
pgxnlme/fit.py               MCMC fit of one dataset with one prior
pgxnlme/calibration.py       Monte Carlo prior checks (effect sizes, implied model size)
pgxnlme/selection.py         inclusion probabilities, LD components, Cartesian credible sets (Griffin, 2025)
scripts/                     command-line entry points (see below)
data/{p134,p800}/{H0,H1}/    10 simulated datasets per design and hypothesis
environment/                 Python dependencies
```

**Datasets.** Each `replicate_XXX.npz` holds sampling times, concentrations (400 individuals, 3 samples each) and the
standardized genotype matrix `X`; the `.json` file next to it holds the causal SNP, the effect size and the simulation
parameters. Under H1, SNP `rs3745274` (index 0) has effect $-0.13$ on log-clearance; under H0 all effects are zero.

- `p134`: rows resampled from the 134-SNP candidate panel of the ANRS 12154 study.
- `p800`: rows resampled from the 100 European-ancestry donors of the public HAPNEST chromosome-19 panel
  (BioStudies S-BSST936), with raw allele counts in `genotypes`.

The paper uses 100 datasets per design and hypothesis; the 10 provided here reproduce the analysis at a smaller scale,
so figures will be noisier than in the paper.

## Installation

Python 3.13 (the paper used 3.13.3 on Linux x86-64).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r environment/requirements.txt         # direct dependencies, pinned
# or, for the exact environment used: pip install -r environment/requirements-lock.txt
```

Fits run on CPU in double precision, with one CPU core per chain (5 chains).

## Running the simulations

Sampler settings (`configs/priors.yaml`): 5 chains, 3,000 warm-up and 2,000 post-warm-up iterations, target acceptance
probability 0.99, maximal tree depth 14, random initialization from the prior (from the prior mean for R2–D2,
see `sampler_overrides`).

One fit:

```bash
python scripts/fit.py --design p134 --hypothesis H1 --prior horseshoe --replicate 1
```

All fits (2 designs × 2 hypotheses × 6 priors × 10 datasets = 240 fits), run sequentially and skipping completed ones:

```bash
python scripts/run_simulations.py                     # optionally --design p134 --prior horseshoe ...
```

On a Slurm cluster, one fit per array task:

```bash
mkdir -p logs
python scripts/run_simulations.py --list > jobs.txt
sbatch --array=1-$(wc -l < jobs.txt) scripts/slurm_array.sh   # add --account/--partition as needed
```

Each fit writes `results/<design>/<hypothesis>/<prior>/replicate_XXX/` with `samples.npz` (posterior draws of the PK
parameters, SNP effects and inclusion variables), `summary.csv` (R-hat and ESS) and `diagnostics.json` (run time,
divergences, tree-depth saturations, versions).

**Storage.** Saving the posterior samples of every dataset and prior is storage-heavy: up to about 10 MB per fit at
$p=134$ and 60 MB per fit at $p=800$ (continuous spike-and-slab, which also stores the conditional inclusion
probabilities), i.e. about 4 GB for the 240 fits, and ten times more for the 100 datasets per setting of the paper. Once `summarize_fits.py` has run (below), figures only need the small `derived.*` files, so
`samples.npz` can then be deleted.

**Run time** (per fit, 5 cores; Supplementary Table S3 of the paper). Median times range from 2 to 16 minutes at
$p=134$ and from 7 to 31 minutes at $p=800$, except R2–D2 at $p=800$, which takes about 8–11 hours. Peak memory stays
below 2 GB.

After the fits, derive inclusion probabilities, selections and R-hat values:

```bash
python scripts/summarize_fits.py --workers 8          # about 2 minutes per p=800 fit and worker
```

This writes `derived.npz` and `derived.json` in every fit directory. Selection uses the effect tolerance
$\varepsilon = 0.01$ (for the priors without inclusion indicators) and the cutoff 0.5, at three levels: single SNPs,
LD components ($|r| > 0.9$) and Cartesian credible-set blocks.

## Prior calibration

The Monte Carlo prior checks used to calibrate the hyperparameters (200,000 draws per prior, $N=400$, reference
$\omega_{CL}=0.3$) reproduce Supplementary Table S1 and the prior draws of Figure S1:

```bash
python scripts/prior_calibration.py
```

Outputs go to `results/calibration/`: the implied prior model size of every prior (`*_model_size.csv`), the prior
probabilities of effect-size regimes (`*_effect_regimes.csv`) and the draws (`*_prior_draws.npz`).
For a new design, closed-form starting values for a target prior model size are printed by

```bash
python scripts/prior_calibration.py --closed-form --n 400 --p 2000 --target 5 --design p800
```

and should then be checked with the Monte Carlo summaries above (by adding the new design to `configs/priors.yaml`).

## Figures

Once the fits are summarized and the prior calibration has run, all figures are produced in `figures/` by

```bash
python scripts/make_figures.py                        # or --only 3 4 S1 S2 S3 S4
```

| Paper figure | Output | Content |
|---|---|---|
| Figure 3 | `fig3_group_fwer.pdf` | Global-null FWER (H0) at the SNP, LD and Cartesian-block levels |
| Figure 4 | `fig4_group_f1.pdf` | Mean $F_1$-score (H1) at the same three levels |
| Figure S1 | `figS1_prior_calibration_p800.pdf` | Prior effect densities and implied model sizes at $p=800$ |
| Figure S2 | `figS2_pk_posteriors_{H0,H1}.pdf` | Posterior densities of the PK parameters and of the `rs3745274` effect ($p=134$) |
| Figure S3 | `figS3_rhat_{H0,H1}.pdf` | R-hat of the same parameters across datasets ($p=134$) |
| Figure S4 | `figS4_pip_tolerance.pdf` | Inclusion probabilities against the effect tolerance $\varepsilon$ |

The numbers behind Figures 3 and 4 are also written as CSV files.

## Citation

If you use this code, please consider citing the paper above.
