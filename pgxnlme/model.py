"""Joint NLME pharmacokinetic model with a sparsity prior on SNP effects.

One-compartment model with first-order absorption at steady state, lognormal
residual error, and log-normal random effects on ka, CL and V. SNP effects act
on log-clearance and are scaled by the clearance random-effect SD omegaCL.
"""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax.nn import softplus

from .priors import sparsity_prior


def one_compartment(time, dose, CL, V1, ka, period=12.0):
    """Steady-state concentrations, shape (N, T), assuming ka > CL / V1."""
    k = CL / V1
    dm_k = -jnp.expm1(-k * period)
    dm_ka = -jnp.expm1(-ka * period)
    k, ka = k[:, None], ka[:, None]
    factor = (dose / V1)[:, None] * ka / (ka - k)
    return factor * (jnp.exp(-k * time) / dm_k[:, None] - jnp.exp(-ka * time) / dm_ka[:, None])


def model(time, conc, mask, dose, X, prior):
    """NumPyro model.

    time, conc, mask: (N, T) arrays; X: (N, p) standardized genotypes;
    prior: dict with the prior ``name`` and its hyperparameters.
    """
    N, p = X.shape
    # Population parameters (log scale) and residual SD.
    log_kaHat = numpyro.sample("log_kaHat", dist.Normal(0.01, 0.25))
    log_CLHat = numpyro.sample("log_CLHat", dist.Normal(0.9, 0.25))
    log_V1Hat = numpyro.sample("log_V1Hat", dist.Normal(5.3, 0.25))
    log_sigma1 = numpyro.sample("log_sigma1", dist.Normal(jnp.log(0.1), 1.0))
    # Between-subject variability.
    omegaka = numpyro.sample("omegaka", dist.HalfNormal(1.0))
    omegaCL = numpyro.sample("omegaCL", dist.HalfNormal(1.0))
    omegaV1 = numpyro.sample("omegaV1", dist.HalfNormal(1.0))
    with numpyro.plate("individuals", N):
        zka = numpyro.sample("zka", dist.Normal(0.0, 1.0))
        zCL = numpyro.sample("zCL", dist.Normal(0.0, 1.0))
        zV1 = numpyro.sample("zV1", dist.Normal(0.0, 1.0))
    # Genetic effects on log-clearance.
    beta = sparsity_prior(p, prior, omegaCL)
    CL = jnp.exp(log_CLHat + omegaCL * zCL + X @ beta)
    V1 = jnp.exp(log_V1Hat + omegaV1 * zV1)
    # ka > k is enforced smoothly: ka = k + softplus(ka_tilde - k).
    k = CL / V1
    ka = k + softplus(jnp.exp(log_kaHat + omegaka * zka) - k) + 1e-8
    pred = one_compartment(time, dose, CL, V1, ka)
    likelihood = dist.LogNormal(jnp.log(pred.reshape(-1) + 1e-9), jnp.exp(log_sigma1))
    with numpyro.plate("obs", conc.size):
        numpyro.sample("conc", likelihood.mask(mask.reshape(-1)), obs=conc.reshape(-1))
