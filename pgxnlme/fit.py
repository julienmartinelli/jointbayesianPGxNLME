"""Fit one dataset with one prior and save posterior samples and diagnostics.

JAX reads the number of host devices at import, so callers must set it (see
``scripts/fit.py``) before importing this module.
"""

import hashlib
import json
import platform
import time
from pathlib import Path

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro.infer import MCMC, NUTS, DiscreteHMCGibbs, init_to_mean, init_to_median, init_to_sample

from .data import load_dataset
from .model import model

PK_SITES = ("log_kaHat", "log_CLHat", "log_V1Hat", "log_sigma1", "omegaka", "omegaCL", "omegaV1")
KEEP_SITES = PK_SITES + ("beta", "theta", "incl_prob", "z")


def fit_seed(design, hypothesis, prior, replicate):
    key = f"{design}/{hypothesis}/{prior}/{int(replicate)}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "little")


def run_fit(dataset, prior_name, hyperparameters, sampler, output_dir, seed):
    jax.config.update("jax_enable_x64", True)
    arrays, metadata = load_dataset(dataset)
    data = dict(time=jnp.asarray(arrays["time"]), conc=jnp.asarray(arrays["conc"]),
                mask=jnp.ones(arrays["conc"].shape, dtype=bool), dose=float(metadata["dose"]),
                X=jnp.asarray(arrays["X"]), prior=dict(name=prior_name, **hyperparameters))

    init = {"sample": init_to_sample, "mean": init_to_mean, "median": init_to_median}[sampler["init"]]()
    nuts = NUTS(model, init_strategy=init, target_accept_prob=float(sampler["target_accept_prob"]),
                max_tree_depth=int(sampler["max_tree_depth"]))
    if prior_name == "point-mass-spike-slab":
        kernel = DiscreteHMCGibbs(nuts, modified=True)
        fields = ("hmc_state.diverging", "hmc_state.num_steps", "hmc_state.accept_prob")
    else:
        kernel = nuts
        fields = ("diverging", "num_steps", "accept_prob")
    mcmc = MCMC(kernel, num_warmup=int(sampler["num_warmup"]), num_samples=int(sampler["num_samples"]),
                num_chains=int(sampler["num_chains"]), chain_method="parallel", progress_bar=False)
    start = time.perf_counter()
    mcmc.run(jax.random.PRNGKey(int(seed)), extra_fields=fields, **data)
    samples = {key: np.asarray(value) for key, value in mcmc.get_samples(group_by_chain=True).items()}
    seconds = time.perf_counter() - start
    extra = {key.split(".")[-1]: np.asarray(value)
             for key, value in mcmc.get_extra_fields(group_by_chain=True).items()}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    kept = {key: (value.astype(np.int8) if key == "z" else value.astype(np.float32))
            for key, value in samples.items() if key in KEEP_SITES}
    np.savez_compressed(output_dir / "samples.npz", **kept)

    # Rank-normalized split R-hat and ESS for the PK parameters and the SNP effects.
    posterior = az.from_dict(posterior={key: samples[key] for key in PK_SITES + ("beta",)})
    summary = az.summary(posterior, kind="diagnostics")
    summary.to_csv(output_dir / "summary.csv")

    saturated = int(np.sum(extra["num_steps"] >= 2 ** int(sampler["max_tree_depth"]) - 1))
    pk_rhat = summary.loc[list(PK_SITES), "r_hat"]
    diagnostics = dict(
        dataset=str(Path(dataset).relative_to(Path(dataset).parents[2])), prior=prior_name,
        hyperparameters=hyperparameters, sampler=sampler, seed=int(seed), seconds=seconds,
        divergences=int(np.sum(extra["diverging"])), tree_depth_saturations=saturated,
        mean_accept_prob=float(np.mean(extra["accept_prob"])), mean_num_steps=float(np.mean(extra["num_steps"])),
        max_pk_rhat=float(pk_rhat.max()),
        versions=dict(python=platform.python_version(), jax=jax.__version__, numpyro=numpyro.__version__,
                      arviz=az.__version__, numpy=np.__version__),
        host=platform.node())
    (output_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=1) + "\n")
    return diagnostics
