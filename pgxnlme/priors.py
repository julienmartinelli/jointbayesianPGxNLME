"""The six sparsity priors on SNP effects (NumPyro).

Every prior scales the effects by omegaCL, the SD of the clearance random
effect, so that hyperparameters refer to the dimensionless beta / omegaCL.
Each function registers a deterministic site ``beta`` (length p).
"""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import lax
from jax.nn import logsumexp
from jax.scipy.special import gammaln

PRIORS = (
    "point-mass-spike-slab",
    "spike-slab",
    "l1-ball",
    "hierarchical-lasso",
    "horseshoe",
    "r2d2",
)


def point_mass_spike_slab(p, omegaCL, theta_a, theta_b, slab_sd):
    """Point mass at zero and Normal slab; z is updated by discrete Gibbs."""
    theta = numpyro.sample("theta", dist.Beta(theta_a, theta_b))
    with numpyro.plate("snps", p):
        z = numpyro.sample("z", dist.Bernoulli(probs=theta))
        slab_raw = numpyro.sample("slab_raw", dist.Normal(0.0, 1.0))
    beta = numpyro.deterministic("beta", z * omegaCL * slab_sd * slab_raw)
    numpyro.deterministic("model_size", jnp.sum(z))
    return beta


def spike_slab(p, omegaCL, theta_a, theta_b, slab_sd, spike_sd):
    """Continuous (two-Normal) spike-and-slab; indicators are enumerated out.

    The coefficient is sampled on the scale beta / omegaCL, which leaves the
    prior unchanged. ``incl_prob`` records p(z_j = 1 | beta_j, theta, omegaCL),
    whose posterior mean is the Rao-Blackwellized inclusion probability.
    """
    theta = numpyro.sample("theta", dist.Beta(theta_a, theta_b))
    with numpyro.plate("snps", p):
        z = numpyro.sample("z", dist.Bernoulli(probs=theta), infer={"enumerate": "parallel"})
        beta_unit = numpyro.sample("beta_unit", dist.Normal(0.0, spike_sd * (1.0 - z) + slab_sd * z))
    beta = numpyro.deterministic("beta", omegaCL * beta_unit)
    log_slab = jnp.log(theta) + dist.Normal(0.0, slab_sd * omegaCL).log_prob(beta)
    log_spike = jnp.log1p(-theta) + dist.Normal(0.0, spike_sd * omegaCL).log_prob(beta)
    numpyro.deterministic("incl_prob", jnp.exp(log_slab - jnp.logaddexp(log_slab, log_spike)))
    return beta


def hierarchical_lasso(p, omegaCL, scale_tau, exp):
    """beta_j | tau, lambda_j ~ N(0, omegaCL^2 tau^2 lambda_j^2), tau ~ HN(scale_tau), lambda_j ~ Exp(exp)."""
    z = numpyro.sample("z", dist.Normal(0.0, 1.0).expand([p]))
    lam = numpyro.sample("lambda", dist.Exponential(exp).expand([p]))
    tau_raw = numpyro.sample("tau_raw", dist.HalfNormal(1.0))
    return numpyro.deterministic("beta", z * lam * scale_tau * tau_raw * omegaCL)


def horseshoe(p, omegaCL, scale_global, nu_global, nu_local, slab_sd, slab_df):
    """Regularized horseshoe; half-t scales written as HalfNormal * sqrt(InverseGamma)."""
    r1_local = numpyro.sample("r1_local", dist.HalfNormal(1.0).expand([p]))
    r2_local = numpyro.sample("r2_local", dist.InverseGamma(0.5 * nu_local, 0.5 * nu_local).expand([p]))
    r1_global = numpyro.sample("r1_global", dist.HalfNormal(1.0))
    r2_global = numpyro.sample("r2_global", dist.InverseGamma(0.5 * nu_global, 0.5 * nu_global))
    caux = numpyro.sample("caux", dist.InverseGamma(0.5 * slab_df, 0.5 * slab_df))
    lam = r1_local * jnp.sqrt(r2_local)
    tau = r1_global * jnp.sqrt(r2_global) * scale_global * omegaCL
    c = slab_sd * jnp.sqrt(caux) * omegaCL
    lam_tilde = jnp.sqrt(c**2 * lam**2 / (c**2 + tau**2 * lam**2))
    z = numpyro.sample("z", dist.Normal(0.0, 1.0).expand([p]))
    return numpyro.deterministic("beta", z * lam_tilde * tau)


def l1_ball_projection(eta, r):
    """Euclidean projection of eta onto the l1 ball of radius r."""
    abs_eta = jnp.abs(eta)

    def project(_):
        a = jnp.sort(abs_eta)[::-1]
        cumsum = jnp.cumsum(a)
        j = jnp.arange(1, eta.size + 1)
        c = jnp.maximum(jnp.sum(a > (cumsum - r) / j).astype(jnp.int32), 1)
        mu = (cumsum[c - 1] - r) / c
        return jnp.sign(eta) * jnp.maximum(abs_eta - mu, 0.0)

    return lax.cond(jnp.sum(abs_eta) <= r, lambda _: eta, project, operand=None)


def l1_ball(p, omegaCL, b_eta, r_lambda):
    """eta_j ~ Laplace(0, b_eta), r ~ Exp(r_lambda), beta = omegaCL * Proj_r(eta)."""
    eta_raw = numpyro.sample("eta_raw", dist.Laplace(0.0, b_eta).expand([p]))
    r_raw = numpyro.sample("r_raw", dist.Exponential(r_lambda))
    return numpyro.deterministic("beta", omegaCL * l1_ball_projection(eta_raw, r_raw))


def _log_gamma(concentration, rate, p, name):
    """Log of p independent Gamma(concentration, rate) values, sampled on a stable scale."""
    location = jnp.log(concentration) - jnp.log(rate)
    proposal = dist.Normal(0.0, 1.0).expand([p])
    raw = numpyro.sample(name, proposal)
    log_values = location + raw / concentration
    log_density = (concentration * log_values - rate * jnp.exp(log_values)
                   + concentration * jnp.log(rate) - gammaln(concentration) - jnp.log(concentration))
    numpyro.factor(f"{name}_log_density", jnp.sum(log_density - proposal.log_prob(raw)))
    return log_values


def r2d2(p, omegaCL, a, b, alpha):
    """R2-D2: R2 ~ Beta(a, b), phi ~ Dirichlet(alpha/p), beta_j ~ N(0, omegaCL^2 R2/(1-R2) phi_j).

    The Dirichlet weights are normalized Gamma(alpha/p, 1) variables, evaluated
    in log space to avoid underflow when alpha/p is small.
    """
    R2 = numpyro.sample("R2_genetic", dist.Beta(a, b))
    log_gamma = _log_gamma(alpha / p, 1.0, p, "r2d2_log_allocation_raw")
    log_phi = log_gamma - logsumexp(log_gamma)
    log_scale = 0.5 * (jnp.log(R2) - jnp.log1p(-R2) + 2.0 * jnp.log(omegaCL) + log_phi)
    z = numpyro.sample("z_beta", dist.Normal(0.0, 1.0).expand([p]))
    numpyro.deterministic("phi", jnp.exp(log_phi))
    return numpyro.deterministic("beta", jnp.exp(log_scale) * z)


def sparsity_prior(p, prior, omegaCL):
    """Dispatch on prior['name']; the remaining keys are hyperparameters."""
    functions = {
        "point-mass-spike-slab": point_mass_spike_slab,
        "spike-slab": spike_slab,
        "l1-ball": l1_ball,
        "hierarchical-lasso": hierarchical_lasso,
        "horseshoe": horseshoe,
        "r2d2": r2d2,
    }
    hyper = {key: value for key, value in prior.items() if key != "name"}
    return functions[prior["name"]](p, omegaCL, **hyper)
