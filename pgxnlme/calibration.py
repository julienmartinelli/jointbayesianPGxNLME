"""Prior calibration: Monte Carlo draws of SNP effects and implied model sizes.

The parameterizations mirror ``priors.py``. Effects are drawn at a reference
omega_CL (0.3 in the paper). Model size is K_z (slab allocations) for the
spike-and-slab priors, the exact support size K for the l1-ball, and
m_eff = sum_j (1 - kappa_j) for the global-local priors, under an orthogonal
Gaussian design with X'X = N I.
"""

import numpy as np
from scipy.stats import betabinom

SPIKE_SLABS = ("point-mass-spike-slab", "spike-slab")


def _project_l1_rows(values, radii):
    """Euclidean projection of each row onto an l1 ball with row-specific radius."""
    absolute = np.abs(values)
    projected = values.copy()
    active = absolute.sum(axis=1) > radii
    if not np.any(active):
        return projected
    selected = absolute[active]
    ordered = np.sort(selected, axis=1)[:, ::-1]
    cumulative = np.cumsum(ordered, axis=1)
    index = np.arange(1, values.shape[1] + 1)
    rho = np.maximum((ordered > (cumulative - radii[active, None]) / index).sum(axis=1), 1)
    threshold = (cumulative[np.arange(rho.size), rho - 1] - radii[active]) / rho
    projected[active] = np.sign(values[active]) * np.maximum(selected - threshold[:, None], 0.0)
    return projected


def effect_samples(prior, hyper, *, p, omega_ref, draws, seed, batch_size=2_000):
    """Draws of one exchangeable coefficient beta_j from the prior."""
    rng = np.random.default_rng(seed)
    out, remaining = [], int(draws)
    while remaining:
        size = min(batch_size, remaining)
        if prior in SPIKE_SLABS:
            theta = rng.beta(hyper["theta_a"], hyper["theta_b"], size=size)
            included = rng.random(size) < theta
            if prior == "point-mass-spike-slab":
                beta = np.where(included, rng.normal(size=size) * hyper["slab_sd"] * omega_ref, 0.0)
            else:
                scale = np.where(included, hyper["slab_sd"], hyper["spike_sd"])
                beta = rng.normal(size=size) * scale * omega_ref
        elif prior == "hierarchical-lasso":
            local = rng.exponential(1.0 / hyper["exp"], size=size)
            tau = hyper["scale_tau"] * np.abs(rng.normal(size=size)) * omega_ref
            beta = rng.normal(size=size) * local * tau
        elif prior == "horseshoe":
            # Log scale avoids overflow when an inverse-Gamma draw is huge.
            tiny = np.finfo(float).tiny
            nu_g, nu_l, df = hyper["nu_global"], hyper["nu_local"], hyper["slab_df"]
            g_g, g_l, g_s = (rng.gamma(0.5 * nu_g, size=size), rng.gamma(0.5 * nu_l, size=size),
                             rng.gamma(0.5 * df, size=size))
            log_tau = (np.log(np.maximum(np.abs(rng.normal(size=size)), tiny))
                       + 0.5 * (np.log(0.5 * nu_g) - np.log(np.maximum(g_g, tiny)))
                       + np.log(hyper["scale_global"] * omega_ref))
            log_lam = (np.log(np.maximum(np.abs(rng.normal(size=size)), tiny))
                       + 0.5 * (np.log(0.5 * nu_l) - np.log(np.maximum(g_l, tiny))))
            log_c = np.log(hyper["slab_sd"] * omega_ref) + 0.5 * (np.log(0.5 * df) - np.log(np.maximum(g_s, tiny)))
            log_raw = log_tau + log_lam
            log_scale = log_c + log_raw - 0.5 * np.logaddexp(2.0 * log_c, 2.0 * log_raw)
            beta = rng.normal(size=size) * np.exp(log_scale)
        elif prior == "l1-ball":
            eta = rng.laplace(0.0, hyper["b_eta"] * omega_ref, size=(size, p))
            radius = rng.exponential(omega_ref / hyper["r_lambda"], size=size)
            beta = _project_l1_rows(eta, radius)[np.arange(size), rng.integers(0, p, size=size)]
        elif prior == "r2d2":
            r2 = rng.beta(hyper["a"], hyper["b"], size=size)
            alpha = hyper["alpha"]
            allocation = rng.beta(alpha / p, alpha - alpha / p, size=size)
            beta = rng.normal(size=size) * np.sqrt(omega_ref**2 * r2 / (1.0 - r2) * allocation)
        else:
            raise ValueError(f"Unknown prior: {prior}")
        out.append(beta)
        remaining -= size
    return np.concatenate(out)


def model_size_samples(prior, hyper, *, n, p, draws, seed, batch_size=500):
    """Draws of the implied model size (K_z, K or m_eff, see module docstring)."""
    rng = np.random.default_rng(seed)
    out, remaining = [], int(draws)
    while remaining:
        size = min(batch_size, remaining)
        if prior in SPIKE_SLABS:
            size_draw = rng.binomial(p, rng.beta(hyper["theta_a"], hyper["theta_b"], size=size)).astype(float)
        elif prior == "hierarchical-lasso":
            local = rng.exponential(1.0 / hyper["exp"], size=(size, p))
            tau = hyper["scale_tau"] * np.abs(rng.normal(size=size))
            size_draw = np.sum(1.0 - 1.0 / (1.0 + n * (tau[:, None] * local) ** 2), axis=1)
        elif prior == "horseshoe":
            nu_l, nu_g, df = hyper["nu_local"], hyper["nu_global"], hyper["slab_df"]
            inv_local = (0.5 * nu_l) / rng.gamma(0.5 * nu_l, size=(size, p))
            inv_global = (0.5 * nu_g) / rng.gamma(0.5 * nu_g, size=size)
            inv_slab = (0.5 * df) / rng.gamma(0.5 * df, size=size)
            lam = rng.normal(size=(size, p)) * np.sqrt(inv_local)
            tau = rng.normal(size=size) * np.sqrt(inv_global) * hyper["scale_global"]
            c = hyper["slab_sd"] * np.sqrt(inv_slab)
            lam_tilde_sq = c[:, None] ** 2 * lam**2 / (c[:, None] ** 2 + tau[:, None] ** 2 * lam**2)
            size_draw = np.sum(1.0 - 1.0 / (1.0 + n * tau[:, None] ** 2 * lam_tilde_sq), axis=1)
        elif prior == "l1-ball":
            eta = rng.laplace(0.0, hyper["b_eta"], size=(size, p))
            radius = rng.exponential(1.0 / hyper["r_lambda"], size=size)
            size_draw = np.count_nonzero(_project_l1_rows(eta, radius), axis=1).astype(float)
        elif prior == "r2d2":
            r2 = rng.beta(hyper["a"], hyper["b"], size=size)
            allocation = rng.dirichlet(np.full(p, hyper["alpha"] / p), size=size)
            size_draw = np.sum(1.0 - 1.0 / (1.0 + n * (r2 / (1.0 - r2))[:, None] * allocation), axis=1)
        else:
            raise ValueError(f"Unknown prior: {prior}")
        out.append(size_draw)
        remaining -= size
    return np.concatenate(out)


def beta_binomial_summary(p, a, b):
    """Exact law of K_z = sum_j z_j under theta ~ Beta(a, b)."""
    law = betabinom(p, a, b)
    return dict(mean=p * a / (a + b), sd=float(law.std()), median=int(law.ppf(0.5)),
                q025=int(law.ppf(0.025)), q975=int(law.ppf(0.975)))


def l1_ball_rate(p, b_eta, target_mean):
    """Radius rate giving E[K] = target_mean exactly (P(K >= k) = c^(k-1), c = 1/(1 + r b))."""
    lower, upper, powers = 0.0, 1.0, np.arange(p, dtype=float)
    for _ in range(100):
        c = 0.5 * (lower + upper)
        lower, upper = (c, upper) if np.sum(c**powers) < target_mean else (lower, c)
    return (1.0 / (0.5 * (lower + upper)) - 1.0) / b_eta


def closed_form_calibration(priors, n, p, target_size):
    """Closed-form starting values for a target prior model size m (to be checked by simulation).

    Effect-magnitude hyperparameters (slab SD, b_eta, rho, alpha + a + b) are kept.
    """
    out = {name: dict(values) for name, values in priors.items()}
    m = float(target_size)
    for name in SPIKE_SLABS:
        if name in out:
            out[name]["theta_b"] = out[name]["theta_a"] * (p / m - 1.0)
    lasso = out["hierarchical-lasso"]
    lasso["scale_tau"] = lasso["exp"] / np.sqrt(2.0 * n) * np.sqrt(m / (p - m))
    horseshoe = out["horseshoe"]
    floor = 1.0 / (1.0 + n * horseshoe["slab_sd"] ** 2)
    horseshoe["scale_global"] = m / ((1.0 - floor) * p - m) / np.sqrt(n)
    out["l1-ball"]["r_lambda"] = l1_ball_rate(p, out["l1-ball"]["b_eta"], m)
    r2d2 = out["r2d2"]
    concentration = r2d2["a"] + r2d2["b"]
    odds = (p / n) * (m / (p - m))
    r2d2["a"], r2d2["b"] = concentration * odds / (1 + odds), concentration / (1 + odds)
    return out
