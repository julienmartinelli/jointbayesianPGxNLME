"""Regenerate Figures 3, 4 and S1-S4 of the paper from the summarized fits.

Requires scripts/summarize_fits.py (Figures 3, 4, S2-S4) and
scripts/prior_calibration.py (Figure S1). Figures are written to figures/.

Examples:
    python scripts/make_figures.py                 # all figures
    python scripts/make_figures.py --only S1 S4
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy.stats import beta as beta_dist, betabinom, gaussian_kde  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pgxnlme.data import load_config  # noqa: E402

ORDER = ("point-mass-spike-slab", "spike-slab", "l1-ball", "hierarchical-lasso", "horseshoe", "r2d2")
SPIKE_SLABS = ("point-mass-spike-slab", "spike-slab")
# Figures S1-S4.
NAMES = {"point-mass-spike-slab": "Point-mass S&S", "spike-slab": "Continuous S&S", "l1-ball": r"$\ell_1$-ball",
         "hierarchical-lasso": "Hierarchical lasso", "horseshoe": "Regularized horseshoe", "r2d2": "R2–D2"}
COLORS = {"point-mass-spike-slab": "black", "spike-slab": "purple", "l1-ball": "orange",
          "hierarchical-lasso": "red", "horseshoe": "green", "r2d2": "cyan"}
ZORDER = {"spike-slab": 5, "hierarchical-lasso": 4, "horseshoe": 3, "r2d2": 2, "l1-ball": 1, "point-mass-spike-slab": 6}
# Figures 3-4.
GROUP_STYLES = [("hierarchical-lasso", "Hierarchical lasso", "#D55E00", "o"), ("horseshoe", "Horseshoe", "#009E73", "s"),
                ("l1-ball", r"$\ell_1$-ball", "#E69F00", "D"), ("spike-slab", "Continuous spike-and-slab", "#CC79A7", "^"),
                ("point-mass-spike-slab", "Point-mass spike-and-slab", "#333333", "v"), ("r2d2", "R2–D2", "#56B4E9", "P")]
LEVELS = ["individual", "ld", "griffin"]
TRUE_PK = {"CLHat": 2.5, "omegaCL": 0.3, "beta": -0.13, "omegaka": 0.3, "omegaV1": 0.3, "sigma1": 0.1,
           "V1Hat": 200.0, "kaHat": 1.0}
PK_PARAMS = {"V1Hat": ("log_V1Hat", True), "omegaV1": ("omegaV1", False), "sigma1": ("log_sigma1", True),
             "kaHat": ("log_kaHat", True), "omegaka": ("omegaka", False), "CLHat": ("log_CLHat", True),
             "omegaCL": ("omegaCL", False)}
PK_LABELS = {"V1Hat": r"$\hat{V_1}$", "omegaV1": r"$\omega_{V_1}$", "sigma1": r"$\sigma_1$", "kaHat": r"$\hat{k_a}$",
             "omegaka": r"$\omega_{k_a}$", "beta": r"$\beta_{\mathrm{rs37}}$", "CLHat": r"$\hat{CL}$",
             "omegaCL": r"$\omega_{CL}$"}
PK_GRID = [["V1Hat", "omegaV1", "sigma1"], ["kaHat", "omegaka", "beta"], ["CLHat", "omegaCL", None]]


def paper_style(size=24):
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    matplotlib.rcParams.update({"font.size": size, "axes.labelsize": size, "axes.titlesize": size,
                                "xtick.labelsize": size - 4, "ytick.labelsize": size - 4, "legend.fontsize": size - 4,
                                "lines.linewidth": 3, "axes.linewidth": 3, "axes.spines.right": False,
                                "axes.spines.top": False, "pdf.fonttype": 42})


def fits(results):
    """All summarized fits as (record, fit directory)."""
    for path in sorted(Path(results).glob("*/*/*/replicate_*/derived.json")):
        yield json.loads(path.read_text()), path.parent


# ------------------------------------------------------------------ Figures 3 and 4

def selection_table(results):
    rows = [dict(design=r["design"], hypothesis=r["hypothesis"], prior=r["prior"], replicate=r["replicate"], **s)
            for r, _ in fits(results) for s in r["selection"]]
    return pd.DataFrame(rows)


def group_figure(summary, ylabel, ylim, yticks, path):
    plt.rcParams.update(matplotlib.rcParamsDefault)
    plt.rcParams.update({"font.size": 10.5, "axes.labelsize": 11.5, "axes.titlesize": 12.5, "xtick.labelsize": 10.5,
                         "ytick.labelsize": 10, "legend.fontsize": 9.5, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 2.0})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.255, top=0.85, wspace=0.14)
    x, offsets = np.arange(3.0), np.linspace(-0.09, 0.09, len(GROUP_STYLES))
    handles = []
    for ax, design in zip(axes, ["p134", "p800"]):
        for j, (prior, label, color, marker) in enumerate(GROUP_STYLES):
            d = summary[(summary.design == design) & (summary.prior == prior)].set_index("level").reindex(LEVELS)
            h = ax.errorbar(x + offsets[j], d["value"], yerr=np.vstack([d["lower"], d["upper"]]), label=label,
                            color=color, marker=marker, markersize=6, markerfacecolor="white", markeredgewidth=1.2,
                            linestyle="none", elinewidth=1.05, capsize=3, capthick=1.05, zorder=3 + j / 10)
            if design == "p134":
                handles.append(h)
        ax.set_title(rf"$p={design[1:]}$", pad=10)
        ax.set_xticks(x, ["Single SNPs", "LD", "Griffin"])
        ax.set_xlim(-0.27, 2.27)
        ax.set_ylim(*ylim)
        ax.set_yticks(yticks)
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.7, zorder=0)
        ax.tick_params(axis="both", length=5, width=2.0)
        ax.set_axisbelow(True)
    axes[0].set_ylabel(ylabel)
    order = [0, 3, 1, 4, 2, 5]
    fig.legend([handles[i] for i in order], [GROUP_STYLES[i][1] for i in order], loc="lower center",
               bbox_to_anchor=(0.5, 0.025), ncol=3, frameon=False, handlelength=2.3, columnspacing=2.3,
               labelspacing=0.85)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def figure3(results, out):
    """Global-null FWER: share of H0 datasets with at least one selection, 95% Wilson intervals."""
    table = selection_table(results)
    null = table[table.hypothesis == "H0"].assign(any=lambda d: (d.fp > 0).astype(int))
    summary = null.groupby(["design", "prior", "level"])["any"].agg(["sum", "count", "mean"]).reset_index()
    z, n, rate = 1.959963984540054, summary["count"], summary["mean"]
    center = (rate + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    summary = summary.assign(value=rate, lower=np.maximum(0, rate - np.maximum(0, center - half)),
                             upper=np.maximum(0, np.minimum(1, center + half) - rate))
    top = max(0.26, float((summary.value + summary.upper).max()) + 0.02)
    group_figure(summary, "Global-null FWER", (-0.008, top), np.arange(0, top, 0.05 if top < 0.45 else 0.1),
                 out / "fig3_group_fwer.pdf")
    summary.to_csv(out / "fig3_group_fwer.csv", index=False)


def figure4(results, out):
    """Mean F1-score under H1 with one standard error."""
    table = selection_table(results)
    summary = (table[table.hypothesis == "H1"].groupby(["design", "prior", "level"])["f1"]
               .agg(["mean", "sem", "count"]).reset_index())
    summary = summary.assign(value=summary["mean"], lower=summary["sem"], upper=summary["sem"])
    low = float(np.floor(10 * min(0.4, (summary.value - summary.lower).min())) / 10)
    group_figure(summary, r"Mean $F_1$-score", (low, 1.025), np.arange(low, 1.01, 0.1), out / "fig4_group_f1.pdf")
    summary.to_csv(out / "fig4_group_f1.csv", index=False)


# ------------------------------------------------------------------ Figure S1

def figure_s1(results, out):
    """Prior calibration at p=800: effect densities, model sizes, gamma prior, beta-binomial K_z."""
    draws = np.load(Path(results) / "calibration" / "p800_prior_draws.npz")
    hyper = load_config()["designs"]["p800"]["spike-slab"]
    a, b, p = hyper["theta_a"], hyper["theta_b"], 800
    shrinkage = ("l1-ball", "hierarchical-lasso", "horseshoe", "r2d2")
    paper_style()
    fig, axes = plt.subplots(2, 2, figsize=(13, 10.5))
    x = np.linspace(-0.0075, 0.0075, 2000)
    for prior in shrinkage:
        axes[0, 0].plot(x, gaussian_kde(draws[prior + "__beta"])(x), color=COLORS[prior], zorder=ZORDER[prior])
    axes[0, 0].set(xlim=(-0.0075, 0.0075), ylim=(-40, 1500), xlabel=r"$\beta$", ylabel="Density", xticks=[])
    for prior in ("hierarchical-lasso", "horseshoe", "r2d2", "l1-ball"):
        values = draws[prior + "__size"]
        bins = np.arange(-0.5, 12.5) if prior == "l1-ball" else np.linspace(0, 11, 23)
        axes[0, 1].hist(values, bins=bins, weights=np.full(values.size, 1 / (values.size * np.diff(bins)[0])),
                        alpha=0.4, color=COLORS[prior], zorder=ZORDER[prior])
        axes[0, 1].axvline(values.mean(), linestyle="--", color=COLORS[prior], zorder=ZORDER[prior])
    axes[0, 1].set(xlabel=r"$K$ or $m_{\mathrm{eff}}$", ylabel="Density", xlim=(-1, 11))
    g = np.linspace(0, beta_dist.ppf(0.999, a, b), 500)
    axes[1, 0].plot(g, beta_dist.pdf(g, a, b), color="purple")
    axes[1, 0].fill_between(g, 0, beta_dist.pdf(g, a, b), alpha=0.4, color="purple")
    axes[1, 0].axvline(a / (a + b), linestyle="--", color="black")
    axes[1, 0].set(xlabel=r"$\gamma$", ylabel="Density", xlim=(0, g[-1]))
    k = np.arange(21)
    axes[1, 1].bar(k, betabinom.pmf(k, p, a, b), color="purple", alpha=0.4, width=0.85)
    axes[1, 1].axvline(p * a / (a + b), linestyle="--", color="black")
    axes[1, 1].set(xlabel=r"$K_z$", ylabel="Probability", xlim=(-0.7, 20.7))
    for ax, letter in zip(axes.flat, "abcd"):
        ax.set_title(f"({letter})", loc="left")
    handles = [Line2D([], [], color=COLORS[m], linewidth=3) for m in shrinkage] + [
        Line2D([], [], color="purple", linewidth=3)]
    fig.legend(handles, [NAMES[m] for m in shrinkage] + ["Both spike-and-slab priors"], loc="lower center",
               ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    fig.savefig(out / "figS1_prior_calibration_p800.pdf", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ Figures S2 and S3

def figures_s2_s3(results, out):
    """PK posteriors (pooled over datasets) and split R-hat at p=134."""
    draws, rhat = {}, []
    for record, fit_dir in fits(results):
        if record["design"] != "p134":
            continue
        key = (record["prior"], record["hypothesis"])
        with np.load(fit_dir / "derived.npz") as derived:
            for name, (site, log_scale) in PK_PARAMS.items():
                draws.setdefault(key + (name,), []).append(np.exp(derived[site]) if log_scale else derived[site])
            draws.setdefault(key + ("beta",), []).append(derived["beta_reference"])
        values = record["rhat"]
        for name, (site, _) in PK_PARAMS.items():
            rhat.append(dict(prior=record["prior"], hypothesis=record["hypothesis"], param=name, rhat=values[site]))
        rhat.append(dict(prior=record["prior"], hypothesis=record["hypothesis"], param="beta",
                         rhat=values["beta_reference"]))
    draws = {key: np.concatenate(value) for key, value in draws.items()}
    rhat = pd.DataFrame(rhat)
    rng = np.random.default_rng(2605)
    for hypothesis in ("H0", "H1"):
        paper_style()
        fig, axes = plt.subplots(3, 3, figsize=(14, 8))
        for r in range(3):
            for c in range(3):
                ax, param = axes[r, c], PK_GRID[c][r]
                series = [(m, draws[(m, hypothesis, param)]) for m in ORDER if (m, hypothesis, param) in draws]
                if param is None or not series:
                    ax.set_visible(False)
                    continue
                lo, hi = np.quantile(np.concatenate([v for _, v in series]), [0.001, 0.999])
                truth = None if (param == "beta" and hypothesis == "H0") else TRUE_PK[param]
                if truth is not None:
                    lo, hi = min(lo, truth), max(hi, truth)
                grid = np.linspace(lo - 0.05 * (hi - lo + 1e-12), hi + 0.05 * (hi - lo + 1e-12), 400)
                for m, v in series:
                    if np.ptp(v) > 0:
                        ax.plot(grid, gaussian_kde(v)(grid), color=COLORS[m], linewidth=2.5, zorder=ZORDER[m])
                    else:  # e.g. point-mass draws all exactly zero
                        ax.axvline(v[0], color=COLORS[m], linewidth=2.5, zorder=ZORDER[m])
                if truth is not None:
                    ax.axvline(truth, color="black", linestyle="--", linewidth=2.5, alpha=0.9, zorder=0)
                ax.set_title(PK_LABELS[param])
        fig.supylabel(f"{hypothesis} - Density")
        if hypothesis == "H0":
            fig.legend([Line2D([0], [0], color=COLORS[m], lw=3) for m in ORDER], [NAMES[m] for m in ORDER],
                       loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.07))
        plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=0.4, w_pad=0.4)
        fig.savefig(out / f"figS2_pk_posteriors_{hypothesis}.pdf", bbox_inches="tight")
        plt.close(fig)

        paper_style()
        frame = rhat[rhat.hypothesis == hypothesis]
        fig, axes = plt.subplots(3, 3, figsize=(14, 8), sharey=True)
        for r in range(3):
            for c in range(3):
                ax, param = axes[r, c], PK_GRID[r][c]
                if param is None:
                    ax.set_visible(False)
                    continue
                values = [frame[(frame.param == param) & (frame.prior == m)].rhat.values for m in ORDER]
                boxes = ax.boxplot(values, positions=np.arange(1, 7), widths=0.55, patch_artist=True,
                                   showfliers=False)
                for j, m in enumerate(ORDER):
                    color = COLORS[m]
                    boxes["boxes"][j].set(facecolor=color, alpha=0.25, edgecolor=color, linewidth=1.2)
                    boxes["medians"][j].set(color=color, linewidth=1.6)
                    for whisker in boxes["whiskers"][2 * j:2 * j + 2]:
                        whisker.set(color=color, linewidth=1.2)
                    ax.scatter(j + 1 + rng.uniform(-0.06, 0.06, len(values[j])), values[j], s=10, color=color,
                               alpha=0.7, linewidths=0)
                for cap in boxes["caps"]:
                    cap.set(linewidth=1.2, color="0.35")
                ax.axhline(1.1, color="0.4", linestyle="--", linewidth=1.5)
                ax.set(ylim=(0.95, 1.2), xticks=[])
                ax.set_title(PK_LABELS[param])
        fig.supylabel(r"$\widehat{R}$" + f"  ({hypothesis})")
        if hypothesis == "H0":
            fig.legend([Patch(facecolor=COLORS[m], edgecolor=COLORS[m], alpha=0.25) for m in ORDER],
                       [NAMES[m] for m in ORDER], loc="upper center", ncol=3, frameon=False,
                       bbox_to_anchor=(0.5, 1.07))
        fig.tight_layout(h_pad=0.4, w_pad=0.4)
        fig.savefig(out / f"figS3_rhat_{hypothesis}.pdf", bbox_inches="tight")
        plt.close(fig)


# ------------------------------------------------------------------ Figure S4

def figure_s4(results, out):
    """Inclusion probabilities against the effect tolerance, reference SNP in pink (C)."""
    curves, grids, focal = {}, {}, {}
    for record, fit_dir in fits(results):
        key = (record["design"], record["hypothesis"], record["prior"])
        with np.load(fit_dir / "derived.npz") as derived:
            curves.setdefault(key, []).append(derived["pip_curves"])
            grids[key] = derived["epsilon"]
            focal[key] = int(derived["reference_index"])
    plt.rcParams.update(matplotlib.rcParamsDefault)
    plt.rcParams.update({"axes.linewidth": 2.0, "axes.spines.top": False, "axes.spines.right": False,
                         "xtick.major.width": 2.0, "ytick.major.width": 2.0, "pdf.fonttype": 42})
    designs = sorted({key[0] for key in curves})
    fig = plt.figure(figsize=(18, 6.5 * len(designs)), layout="constrained")
    subfigures = fig.subfigures(len(designs), 1, hspace=0.08) if len(designs) > 1 else [fig]
    for sub, design in zip(np.atleast_1d(subfigures), designs):
        axes = sub.subplots(2, 6, sharey="row")
        for hi, hypothesis in enumerate(("H0", "H1")):
            for mi, prior in enumerate(ORDER):
                ax, key = axes[hi, mi], (design, hypothesis, prior)
                if key not in curves:
                    ax.set_visible(False)
                    continue
                pips, f = np.stack(curves[key]), focal[key]
                if prior in SPIKE_SLABS:
                    values = pips[:, 0, :]
                    others = [j for j in np.argsort(-values.mean(axis=0), kind="stable") if j != f][:9]
                    art = ax.boxplot([values[:, f]] + [values[:, j] for j in others], patch_artist=True,
                                     showfliers=False, widths=0.6)
                    for j, color in enumerate(["hotpink"] + [".5"] * 9):
                        art["boxes"][j].set(facecolor=color, edgecolor=color, alpha=0.35)
                        for part in ("whiskers", "caps"):
                            for line in art[part][2 * j:2 * j + 2]:
                                line.set(color=color, linewidth=1.5)
                        art["medians"][j].set(color=color, linewidth=2)
                    ax.set_xticks(range(1, 11), ["C"] + [str(j) for j in range(1, 10)], fontsize=15)
                    ax.get_xticklabels()[0].set_color("deeppink")
                    if hi == 1:
                        ax.set_xlabel("SNP rank (C: causal)", fontsize=16)
                else:
                    grid = grids[key]
                    mean = pips.mean(axis=0)
                    for j in range(mean.shape[1]):
                        if j != f:
                            ax.plot(grid, mean[:, j], color=".5", lw=0.5, alpha=0.35, rasterized=True)
                    sem = pips[:, :, f].std(axis=0, ddof=1) / np.sqrt(len(pips)) if len(pips) > 1 else 0 * grid
                    ax.plot(grid, mean[:, f], color="hotpink", lw=2.5)
                    ax.fill_between(grid, np.clip(mean[:, f] - sem, 0, 1), np.clip(mean[:, f] + sem, 0, 1),
                                    color="hotpink", alpha=0.25, lw=0)
                    ax.axvline(0.01, color=".15", ls="--", lw=1.2)
                    ax.set(xscale="log", xlim=(grid.min(), grid.max()))
                    if hi == 1:
                        ax.set_xlabel(r"Effect tolerance $\varepsilon$", fontsize=16)
                ax.set_ylim(-0.02, 1.02)
                if hi == 0:
                    ax.set_title(NAMES[prior], fontsize=17)
                ax.tick_params(labelsize=15)
                ax.grid(alpha=0.18)
            axes[hi, 0].set_ylabel(f"{hypothesis}\nInclusion probability", fontsize=17)
        sub.suptitle(f"$p={design[1:]}$", fontsize=22)
    fig.savefig(out / "figS4_pip_tolerance.pdf", bbox_inches="tight", dpi=200)
    plt.close(fig)


FIGURES = {"3": figure3, "4": figure4, "S1": figure_s1, "S2": figures_s2_s3, "S3": figures_s2_s3, "S4": figure_s4}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=str(ROOT / "results"))
    parser.add_argument("--out", default=str(ROOT / "figures"))
    parser.add_argument("--only", nargs="+", choices=list(FIGURES), default=list(FIGURES))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    done = set()
    for name in args.only:
        if FIGURES[name] not in done:
            FIGURES[name](args.results, out)
            done.add(FIGURES[name])
            print(f"Figure {'S2 and S3' if name in ('S2', 'S3') else name} written to {out}")


if __name__ == "__main__":
    main()
