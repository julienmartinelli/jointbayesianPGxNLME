"""Posterior inclusion probabilities and selection of SNPs or groups of SNPs.

Inclusion events are z_j = 1 for the spike-and-slab priors and |beta_j| > eps
for the other priors. Groups are either fixed LD components (|corr| > 0.9) or
Cartesian blocks learned from joint posterior inclusion draws (Griffin, 2025,
Bayesian Analysis 20(4):1371-1397), following the released implementation:
greedy pattern deletion scored by n * p_min / mass, partition score
log(cardinality) + sum_k [log M + log Gamma(|B_k|)], level 0.95 and M = 2.
"""

import heapq
from dataclasses import dataclass
from math import lgamma

import numpy as np

EPSILON_GRID = np.unique(np.concatenate((np.logspace(-4, 0, 80), np.logspace(-3, 0, 100),
                                         [0.001, 0.005, 0.01, 0.02, 0.05])))


def inclusion_events(samples, prior, epsilon=0.01, seed=2605):
    """Binary inclusion draws (chain, draw, p) and, for continuous spike-and-slab, their probabilities.

    For continuous spike-and-slab, one Bernoulli event is drawn per posterior
    draw from p(z_j = 1 | beta_j, theta, omegaCL); the probabilities are also
    returned so that group scores can be Rao-Blackwellized.
    """
    if prior == "point-mass-spike-slab":
        return samples["z"].astype(np.uint8), None
    if prior == "spike-slab":
        weights = samples["incl_prob"].astype(float)
        return (np.random.default_rng(seed).random(weights.shape) < weights).astype(np.uint8), weights
    return (np.abs(samples["beta"]) > epsilon).astype(np.uint8), None


def pip(samples, prior, epsilon=0.01):
    """Marginal inclusion probabilities: p(z_j = 1 | D) or p(|beta_j| > eps | D)."""
    if prior == "point-mass-spike-slab":
        return samples["z"].mean(axis=(0, 1))
    if prior == "spike-slab":
        return samples["incl_prob"].astype(float).mean(axis=(0, 1))
    return (np.abs(samples["beta"]) > epsilon).mean(axis=(0, 1))


def ld_components(X, threshold=0.9):
    """Connected components of the graph linking SNPs with |corr| > threshold."""
    X = np.asarray(X, dtype=float)
    centered = X - X.mean(axis=0)
    norm = np.sqrt((centered**2).sum(axis=0))
    unit = np.divide(centered, norm, out=np.zeros_like(centered), where=norm > 0)
    linked = np.abs(np.clip(unit.T @ unit, -1, 1)) > threshold
    remaining, groups = set(range(X.shape[1])), []
    while remaining:
        todo, group = [min(remaining)], set()
        while todo:
            j = todo.pop()
            if j in remaining:
                remaining.remove(j)
                group.add(j)
                todo.extend(int(k) for k in np.flatnonzero(linked[j]) if k in remaining)
        groups.append(tuple(sorted(group)))
    return tuple(groups)


# ---------------------------------------------------------------- Cartesian credible sets

@dataclass(frozen=True)
class Block:
    indices: tuple
    patterns: np.ndarray     # observed inclusion patterns of the block
    probability: np.ndarray  # their empirical probabilities
    labels: np.ndarray       # pattern index of every draw
    entropy: float


def _block(events, indices):
    patterns, labels, count = np.unique(events[:, indices], axis=0, return_inverse=True, return_counts=True)
    probability = count / count.sum()
    return Block(indices, patterns, probability, labels.ravel(), float(-np.dot(probability, np.log(probability))))


def _mutual_information(a, b):
    _, count = np.unique(a.labels * len(b.probability) + b.labels, return_counts=True)
    probability = count / count.sum()
    return max(0.0, a.entropy + b.entropy + float(np.dot(probability, np.log(probability))))


def _merge(a, b):
    code, labels, count = np.unique(a.labels * len(b.probability) + b.labels,
                                    return_inverse=True, return_counts=True)
    indices = a.indices + b.indices
    patterns = np.concatenate((a.patterns[code // len(b.probability)], b.patterns[code % len(b.probability)]),
                              axis=1)[:, np.argsort(indices)]
    probability = count / count.sum()
    return Block(tuple(sorted(indices)), patterns, probability, labels.ravel(),
                 float(-np.dot(probability, np.log(probability))))


def _credible_set(blocks, level, M):
    """Greedy deletion of the least probable patterns while the factorized mass stays >= level."""
    order = [np.argsort(b.probability, kind="stable") for b in blocks]
    removed = np.zeros(len(blocks), dtype=int)
    mass = np.ones(len(blocks))
    log_mass, heap = 0.0, []

    def push(i):
        n = len(order[i]) - removed[i]
        if n > 1:
            heapq.heappush(heap, (float(blocks[i].probability[order[i][removed[i]]] * n / mass[i]), i))

    for i in range(len(blocks)):
        push(i)
    while heap:
        _, i = heapq.heappop(heap)
        p_min = blocks[i].probability[order[i][removed[i]]]
        next_log_mass = log_mass + float(np.log1p(-p_min / mass[i]))
        if next_log_mass < np.log(level) - 1e-12:
            break
        mass[i] -= p_min
        log_mass = next_log_mass
        removed[i] += 1
        push(i)
    retained = tuple(o[r:] for o, r in zip(order, removed))
    score = sum(np.log(len(k)) for k in retained) + sum(np.log(M) + lgamma(len(b.indices)) for b in blocks)
    return blocks, retained, float(score)


def cartesian_blocks(events, level=0.95, M=2.0):
    """Learn the partition along the mutual-information merge path; return the best (blocks, retained)."""
    events = np.asarray(events, dtype=np.uint8)
    active = {j: _block(events, (j,)) for j in range(events.shape[1])}
    heap = [(-_mutual_information(active[i], active[j]), i, j)
            for i in active for j in range(i + 1, len(active))]
    heapq.heapify(heap)
    best = None
    while True:
        candidate = _credible_set(tuple(sorted(active.values(), key=lambda b: b.indices)), level, M)
        if best is None or candidate[2] < best[2] - 1e-12:
            best = candidate
        if len(active) == 1:
            return best[0], best[1]
        while True:
            _, i, j = heapq.heappop(heap)
            if i in active and j in active:
                break
        merged = _merge(active.pop(i), active.pop(j))
        new_id = events.shape[1] + (events.shape[1] - len(active) - 2)
        for k, other in active.items():
            heapq.heappush(heap, (-_mutual_information(other, merged), k, new_id))
        active[new_id] = merged


def _pattern_probability(weights, patterns):
    """Probability, within each draw, that independent Bernoulli(weights) events match one of the patterns."""
    if not len(patterns):
        return np.zeros(len(weights))
    patterns = np.unique(patterns, axis=0)
    yes = np.log(weights, out=np.zeros_like(weights), where=weights > 0)
    no = np.log1p(-weights, out=np.zeros_like(weights), where=weights < 1)
    log_probability = (yes - no) @ patterns.T + no.sum(axis=1)[:, None]
    impossible = ((weights == 0).astype(float) @ patterns.T > 0) | ((weights == 1).astype(float) @ (1 - patterns).T > 0)
    log_probability[impossible] = -np.inf
    return np.clip(np.exp(log_probability).sum(axis=1), 0, 1)


def block_scores(blocks, retained, weights=None):
    """Score of every block: posterior mass of its retained nonzero patterns."""
    scores = []
    for block, keep in zip(blocks, retained):
        nonzero = keep[block.patterns[keep].any(axis=1)]
        if weights is None:
            scores.append(block.probability[nonzero].sum())
        else:
            scores.append(_pattern_probability(weights[:, list(block.indices)], block.patterns[nonzero]).mean())
    return np.clip(np.asarray(scores, dtype=float), 0, 1)


def selection_counts(scores, true_group, cutoff=0.5):
    """(TP, FP, FN, F1) when selecting every group with score > cutoff; true_group is None under H0."""
    selected = np.asarray(scores) > cutoff
    tp = int(selected[true_group]) if true_group is not None else 0
    fp = int(selected.sum()) - tp
    fn = int(true_group is not None) - tp
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return tp, fp, fn, f1
