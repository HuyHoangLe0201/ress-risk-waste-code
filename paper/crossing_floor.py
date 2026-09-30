"""Are the predictor-frontier crossings resolvable, or are they below the paper's own floor?

Theorem 3 says two predictors either order at every cost ratio or cross, and the experiment
reported that the RMSE-best family is not the cost-best family in 11 of 15 fleet-seed
combinations.  That claim is mine and it was never held to this paper's own standard.  A
crossing is only real if the envelope difference at the operating cost ratio exceeds what a
fleet of this size can resolve.

The floor comes from the paper's own machinery: resample UNITS, not splits, and read the
standard error of the DIFFERENCE between two envelopes at a fixed cost ratio.  Both
envelopes are recomputed on each resampled fleet, so the pairing is preserved and the
common fleet-draw variation cancels, which is the whole point of the paired design in
Section 5.

If most pairwise differences sit inside two standard errors, the ranking of predictors is
not established at these fleet sizes and the earlier claim has to be reported as a direction
rather than an ordering.

Grid note.  The envelope is a minimum over thresholds, so a coarse grid returns a value
above the true minimum by an amount that differs per predictor.  At 120 points those
errors were the same size as the differences being measured -- the pairwise gaps moved
by a factor of two between grids -- so the grid is set to 400, where they are stable.
"""
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
import os.path as _p; sys.path.insert(0, _p.join(_p.dirname(_p.abspath(__file__)), "shared"))
import makefigs6 as M                                            # noqa: E402
from sklearn.neighbors import KNeighborsRegressor                # noqa: E402
from sklearn.neural_network import MLPRegressor                  # noqa: E402


def predictors(X, R, cal, val, seed):
    A = np.vstack([X[i] for i in cal])
    y = np.concatenate([R[i] for i in cal])
    rg = np.random.default_rng(seed)
    out = {}
    w = M.fit(A, y)
    out["ridge"] = {i: X[i] @ w for i in val}
    d = A.shape[1]
    W = rg.normal(0, 1.0 / np.sqrt(d), (d, 300))
    b = rg.uniform(0, 2 * np.pi, 300)
    phi = lambda Z: np.sqrt(2.0 / 300) * np.cos(Z @ W + b)        # noqa: E731
    wr = M.fit(phi(A), y)
    out["randFourier"] = {i: phi(X[i]) @ wr for i in val}
    kn = KNeighborsRegressor(n_neighbors=15).fit(A, y)
    out["kNN15"] = {i: kn.predict(X[i]) for i in val}
    nn = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300,
                      random_state=seed, early_stopping=True).fit(A, y)
    out["network"] = {i: nn.predict(X[i]) for i in val}
    return out


def envelope_at(pa, R, units, chi, ngrid=400):
    acc = {i: np.minimum.accumulate(pa[i]) for i in units}
    lo = min(acc[i].min() for i in units)
    hi = max(acc[i].max() for i in units)
    Tb = float(np.mean([len(pa[i]) for i in units]))
    best = np.inf
    for e in np.linspace(lo, hi, ngrid):
        w, f = [], []
        for i in units:
            h = np.nonzero(acc[i] <= e)[0]
            if len(h) == 0 or R[i][h[0]] <= 0:
                w.append(0.0); f.append(1.0)
            else:
                w.append(float(R[i][h[0]])); f.append(0.0)
        om = float(np.mean(w)) / Tb
        if om < 1:
            best = min(best, (1 + chi * float(np.mean(f))) / (1 - om))
    return best


def run(name, ld, chi=9.0, boots=100, seed=0):
    seq, life = ld()
    X = M.design(seq)
    R = {i: np.arange(len(s) - 1, -1, -1.0) for i, s in enumerate(seq)}
    n = len(seq)
    p = np.random.default_rng(seed).permutation(n)
    cal, val = p[:n // 2], p[n // 2:]
    P = predictors(X, R, cal, val, seed)
    names = list(P)

    point = {k: envelope_at(P[k], R, val, chi) for k in names}
    rng = np.random.default_rng(seed + 7)
    draws = {k: [] for k in names}
    for _ in range(boots):
        bs = rng.choice(val, len(val), replace=True)     # resample UNITS
        for k in names:
            draws[k].append(envelope_at(P[k], R, bs, chi))

    print("=" * 78)
    print("%s   chi=%.0f, %d unit-level resamples of %d held-out units"
          % (name, chi, boots, len(val)))
    print("%14s %14s %10s %9s %9s" %
          ("A", "B", "L_A - L_B", "s.e.", "resolved"))
    for x in range(len(names)):
        for y in range(x + 1, len(names)):
            a, b = names[x], names[y]
            d = point[a] - point[b]
            dd = np.array(draws[a]) - np.array(draws[b])       # paired difference
            se = float(np.std(dd, ddof=1))
            print("%14s %14s %10.4f %9.4f %9s"
                  % (a, b, d, se, "yes" if abs(d) > 2 * se else "no"))


if __name__ == "__main__":
    for nm, ld in (("FD002", M.FD002), ("FD004", M.FD004),
                   ("Severson", M.load_severson)):
        run(nm, ld)
