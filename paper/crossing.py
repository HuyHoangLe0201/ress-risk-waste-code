"""Does 'which predictor is better' have an answer independent of the cost ratio?

The literature ranks predictors by RMSE and stops.  In this framework a predictor is
not a number but an achievable SET, and comparing two of them is comparing two lower
envelopes.  Convex duality says exactly when a ranking exists:

    L*_1(chi) <= L*_2(chi) for all chi > 0   <=>   U_2 subset U_1,

where U = conv(D) + R^2_+ is the upward closure of the dual point set D = {(b_p,a_p)}.
L*(chi) = min over D of a + b chi is the support function of U in direction (1,chi), and
two closed convex upward sets are ordered exactly when their support functions are.  So
either one predictor dominates at EVERY cost ratio, or the envelopes cross and each is
preferable on a non-empty open set of cost ratios.  There is no third case, and no
scalar accuracy score can decide between the second one's branches.

This script builds the paper's four predictor families on one feature convention, fits
each on the calibration half, traces each frontier on the validation half, and looks for
crossings.  If the RMSE order and the cost order disagree anywhere, ranking predictors by
accuracy alone is not a simplification but an error.
"""
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
import os.path as _p; sys.path.insert(0, _p.join(_p.dirname(_p.abspath(__file__)), "shared"))
import makefigs6 as M                                            # noqa: E402
from sklearn.neighbors import KNeighborsRegressor                # noqa: E402
from sklearn.neural_network import MLPRegressor                  # noqa: E402


def predictors(X, R, cal, seed=0):
    """Predicted-RUL paths for every unit, from four families fitted on `cal`."""
    Atr = np.vstack([X[i] for i in cal])
    ytr = np.concatenate([R[i] for i in cal])
    rg = np.random.default_rng(seed)
    out = {}

    w = M.fit(Atr, ytr)
    out["ridge"] = {i: X[i] @ w for i in range(len(X))}

    d = Atr.shape[1]
    W = rg.normal(0, 1.0 / np.sqrt(d), (d, 300))
    bb = rg.uniform(0, 2 * np.pi, 300)
    phi = lambda A: np.sqrt(2.0 / 300) * np.cos(A @ W + bb)       # noqa: E731
    wr = M.fit(phi(Atr), ytr)
    out["randomFourier"] = {i: phi(X[i]) @ wr for i in range(len(X))}

    kn = KNeighborsRegressor(n_neighbors=15).fit(Atr, ytr)
    out["kNN15"] = {i: kn.predict(X[i]) for i in range(len(X))}

    nn = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300,
                      random_state=seed, early_stopping=True).fit(Atr, ytr)
    out["network"] = {i: nn.predict(X[i]) for i in range(len(X))}
    return out


def frontier(pa, R, units, Tb, n_thr=220):
    """Achievable (omega, P_f) points for constant thresholds, on `units`."""
    rm = {i: np.minimum.accumulate(pa[i]) for i in units}
    lo = min(rm[i].min() for i in units)
    hi = max(rm[i].max() for i in units)
    A = []
    for e in np.linspace(lo, hi, n_thr):
        w, f = [], []
        for i in units:
            h = np.nonzero(rm[i] <= e)[0]
            if len(h) == 0 or R[i][h[0]] <= 0:
                w.append(0.0); f.append(1.0)
            else:
                w.append(float(R[i][h[0]])); f.append(0.0)
        A.append((float(np.mean(w)) / Tb, float(np.mean(f))))
    A = np.array(A)
    keep = A[:, 0] < 1.0
    a = 1.0 / (1.0 - A[keep, 0])
    b = A[keep, 1] / (1.0 - A[keep, 0])
    return a, b


def env(chi, a, b):
    return np.min(a[None, :] + b[None, :] * np.atleast_1d(chi)[:, None], axis=1)


def run(name, ld, seed=0):
    seq, life = ld()
    X = M.design(seq)
    R = {i: np.arange(len(s) - 1, -1, -1.0) for i, s in enumerate(seq)}
    n = len(seq)
    p = np.random.default_rng(seed).permutation(n)
    cal, val = p[:n // 2], p[n // 2:]
    Tb = float(life[val].mean())

    P = predictors(X, R, cal, seed=seed)
    grid = np.linspace(0.5, 60.0, 3000)

    print("=" * 78)
    print("%s   n=%d, frontier traced on the %d held-out units" % (name, n, len(val)))
    E, rms = {}, {}
    for k, pa in P.items():
        a, b = frontier(pa, R, val, Tb)
        E[k] = env(grid, a, b)
        rms[k] = float(np.sqrt(np.mean(np.concatenate(
            [(pa[i] - R[i]) ** 2 for i in val])))) / Tb

    order = sorted(rms, key=rms.get)
    print("    RMSE/Tbar order: " + " < ".join("%s %.3f" % (k, rms[k]) for k in order))

    names = list(P)
    print("%16s %16s %9s %9s %s" %
          ("A", "B", "A better", "B better", "crossings in [0.5,60]"))
    for x in range(len(names)):
        for y in range(x + 1, len(names)):
            u, v = names[x], names[y]
            d = E[u] - E[v]
            cr = int(np.sum(np.sign(d[:-1]) * np.sign(d[1:]) < 0))
            where = grid[:-1][np.sign(d[:-1]) * np.sign(d[1:]) < 0]
            s = ", ".join("%.1f" % z for z in where[:4]) if cr else "none: one dominates"
            print("%16s %16s %9.1f%% %9.1f%%  %s"
                  % (u, v, 100 * np.mean(d < 0), 100 * np.mean(d > 0), s))


if __name__ == "__main__":
    for nm, ld in (("FD002", M.FD002), ("FD004", M.FD004),
                   ("Severson", M.load_severson)):
        run(nm, ld)
