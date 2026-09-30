"""Is the RMSE-versus-cost disagreement stable, or an artefact of one seed?

The single-seed run said the RMSE-best family is not the cost-best family, and that
six of eighteen frontier pairs cross.  Two of the four families are stochastic and the
seed also sets the calibration split, so the finding has to survive reseeding before it
can be stated.  This repeats the whole comparison over several seeds and asks one blunt
question per seed: which family has the lowest RMSE, and which has the lowest cost at
each of three operating points?  If those columns disagree consistently, ranking
predictors by accuracy is not a shortcut but a mistake.

Only held-out units are predicted, which is all the frontier and the RMSE need.
"""
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
import os.path as _p; sys.path.insert(0, _p.join(_p.dirname(_p.abspath(__file__)), "shared"))
import makefigs6 as M                                            # noqa: E402
from sklearn.neighbors import KNeighborsRegressor                # noqa: E402
from sklearn.neural_network import MLPRegressor                  # noqa: E402
from crossing import frontier, env                               # noqa: E402


def predictors(X, R, cal, val, seed):
    Atr = np.vstack([X[i] for i in cal])
    ytr = np.concatenate([R[i] for i in cal])
    rg = np.random.default_rng(seed)
    out = {}
    w = M.fit(Atr, ytr)
    out["ridge"] = {i: X[i] @ w for i in val}
    d = Atr.shape[1]
    W = rg.normal(0, 1.0 / np.sqrt(d), (d, 300))
    bb = rg.uniform(0, 2 * np.pi, 300)
    phi = lambda A: np.sqrt(2.0 / 300) * np.cos(A @ W + bb)       # noqa: E731
    wr = M.fit(phi(Atr), ytr)
    out["randFourier"] = {i: phi(X[i]) @ wr for i in val}
    kn = KNeighborsRegressor(n_neighbors=15).fit(Atr, ytr)
    out["kNN15"] = {i: kn.predict(X[i]) for i in val}
    nn = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300,
                      random_state=seed, early_stopping=True).fit(Atr, ytr)
    out["network"] = {i: nn.predict(X[i]) for i in val}
    return out


def run(name, ld, seeds=(0, 1, 2, 3, 4)):
    seq, life = ld()
    X = M.design(seq)
    R = {i: np.arange(len(s) - 1, -1, -1.0) for i, s in enumerate(seq)}
    n = len(seq)
    ops = (3.0, 9.0, 30.0)
    print("=" * 78)
    print("%s  n=%d" % (name, n))
    print("%6s %14s %14s %14s %14s %8s" %
          ("seed", "best RMSE", "best chi=3", "best chi=9", "best chi=30", "crosses"))
    dis = 0
    for sd in seeds:
        p = np.random.default_rng(sd).permutation(n)
        cal, val = p[:n // 2], p[n // 2:]
        Tb = float(life[val].mean())
        P = predictors(X, R, cal, val, sd)
        E, rms = {}, {}
        for k, pa in P.items():
            a, b = frontier(pa, R, val, Tb, n_thr=180)
            E[k] = env(np.array(ops), a, b)
            rms[k] = float(np.sqrt(np.mean(np.concatenate(
                [(pa[i] - R[i]) ** 2 for i in val])))) / Tb
        names = list(P)
        best_r = min(rms, key=rms.get)
        bests = [names[int(np.argmin([E[k][j] for k in names]))]
                 for j in range(len(ops))]
        grid = np.linspace(0.5, 60.0, 1500)
        Eg = {}
        for k, pa in P.items():
            a, b = frontier(pa, R, val, Tb, n_thr=180)
            Eg[k] = env(grid, a, b)
        cr = 0
        for x in range(len(names)):
            for y in range(x + 1, len(names)):
                d = Eg[names[x]] - Eg[names[y]]
                cr += int(np.sum(np.sign(d[:-1]) * np.sign(d[1:]) < 0))
        if any(b != best_r for b in bests):
            dis += 1
        print("%6d %14s %14s %14s %14s %8d"
              % (sd, best_r, bests[0], bests[1], bests[2], cr))
    print("    seeds where the RMSE-best family is NOT cost-best somewhere: %d/%d"
          % (dis, len(seeds)))


if __name__ == "__main__":
    for nm, ld in (("FD002", M.FD002), ("FD004", M.FD004),
                   ("Severson", M.load_severson)):
        run(nm, ld)
