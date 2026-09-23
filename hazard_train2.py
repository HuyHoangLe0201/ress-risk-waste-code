"""Hazard training versus remaining-life training, selected honestly.

The first pass reported the best window after looking at held-out cost, which is the exact
error this paper spends a section on.  Here the window is chosen inside the calibration
half by an inner split, the model is refitted on the whole calibration half at that window,
and only then is it scored on units it has never seen.  Both predictors get the same
features and the same protocol, and the whole thing repeats over several seeds.

The prediction under test is Theorem 3's: the cost-optimal rule thresholds the conditional
hazard, so a model trained to rank hazard should reach a frontier at least as good as one
trained for squared error on remaining life, even though its RMSE is meaningless.
"""
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
import os.path as _p; sys.path.insert(0, _p.join(_p.dirname(_p.abspath(__file__)), "shared"))
import makefigs6 as M                                            # noqa: E402
from sklearn.linear_model import LogisticRegression              # noqa: E402

WINDOWS = (5, 10, 15, 25, 40, 60)


def cost_at(paths, R, units, Tb, chi, rising, ngrid=160):
    lo = min(paths[i].min() for i in units)
    hi = max(paths[i].max() for i in units)
    acc = {i: (np.maximum.accumulate(paths[i]) if rising
               else np.minimum.accumulate(paths[i])) for i in units}
    best = np.inf
    for e in np.linspace(lo, hi, ngrid):
        w, f = [], []
        for i in units:
            hit = np.nonzero(acc[i] >= e)[0] if rising else np.nonzero(acc[i] <= e)[0]
            if len(hit) == 0 or R[i][hit[0]] <= 0:
                w.append(0.0); f.append(1.0)
            else:
                w.append(float(R[i][hit[0]])); f.append(0.0)
        om = float(np.mean(w)) / Tb
        if om < 1.0:
            best = min(best, (1.0 + chi * float(np.mean(f))) / (1.0 - om))
    return best


def haz_paths(X, R, fit_units, pred_units, w):
    A = np.vstack([X[i] for i in fit_units])
    y = np.concatenate([(R[i] <= w).astype(float) for i in fit_units])
    if y.sum() < 5 or y.sum() == len(y):
        return None
    clf = LogisticRegression(max_iter=2000).fit(A, y)
    return {i: clf.predict_proba(X[i])[:, 1] for i in pred_units}


def run(name, ld, chis=(3.0, 9.0, 30.0), seeds=(0, 1, 2, 3, 4)):
    seq, life = ld()
    X = M.design(seq)
    R = {i: np.arange(len(s) - 1, -1, -1.0) for i, s in enumerate(seq)}
    n = len(seq)
    print("=" * 78)
    print("%s   n=%d" % (name, n))
    print("%6s %6s %10s %10s %9s %8s" %
          ("chi", "seed", "ridge RUL", "hazard", "gain", "w chosen"))
    tally = {c: [] for c in chis}
    for chi in chis:
        for sd in seeds:
            p = np.random.default_rng(sd).permutation(n)
            cal, val = p[:n // 2], p[n // 2:]
            h = len(cal) // 2
            inner_fit, inner_sel = cal[:h], cal[h:]
            Tb_i = float(life[inner_sel].mean())
            Tb_v = float(life[val].mean())

            # window chosen inside the calibration half only
            scores = {}
            for w in WINDOWS:
                hp = haz_paths(X, R, inner_fit, inner_sel, w)
                if hp is None:
                    continue
                scores[w] = cost_at(hp, R, inner_sel, Tb_i, chi, rising=True)
            if not scores:
                continue
            w_star = min(scores, key=scores.get)

            hp = haz_paths(X, R, cal, val, w_star)
            L_h = cost_at(hp, R, val, Tb_v, chi, rising=True)

            wv = M.fit(np.vstack([X[i] for i in cal]),
                       np.concatenate([R[i] for i in cal]))
            rp = {i: X[i] @ wv for i in val}
            L_r = cost_at(rp, R, val, Tb_v, chi, rising=False)

            tally[chi].append(L_r - L_h)
            print("%6.0f %6d %10.4f %10.4f %+9.4f %8d"
                  % (chi, sd, L_r, L_h, L_r - L_h, w_star))
        d = np.array(tally[chi])
        print("%6.0f %6s mean gain %+.4f  hazard better in %d of %d seeds"
              % (chi, "", d.mean(), int((d > 0).sum()), len(d)))


if __name__ == "__main__":
    for nm, ld in (("FD002", M.FD002), ("FD004", M.FD004),
                   ("Severson", M.load_severson)):
        run(nm, ld)
