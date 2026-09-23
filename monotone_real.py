"""Does Theorem 3's sufficient condition hold on the fleets, and does its failure cost?

The condition is that the conditional hazard be non-decreasing along paths.  On a chain
built to violate it the one-step rule throws away a third of the cost.  Whether that
pathology is a constructed curiosity or something these fleets actually exhibit is a
separate question, and it is answerable with the fitted hazard from the training
experiment.

Two measurements.  First, how far the fitted hazard is from monotone along paths: the
fraction of cycles at which it decreases.  Second, and this is the one that matters,
whether the failure of the condition costs anything here -- the one-step rule stopping at
the first crossing of g*/(c_f-c_p) against the best threshold on the running maximum, which
is the rule policy C actually uses.  If the two agree the condition's failure is harmless
on this data; if the one-step rule is much worse, the chain counterexample is describing
something real.

Note the two rules are different objects, as Section 3.4 now says explicitly: the running
maximum makes a threshold rule absorbing, which is not the same as making the region in the
theorem absorbing.
"""
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
import os.path as _p; sys.path.insert(0, _p.join(_p.dirname(_p.abspath(__file__)), "shared"))
import makefigs6 as M                                            # noqa: E402
from sklearn.linear_model import LogisticRegression              # noqa: E402


def fit_hazard(X, R, cal, val, w):
    A = np.vstack([X[i] for i in cal])
    y = np.concatenate([(R[i] <= w).astype(float) for i in cal])
    if y.sum() < 5:
        return None
    clf = LogisticRegression(max_iter=2000).fit(A, y)
    return {i: clf.predict_proba(X[i])[:, 1] for i in val}


def threshold_cost(haz, R, val, Tb, chi, ngrid=400):
    """Best cost over levels of the RUNNING MAXIMUM: the rule policy C uses.

    Candidate levels are quantiles of the observed hazard values, not a uniform grid.
    The fitted hazard sits near zero on almost every cycle with a thin spike at the end,
    so a uniform grid places nearly all its points where nothing happens and misses the
    optimum by a wide margin -- enough, before this was fixed, to make a rule that
    minimises over levels look worse than a particular level.
    """
    acc = {i: np.maximum.accumulate(haz[i]) for i in val}
    allv = np.concatenate([acc[i] for i in val])
    levels = np.unique(np.quantile(allv, np.linspace(0.0, 1.0, ngrid)))
    best, arg = np.inf, levels[0]
    for e in levels:
        wst, f = [], []
        for i in val:
            h = np.nonzero(acc[i] >= e)[0]
            if len(h) == 0 or R[i][h[0]] <= 0:
                wst.append(0.0); f.append(1.0)
            else:
                wst.append(float(R[i][h[0]])); f.append(0.0)
        om = float(np.mean(wst)) / Tb
        if om < 1:
            c = (1 + chi * float(np.mean(f))) / (1 - om)
            if c < best:
                best, arg = c, e
    return best, arg


def window_level(level, w):
    """Convert Theorem 3's per-cycle level to the scale the fitted quantity is on.

    fit_hazard regresses the indicator {R <= w}, so haz is the probability of failing
    within w cycles, not a per-cycle intensity.  Theorem 3 thresholds the per-cycle
    intensity.  Comparing the two directly makes the rule stop far too early and
    charges the difference to the theorem: on FD002 at chi=9 that reports a 147 per
    cent penalty where the correct comparison gives 36.  Under a constant per-cycle
    hazard the two scales are related by 1-(1-h)^w.
    """
    return 1.0 - (1.0 - level) ** w


def ola_cost(haz, R, val, Tb, chi, level):
    """One-step rule: stop at the first crossing of the RAW hazard, not its running max."""
    wst, f = [], []
    for i in val:
        h = np.nonzero(haz[i] >= level)[0]
        if len(h) == 0 or R[i][h[0]] <= 0:
            wst.append(0.0); f.append(1.0)
        else:
            wst.append(float(R[i][h[0]])); f.append(0.0)
    om = float(np.mean(wst)) / Tb
    return np.inf if om >= 1 else (1 + chi * float(np.mean(f))) / (1 - om)


def run(name, ld, w=15, seed=0):
    seq, life = ld()
    X = M.design(seq)
    R = {i: np.arange(len(s) - 1, -1, -1.0) for i, s in enumerate(seq)}
    n = len(seq)
    p = np.random.default_rng(seed).permutation(n)
    cal, val = p[:n // 2], p[n // 2:]
    Tb = float(life[val].mean())
    haz = fit_hazard(X, R, cal, val, w)

    dec = np.concatenate([np.diff(haz[i]) for i in val])
    frac = 100.0 * float(np.mean(dec < 0))
    print("=" * 74)
    print("%s   fitted hazard decreases at %.1f%% of monitored cycles" % (name, frac))
    print("%8s %14s %14s %10s" % ("chi", "threshold rule", "one-step rule", "penalty"))
    for chi in (3.0, 9.0, 30.0):
        L_thr, _ = threshold_cost(haz, R, val, Tb, chi)
        lvl = (L_thr / Tb) / chi              # g*/(c_f-c_p) in these units
        L_ola = ola_cost(haz, R, val, Tb, chi, window_level(lvl, w))
        pen = (L_ola - L_thr) / L_thr * 100.0 if np.isfinite(L_ola) else np.inf
        print("%8.0f %14.4f %14.4f %9.1f%%" % (chi, L_thr, L_ola, pen))


if __name__ == "__main__":
    for nm, ld in (("FD002", M.FD002), ("FD004", M.FD004),
                   ("Severson", M.load_severson)):
        run(nm, ld)
