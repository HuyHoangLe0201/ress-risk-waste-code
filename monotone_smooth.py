"""Is the fitted intensity's non-monotonicity intrinsic, or is it estimation noise?

The fitted conditional intensity decreases at roughly 46 per cent of monitored cycles, and
applying Theorem 3's rule literally to it costs up to 3.7 times the best threshold.  That
leaves a question with a practical answer attached.  If the roughness is noise, smoothing
should restore the monotone hypothesis and shrink the penalty, and the one-step rule
becomes usable on a smoothed estimate.  If it is intrinsic, smoothing will not help however
wide the window.

The intervention is the paper's own: a moving average of increasing width applied to the
same fitted paths, changing roughness and nothing else.  Two things are tracked, the
fraction of cycles at which the smoothed intensity decreases and the cost of the one-step
rule against the best level on the same paths.

Edge handling matters here.  Zero padding drags the ends of every path down, which
manufactures exactly the spurious early crossings under study; the average is taken with
edge padding so that smoothing changes roughness alone.
"""
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
import os.path as _p; sys.path.insert(0, _p.join(_p.dirname(_p.abspath(__file__)), "shared"))
import makefigs6 as M                                            # noqa: E402
from sklearn.linear_model import LogisticRegression              # noqa: E402
from monotone_real import (fit_hazard, threshold_cost, ola_cost,  # noqa: E402
                           window_level)


def smooth(x, m):
    if m <= 1:
        return x
    pad = m // 2
    xp = np.concatenate([np.full(pad, x[0]), x, np.full(pad, x[-1])])
    k = np.ones(m) / m
    return np.convolve(xp, k, mode="valid")[:len(x)]


def run(name, ld, w=15, seed=0, chis=(3.0, 9.0, 30.0)):
    seq, life = ld()
    X = M.design(seq)
    R = {i: np.arange(len(s) - 1, -1, -1.0) for i, s in enumerate(seq)}
    n = len(seq)
    p = np.random.default_rng(seed).permutation(n)
    cal, val = p[:n // 2], p[n // 2:]
    Tb = float(life[val].mean())
    raw = fit_hazard(X, R, cal, val, w)

    print("=" * 78)
    print("%s" % name)
    print("%8s %12s %s" % ("window", "decreasing", "  one-step penalty vs best level"))
    print("%8s %12s %10s %10s %10s" % ("", "", "chi=3", "chi=9", "chi=30"))
    for m in (1, 3, 5, 11, 21, 41):
        haz = {i: smooth(raw[i], m) for i in val}
        dec = np.concatenate([np.diff(haz[i]) for i in val])
        frac = 100.0 * float(np.mean(dec < 0))
        pens = []
        for chi in chis:
            L_thr, _ = threshold_cost(haz, R, val, Tb, chi)
            lvl = (L_thr / Tb) / chi
            L_ola = ola_cost(haz, R, val, Tb, chi, window_level(lvl, w))
            pens.append((L_ola - L_thr) / L_thr * 100.0
                        if np.isfinite(L_ola) else np.inf)
        print("%8d %11.1f%% %9.1f%% %9.1f%% %9.1f%%"
              % (m, frac, pens[0], pens[1], pens[2]))


if __name__ == "__main__":
    for nm, ld in (("FD002", M.FD002), ("FD004", M.FD004),
                   ("Severson", M.load_severson)):
        run(nm, ld)
