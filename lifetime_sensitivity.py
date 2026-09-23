r"""How much of the age-replacement baseline depends on the lifetime law chosen for it?

The identity of Lemma 1 assumes no lifetime law, and neither does kappa* -- that constant is
read off the threshold family's own envelope, so no distributional fit enters it. One
quantity in the paper does depend on a fit: the age-replacement baseline, which
Proposition 2 requires to be computed from a fitted law rather than from an order
statistic. Every comparison of the form g_CBM/g_age therefore inherits whatever that fit
assumes, and the manuscript fits a lognormal without saying what a different choice would
do. That is a fair question to be asked, and it is cheaper to answer than to argue about.

Four laws are put through the same pipeline: lognormal, Weibull and gamma, all with the
location pinned at zero, and the empirical survival function, which fits nothing. Only the
survival curve changes; the split, the predictor, the threshold grid and the scoring are
held identical, so a difference in the reported ratio is a difference between the laws and
nothing else.

Reported per fleet and law, over the same splits:

    tR / Tbar          the fleet-optimal replacement age
    Tbar g_age / c_p   the age baseline, scored on the held-out half
    g_CBM / g_age      the comparison the paper actually makes

The predictor and threshold search are reused from makefigs6 rather than rewritten, and
that module is imported the way the rest of the project imports it -- TR_paper first,
because that is the copy the reported numbers came from and it differs from the file of
the same name in this directory.
"""
import sys

import numpy as np
from scipy.stats import gamma, lognorm, weibull_min

import os.path as _p; sys.path.insert(0, _p.join(_p.dirname(_p.abspath(__file__)), "shared"))
import makefigs6 as M                                            # noqa: E402

CF = M.CF
NSPLIT = 40
NGRID = 4000


def survival(name, Lc, t):
    """S(t) under one law fitted to the calibration lifetimes Lc."""
    if name == "lognormal":
        s, _, sc = lognorm.fit(Lc, floc=0)
        return lognorm.sf(t, s, scale=sc)
    if name == "Weibull":
        c, _, sc = weibull_min.fit(Lc, floc=0)
        return weibull_min.sf(t, c, scale=sc)
    if name == "gamma":
        a, _, sc = gamma.fit(Lc, floc=0)
        return gamma.sf(t, a, scale=sc)
    if name == "empirical":
        # No fit at all: the right-continuous empirical survivor of the calibration half.
        # With run-to-failure data there is no censoring, so Kaplan--Meier reduces to this.
        return (Lc[None, :] > t[:, None]).mean(1)
    raise ValueError(name)


def best_age(S, t):
    """Fleet-optimal replacement age under survival S on grid t, and its cost rate.

    The objective is Lemma 1's, (1 + kappa F)/E[min(T,t)] with the mean cycle length taken
    as the trapezoid integral of S. The first ten grid points are skipped for the same
    reason the manuscript's own baseline skips them: near t = 0 the denominator is a
    vanishing integral and the ratio is numerically meaningless, not economically small.
    """
    cum = np.concatenate([[0.], np.cumsum((S[1:] + S[:-1]) / 2 * np.diff(t))])
    obj = (S + CF * (1 - S)) / np.maximum(cum, 1e-12)
    j = 10 + int(np.argmin(obj[10:]))
    return t[j], obj[j]


def run(fleet, loader, laws=("lognormal", "Weibull", "gamma", "empirical")):
    seq, life = loader()
    X = M.design(seq)
    R = [np.arange(len(s) - 1, -1, -1.0) for s in seq]
    n = len(seq)
    Tb = life.mean()
    out = {k: {"tR": [], "age": [], "ratio": []} for k in laws}

    for seed in range(NSPLIT):
        rg = np.random.default_rng(seed)
        p = rg.permutation(n)
        cal, val = p[:n // 2], p[n // 2:]
        Lc, Lv = life[cal], life[val]
        t = np.linspace(1., 4 * Lc.max(), NGRID)

        # The predictor and the threshold are fitted once per split and shared by all four
        # laws: the comparison is between baselines, so anything else must be held fixed.
        w = M.fit(np.vstack([X[i] for i in cal]),
                  np.concatenate([R[i] for i in cal]))
        pa = {i: X[i] @ w for i in np.concatenate([cal, val])}
        rmn = {i: np.minimum.accumulate(pa[i]) for i in pa}
        grid = np.linspace(0, .4 * Tb, 25)

        def cc(idx, e):
            per = np.array([M.unit_cost(i, e, R, rmn, life) for i in idx])
            return per[:, 0].mean() / per[:, 1].mean()

        cbm = cc(val, grid[int(np.argmin([cc(cal, e) for e in grid]))]) * Tb

        for name in laws:
            tR, _ = best_age(survival(name, Lc, t), t)
            # scored out of sample, exactly as the manuscript scores its baseline
            age = (np.where(Lv <= tR, CF, 1.).mean()
                   / np.minimum(Lv, tR).mean() * Tb)
            out[name]["tR"].append(tR / Tb)
            out[name]["age"].append(age)
            out[name]["ratio"].append(cbm / age)

    print("\n%s  (n = %d, Tbar = %.1f, %d splits)" % (fleet, n, Tb, NSPLIT))
    print("   %-11s %-16s %-16s %s"
          % ("law", "tR / Tbar", "Tbar g_age / c_p", "g_CBM / g_age"))
    base = None
    for name in laws:
        d = out[name]
        med = np.median(d["ratio"])
        if base is None:
            base = med
        print("   %-11s %-16s %-16s %s   (%+.1f%% vs lognormal)"
              % (name,
                 "%.3f [%.3f, %.3f]" % (np.median(d["tR"]),
                                        *np.percentile(d["tR"], [10, 90])),
                 "%.3f" % np.median(d["age"]),
                 "%.3f [%.3f, %.3f]" % (med, *np.percentile(d["ratio"], [10, 90])),
                 100 * (med - base) / base))
    return out


if __name__ == "__main__":
    run("FD002", M.FD002)
    run("FD004", M.FD004)
    run("Severson", M.load_severson)
