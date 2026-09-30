r"""The unit-level bootstrap cannot see the boundary, and the failure has a fixed signature.

Section 2 already corrects one irregularity: the reported cost is a MINIMUM over a grid,
which is directionally but not fully differentiable, so by Fang and Santos the standard
bootstrap is inconsistent and the numerical delta method is used instead.  That argument
assumes the underlying estimator is asymptotically Gaussian and repairs the map applied to
it.

The zero-failure threshold is a different object and fails differently.  It is
ell_0 = max_u ell_u, an extreme of the sample, whose limit law is extreme-value and not
Gaussian at all, so the premise of the delta-method repair is itself absent.  For a maximum
the naive bootstrap is inconsistent in a way that is exactly quantifiable rather than merely
possible:

  - the unit attaining the maximum appears in a resample with probability 1-(1-1/n)^n,
    which tends to 1-1/e = 0.6321, and when it appears the bootstrap maximum EQUALS the
    observed one.  So the bootstrap distribution carries an atom of that mass at the point
    estimate, whatever the underlying law;
  - and a resample is a subset of the sample, so the bootstrap maximum can never EXCEED the
    observed maximum.  The bootstrap distribution lies entirely at or below a quantity that
    the sample underestimates, so the error it is trying to measure is on the side it
    cannot reach.

Both are parameter-free predictions.  They are checked here on the fleets, against an
average functional as a negative control -- if an atom appeared there too it would be an
artefact of the resampling code and not a property of the estimand.

The consequence is then priced where the truth is known.  Drawing ell_u from a law with a
finite endpoint, the coverage of the naive interval for ell_0 is measured against the
m-out-of-n bootstrap, which is consistent for extremes when m/n -> 0.
"""
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
import os.path as _p; sys.path.append(_p.join(_p.dirname(_p.abspath(__file__)), "shared"))
import makefigs6 as M                                            # noqa: E402
from quota import prep                                           # noqa: E402


def atom(lu, B=4000, m=None, seed=0):
    """(mass at the observed max, mass strictly above it) under an m-out-of-n bootstrap."""
    n = len(lu)
    m = n if m is None else m
    rng = np.random.default_rng(seed)
    obs = float(lu.max())
    draws = np.array([lu[rng.integers(0, n, m)].max() for _ in range(B)])
    return float(np.mean(draws == obs)), float(np.mean(draws > obs)), draws, obs


def control(lu, B=4000, seed=0):
    """The same resampling applied to an average, where no atom should appear."""
    n = len(lu)
    rng = np.random.default_rng(seed)
    obs = float(lu.mean())
    draws = np.array([lu[rng.integers(0, n, n)].mean() for _ in range(B)])
    return float(np.mean(draws == obs)), float(np.mean(draws > obs))


def fleets():
    print("=" * 78)
    print("the atom, on the fleets.  prediction: mass 1-(1-1/n)^n at the estimate,")
    print("and nothing above it, for the maximum; neither for the mean")
    print("%-10s %5s %10s %10s %10s %12s %10s"
          % ("fleet", "n", "1-(1-1/n)^n", "atom max", "above max", "atom mean",
             "above mean"))
    for nm, ld in (("FD002", M.FD002), ("FD004", M.FD004),
                   ("Severson", M.load_severson)):
        paths, Rs, lives, lu = prep(ld)
        lu = np.asarray(lu, float)
        n = len(lu)
        pred = 1.0 - (1.0 - 1.0 / n) ** n
        a, ab, _, _ = atom(lu)
        c, cb = control(lu)
        print("%-10s %5d %10.4f %10.4f %10.4f %12.4f %10.4f"
              % (nm, n, pred, a, ab, c, cb))


def mout(seed=0):
    print("=" * 78)
    print("the atom under an m-out-of-n bootstrap: it should fall like m/n")
    print("%-10s %6s %10s %10s %12s" % ("fleet", "m", "atom", "1-(1-1/n)^m", "width95"))
    for nm, ld in (("FD002", M.FD002), ("Severson", M.load_severson)):
        paths, Rs, lives, lu = prep(ld)
        lu = np.asarray(lu, float)
        n = len(lu)
        for m in (n, int(n ** 0.75), int(n ** 0.5), max(4, int(n ** 0.35))):
            a, _, draws, obs = atom(lu, m=m, seed=seed)
            # the interval for ell_0 built from the rescaled deviations
            w = float(np.percentile(draws, 97.5) - np.percentile(draws, 2.5))
            print("%-10s %6d %10.4f %10.4f %12.5f"
                  % (nm, m, a, 1.0 - (1.0 - 1.0 / n) ** m, w))


def coverage(reps=500, B=600, seed=0):
    """Where the endpoint is known: does either interval cover it, and does n help?

    ell_u ~ 1 - Beta(1,2), so 1-F(x) = (1-x)^2 near the endpoint 1: the tail index is
    alpha = 2 and the sample maximum approaches the endpoint at n^{-1/2}.  The m-out-of-n
    bootstrap estimates the law of a_m (M_n - M_m*) and transfers it with the factor
    (m/n)^{1/alpha}; at m = n that factor is one and the procedure is the naive percentile
    interval, atom and all.

    Inconsistency is not undercoverage at one sample size, it is undercoverage that does
    not go away, so n is swept.
    """
    ALPHA = 2.0
    rng = np.random.default_rng(seed)
    END = 1.0
    print("=" * 78)
    print("coverage of a nominal 95% interval for the endpoint, truth known")
    print("%8s %16s %10s %12s" % ("n", "bootstrap", "coverage", "mean width"))
    for n in (100, 400, 1600, 6400):
        for label, m in (("n out of n", n), ("m = n^0.7", max(3, int(n ** 0.7)))):
            cov, wid = 0, []
            for _ in range(reps):
                x = END - rng.beta(1.0, 2.0, n)
                obs = float(x.max())
                d = np.array([x[rng.integers(0, n, m)].max() for _ in range(B)])
                dev = (obs - d) * (m / n) ** (1.0 / ALPHA)
                lo = obs + float(np.percentile(dev, 2.5))
                hi = obs + float(np.percentile(dev, 97.5))
                cov += (lo <= END <= hi)
                wid.append(hi - lo)
            print("%8d %16s %10.3f %12.5f"
                  % (n, label, cov / reps, float(np.mean(wid))))


if __name__ == "__main__":
    fleets()
    mout()
    coverage()
