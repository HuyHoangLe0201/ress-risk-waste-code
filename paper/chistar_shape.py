"""Does chi-star's relative uncertainty shrink with fleet size at all?

chi* is the k=1 secant, so it is driven by the gap between the two largest ell_u.  By the
Renyi representation the top spacings of order statistics are asymptotically exponential,
and an exponential has coefficient of variation exactly 1, mean over median 1/log 2 = 1.443
and skewness 2 -- none of which depend on the sample size.

If chi-star-hat inherits that shape, its RELATIVE uncertainty does not fall as the fleet
grows.  That is a much stronger statement than bias or drift: more units would move the
constant without ever making it precise, and no fleet size would fix it.  The measured
sd-over-mean values of about 1 in the previous experiment are what prompted this.

Measured here across fleet sizes: coefficient of variation, mean over median, and skewness
of the resampling distribution, against the exponential's 1, 1.443 and 2.  A CV that stays
near 1 while n grows four-fold is the claim; a CV that falls like n^{-1/2} would refute it.
"""
import sys

import numpy as np

import os.path as _p; sys.path.insert(0, _p.join(_p.dirname(_p.abspath(__file__)), "shared"))
import makefigs6 as M                                            # noqa: E402
from quota import prep                                           # noqa: E402
from chistar_tailindex import chistar_k1                         # noqa: E402


def run(name, ld, reps=1200, seed=0):
    paths, Rs, lives, lu = prep(ld)
    N = len(paths)
    rng = np.random.default_rng(seed)
    sizes = [s for s in (40, 70, 120, 200) if s <= N]

    print("=" * 76)
    print("%s   N=%d      (exponential: CV 1.000, mean/median 1.443, skew 2.00)"
          % (name, N))
    print("%7s %9s %9s %11s %9s" % ("n", "mean", "CV", "mean/med", "skew"))
    for s in sizes:
        v = np.array([chistar_k1(paths, Rs, lives, rng.choice(N, s, replace=False))
                      for _ in range(reps)], float)
        v = v[np.isfinite(v)]
        m, sd = float(v.mean()), float(v.std(ddof=1))
        sk = float(np.mean(((v - m) / sd) ** 3))
        print("%7d %9.3f %9.3f %11.3f %9.2f"
              % (s, m, sd / m, m / float(np.median(v)), sk))


if __name__ == "__main__":
    for nm, ld in (("FD002", M.FD002), ("FD004", M.FD004),
                   ("Severson", M.load_severson)):
        run(nm, ld)
