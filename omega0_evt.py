"""Can the extreme-anchored quantity be repaired by an endpoint estimator?

Proposition 8 sorts the framework's quantities into averages, which are estimable at the
parametric rate, and maxima, which are not.  omega_0 is a maximum: it is anchored at
ell_0 = max_u ell_u, and a finite fleet's maximum sits below the population endpoint, so
omega_0-hat is biased downward and the ceiling with it.

The sample maximum is not the only estimator of an endpoint.  Where the upper tail has a
finite right endpoint -- a negative extreme-value index -- peaks-over-threshold gives one:
fit a generalised Pareto to the top k exceedances and read off u + sigma/(-xi).  If that
estimate is better than the maximum, omega_0's bias should shrink, and one of the two
quantities Proposition 8 calls unestimable becomes estimable after all.

If it does not help, that is worth knowing too, and for a specific reason: the endpoint
estimator is itself unstable at a hundred units, so this is a test of whether the
difficulty is the estimator or the data.
"""
import sys
import warnings

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")
import os.path as _p; sys.path.insert(0, _p.join(_p.dirname(_p.abspath(__file__)), "shared"))
import makefigs6 as M                                            # noqa: E402


def prep(ld):
    seq, life = ld()
    R, rm, pa, cal, val, _ = M.prep(seq, life, seed=0)
    idx = np.concatenate([cal, val])
    paths = [np.minimum.accumulate(np.asarray(pa[i], float)) for i in idx]
    Rs = [np.asarray(R[i], float) for i in idx]
    lives = np.array([float(len(p)) for p in paths])
    lu = np.array([float(paths[i][Rs[i] > 0].min()) for i in range(len(paths))])
    return paths, Rs, lives, lu


def omega_at(paths, Rs, lives, sub, ell):
    w = []
    for i in sub:
        h = np.nonzero(paths[i] <= ell)[0]
        w.append(float(Rs[i][h[0]]) if len(h) and Rs[i][h[0]] > 0 else 0.0)
    return float(np.mean(w)) / float(lives[sub].mean())


def gpd_endpoint(x, frac=0.30):
    """Peaks-over-threshold endpoint: u + sigma/(-xi), when xi < 0."""
    k = max(int(len(x) * frac), 8)
    if k >= len(x):
        return None
    u = np.sort(x)[-k - 1]
    exc = x[x > u] - u
    if len(exc) < 8:
        return None
    try:
        xi, loc, sigma = stats.genpareto.fit(exc, floc=0.0)
    except Exception:
        return None
    if xi >= -1e-3 or sigma <= 0:
        return None                      # no finite endpoint indicated
    return float(u + sigma / (-xi))


def run(name, ld, reps=300, seed=0):
    paths, Rs, lives, lu = prep(ld)
    N = len(paths)
    full = omega_at(paths, Rs, lives, np.arange(N), lu.max())
    rng = np.random.default_rng(seed)
    sizes = [s for s in (30, 60, 100, 160) if s <= N]

    print("=" * 76)
    print("%s   N=%d, full-fleet omega_0 = %.4f" % (name, N, full))
    print("%7s %12s %12s %12s %10s" %
          ("n", "max-based", "EVT-based", "bias max", "bias EVT"))
    for s in sizes:
        a, b, used = [], [], 0
        for _ in range(reps):
            sub = rng.choice(N, s, replace=False)
            m = lu[sub].max()
            a.append(omega_at(paths, Rs, lives, sub, m))
            e = gpd_endpoint(lu[sub])
            if e is not None and e > m:
                used += 1
                b.append(omega_at(paths, Rs, lives, sub, e))
            else:
                b.append(a[-1])
        a, b = float(np.mean(a)), float(np.mean(b))
        print("%7d %12.4f %12.4f %+12.4f %+10.4f   (EVT fired %d%%)"
              % (s, a, b, a - full, b - full, 100 * used // reps))


if __name__ == "__main__":
    for nm, ld in (("FD002", M.FD002), ("FD004", M.FD004),
                   ("Severson", M.load_severson)):
        run(nm, ld)
