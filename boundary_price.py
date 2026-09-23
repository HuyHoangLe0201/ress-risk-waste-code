r"""What does the zero-failure policy actually cost out of sample?

The fleets select an operating point on the zero-failure boundary, and Theorem 6 says the
unit-level bootstrap cannot measure the error there.  This is the same fact from the
decision side, and it has an exact answer rather than a caution.

A unit fails under threshold e exactly when ell_u > e, so a policy has no failures ON THE
SAMPLE precisely when e >= ell_0 = max_u ell_u.  For a new unit drawn from the same
population, exchangeability of the n+1 values makes the new one the largest with
probability exactly 1/(n+1).  So

    the zero-failure policy fails out of sample at rate 1/(n+1), not 0,

with no distributional assumption beyond exchangeability -- no tail index, no smoothness,
nothing to estimate.  The unit-level bootstrap reports 0, and must: a resample is drawn from
units already seen, none of which exceeds the observed maximum, so the bootstrap standard
error of P_f at this policy is exactly zero.  The boundary vertex is thus the one point of
the achievable set that the paper's own machinery certifies as risk-free, and it is the one
point where that certificate is empty.

Measured here three ways.  First the prediction itself, by repeated splits: fit ell_0 on a
training half and count failures on the held-out half, against 1/(n_train+1).  Second the
bootstrap's verdict on the same policy, which should be identically zero.  Third the
consequence: the boundary vertex moves from b = 0 to b = 1/((n+1)(1-omega_0)) in dual
coordinates, which shifts the kink chi* and can remove the vertex from the envelope
altogether.
"""
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
import os.path as _p; sys.path.append(_p.join(_p.dirname(_p.abspath(__file__)), "shared"))
import makefigs6 as M                                            # noqa: E402
from quota import prep                                           # noqa: E402


def outofsample(lu, reps=4000, seed=0):
    """Split, take the max on one half, count exceedances on the other."""
    n = len(lu)
    rng = np.random.default_rng(seed)
    k = n // 2
    rate = []
    for _ in range(reps):
        p = rng.permutation(n)
        tr, te = lu[p[:k]], lu[p[k:]]
        rate.append(float(np.mean(te > tr.max())))
    return float(np.mean(rate)), 1.0 / (k + 1.0)


def boot_se_at_boundary(lu, B=2000, seed=0):
    """Bootstrap s.e. of P_f at the threshold ell_0, which cannot be anything but zero."""
    n = len(lu)
    rng = np.random.default_rng(seed)
    e = float(lu.max())
    pf = [float(np.mean(lu[rng.integers(0, n, n)] > e)) for _ in range(B)]
    return float(np.mean(pf)), float(np.std(pf, ddof=1))


def omega_at(paths, Rs, lives, e):
    """Waste share at threshold e, over the whole fleet."""
    Tb = float(np.mean(lives))
    w = 0.0
    for p, R in zip(paths, Rs):
        acc = np.minimum.accumulate(p)
        h = np.nonzero(acc <= e)[0]
        if len(h) and R[h[0]] > 0:
            w += float(R[h[0]])
    return w / (len(paths) * Tb)


def run(name, ld):
    paths, Rs, lives, lu = prep(ld)
    lu = np.asarray(lu, float)
    n = len(lu)
    obs, pred = outofsample(lu)
    bm, bs = boot_se_at_boundary(lu)
    om0 = omega_at(paths, Rs, lives, float(lu.max()))
    b_new = 1.0 / ((n + 1.0) * (1.0 - om0))
    print("=" * 78)
    print("%s   n=%d" % (name, n))
    print("   held-out failure rate of the zero-failure policy   %.5f" % obs)
    print("   exchangeability prediction 1/(n_train+1)           %.5f" % pred)
    print("   ratio                                              %.2f" % (obs / pred))
    print("   bootstrap mean P_f at that policy                  %.5f" % bm)
    print("   bootstrap s.e. of P_f at that policy               %.5f" % bs)
    print("   omega_0 = %.4f, so the vertex moves from b = 0 to b = %.5f"
          % (om0, b_new))
    return obs, pred


if __name__ == "__main__":
    R = []
    for nm, ld in (("FD002", M.FD002), ("FD004", M.FD004),
                   ("Severson", M.load_severson)):
        R.append(run(nm, ld))
    print("=" * 78)
    o = np.array([r[0] for r in R])
    p = np.array([r[1] for r in R])
    print("observed / predicted across fleets: %s   (mean %.2f)"
          % (" ".join("%.2f" % x for x in o / p), float(np.mean(o / p))))
