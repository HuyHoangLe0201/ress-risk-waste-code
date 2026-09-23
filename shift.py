"""What happens to the quota when the fleet stops being exchangeable?

Every result from Theorem 11 onwards assumes the calibration units and the future
units are exchangeable.  Fleets are not: units are built to a revised standard, flown
on new routes, charged on a different policy.  Dependence the paper already handles;
this is the other failure, where calibration and deployment have different laws, and
it is the one that would void a quota.

The damage is bounded, and by something the operator can measure.  Write P for the
calibration law and Q for the deployment law, and

    Delta = sup_x |F_P(x) - F_Q(x)|

for the Kolmogorov distance between them.  The risk at index k is E[Qbar(ell_(k))]
with the threshold drawn from P, while k/(n+1) is E[Pbar(ell_(k))] for the same
threshold; the two integrands differ by at most Delta pointwise, so

    | risk - k/(n+1) |  <=  Delta.

Nothing about the shape of the shift enters, only its size, and Delta is estimable
from the two samples by a Kolmogorov-Smirnov statistic.

Checked three ways.  On the real fleets the shift is the paper's own pre-registered
covariate -- the sign of a unit's first principal score at cycle zero, fixed before
any result was seen -- so calibrating on one group and deploying on the other is a
genuine change of population rather than a simulated one.  A random split is the
negative control: it has no shift, so the deviation must collapse while the bound
stays loose, and a test that could not tell those two apart would prove nothing.
Sharpness is checked on a construction whose Delta is known exactly, where the
deviation should reach Delta(1 - k/(n+1)) and so approach the bound for small k.

Run: python shift.py
"""
import sys
import warnings

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")
import os.path as _p; sys.path.append(_p.join(_p.dirname(_p.abspath(__file__)), "shared"))
import makefigs6 as M                                            # noqa: E402

REPS = 4000


def prep_with_cov(ld, seed=0):
    """Unit thresholds and the paper's covariate split, aligned unit for unit."""
    seq, life = ld()
    R, rm, pa, cal, val, _ = M.prep(seq, life, seed=seed)
    idx = np.concatenate([cal, val])
    paths = [np.minimum.accumulate(np.asarray(pa[i], float)) for i in idx]
    Rs = [np.asarray(R[i], float) for i in idx]
    lu = np.array([float(paths[i][Rs[i] > 0].min()) for i in range(len(paths))])
    X = M.design(seq)
    Z = np.vstack([np.asarray(X[i], float)[0] for i in idx])
    Z = Z - Z.mean(0)
    _, _, vt = np.linalg.svd(Z, full_matrices=False)
    sc = Z @ vt[0]
    return lu, (sc > np.median(sc)).astype(int)


def risk_under(A, B, k, n, reps=REPS, seed=0):
    """Expected risk of index k calibrated on A and deployed on B."""
    rng = np.random.default_rng(seed)
    B = np.asarray(B, float)
    tot = 0.0
    for _ in range(reps):
        s = rng.choice(A, size=n, replace=False)
        e = np.sort(s)[::-1][k - 1]
        tot += float(np.mean(B > e))
    return tot / reps


def report(name, A, B, label, ks):
    n = min(len(A), len(B)) // 2
    print("   %-26s n=%-4d  Delta(KS)=%.4f" % (label, n, ks))
    print("      k   nominal    realised   deviation   bound    holds?  slack")
    for k in (1, 2, 3, 5):
        nom = k / float(n + 1)
        got = risk_under(A, B, k, n)
        dev = got - nom
        print("      %-3d %-10.5f %-10.5f %+10.5f  %-8.4f %-7s %.4f"
              % (k, nom, got, dev, ks,
                 "yes" if abs(dev) <= ks + 1e-12 else "NO", ks - abs(dev)))


def sharpness():
    """A shift of known size: does the deviation reach Delta(1 - k/(n+1))?"""
    print()
    print("sharpness: P uniform, Q = P with mass Delta moved above every threshold")
    print("   Delta   k    n     predicted Delta(1-k/(n+1))   measured   ratio")
    rng = np.random.default_rng(1)
    n = 200
    for D in (0.02, 0.10):
        for k in (1, 5, 20):
            tot = 0.0
            for _ in range(6000):
                s = rng.random(n)
                e = np.sort(s)[::-1][k - 1]
                # Q: with probability D the unit sits above everything
                tot += D + (1 - D) * (1.0 - e)
            got = tot / 6000 - k / float(n + 1)
            pred = D * (1 - k / float(n + 1))
            print("   %-7.2f %-4d %-5d %-27.5f %-10.5f %.4f"
                  % (D, k, n, pred, got, got / pred))


def main():
    for name, ld in (("FD002", M.FD002), ("FD004", M.FD004),
                     ("Severson", M.load_severson)):
        lu, grp = prep_with_cov(ld)
        A, B = lu[grp == 0], lu[grp == 1]
        ks = float(stats.ks_2samp(A, B).statistic)
        print("=" * 88)
        print("%s: %d units, covariate groups of %d and %d"
              % (name, len(lu), len(A), len(B)))
        report(name, A, B, "covariate shift A->B", ks)

        rng = np.random.default_rng(3)
        q = rng.permutation(len(lu))
        C, D = lu[q[:len(A)]], lu[q[len(A):]]
        ks0 = float(stats.ks_2samp(C, D).statistic)
        report(name, C, D, "random split (control)", ks0)

    sharpness()


if __name__ == "__main__":
    main()
