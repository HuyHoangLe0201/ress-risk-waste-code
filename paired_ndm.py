r"""The paired estimand: a ratio of two grid-selected minima, corrected properly.

Appendix "What the Non-Differentiability Costs" repairs the single-policy minimum with
the numerical delta method, and the limitations section names the paired case as the
extension it would attempt first: the reported quantity is a RATIO of two minima, one
over the threshold grid and one over the age grid, "whose directional derivative
involves the joint behaviour of two argmin sets and does not reduce to (eq:ndm)".
Until it exists the paired intervals are read as a floor.

Two things dissolve that.  First the derivative does exist in closed form, by the chain
rule: min is Hadamard directionally differentiable with derivative min over the argmin
set, and division is smooth away from zero, so with A = min_k theta_c[k],
B = min_j theta_a[j] and R = A/B,

    psi'(h_c, h_a) = ( min_{k in K_0} h_c[k]  -  R * min_{j in J_0} h_a[j] ) / B,

which does involve both argmin sets, exactly as the paper says -- but is no harder for
being a difference of two of them.  Second, and this is the part that makes the whole
worry evaporate, (eq:ndm) is a PROCEDURE and not a formula: it estimates whatever
directional derivative the functional has by a vanishing step, so it applies verbatim
with phi replaced by psi and the bootstrap deviations taken jointly over the same
resampled units.  The step condition eps -> 0 with sqrt(n) eps -> infinity is a
statement about the step alone and does not involve chi.

Both curves here are ratios of unit-level means, so one resample of the units moves
them together and the pairing is preserved.  The mean lifetime cancels from the ratio,
which removes one estimated quantity from the comparison.

Four things are measured.  The width the correction costs against the percentile floor,
which is the number the paper says is missing.  Agreement between the numerical
derivative and the closed form above, which checks the two derivations against each
other.  Sensitivity of the interval to the step across the swept cost ratios, which is
the uniformity the limitations section asks for.  And a negative control: the same
machinery on a functional that IS fully differentiable -- the ratio at two FIXED grid
points -- where the correction must reproduce the percentile interval, since a linear
derivative makes the two coincide.

Run: python paired_ndm.py
"""
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
import os.path as _p; sys.path.append(_p.join(_p.dirname(_p.abspath(__file__)), "shared"))
import makefigs6 as M                                            # noqa: E402

NBOOT = 3000
NGRID = 120
ALPHA = 0.05


def pieces(ld, seed=0):
    """Per-unit cycle length and failure flag on the threshold grid, plus lifetimes."""
    seq, life = ld()
    R, rm, pa, cal, val, Tb = M.prep(seq, life, seed=seed)
    lu_cal = np.array([pa[i][R[i] > 0].min() for i in cal])
    g = np.linspace(np.quantile(lu_cal, .05),
                    np.quantile(lu_cal, .999) + .28 * Tb, NGRID)
    C = np.array([[M.unit_cost(i, e, R, rm, life) for i in val] for e in g])
    cyc, fail = C[:, :, 1], C[:, :, 2]
    lives = np.array([float(len(pa[i])) for i in val])
    ages = np.linspace(lives.min() * 0.35, lives.max() * 1.02, NGRID)
    acyc = np.minimum(lives[None, :], ages[:, None])
    afail = (lives[None, :] <= ages[:, None]).astype(float)
    return cyc, fail, acyc, afail, len(val)


def curves(cyc, fail, acyc, afail, chi, idx):
    """The two cost curves on one resample of the units."""
    c = (1.0 + chi * fail[:, idx].mean(1)) / cyc[:, idx].mean(1)
    a = (1.0 + chi * afail[:, idx].mean(1)) / acyc[:, idx].mean(1)
    return np.concatenate([c, a])


def psi(theta, K=NGRID):
    return theta[:K].min() / theta[K:].min()


def closed_form(theta, h, K=NGRID, tol=1e-12):
    """psi'(h) from the chain rule, using the argmin sets of theta."""
    c, a = theta[:K], theta[K:]
    A, B = c.min(), a.min()
    K0 = np.nonzero(c <= A + tol * max(1.0, abs(A)))[0]
    J0 = np.nonzero(a <= B + tol * max(1.0, abs(B)))[0]
    return (h[:K][K0].min() - (A / B) * h[K:][J0].min()) / B


def intervals(theta, star, n, fn, eps):
    naive = np.array([fn(s) for s in star])
    lo_n, hi_n = np.quantile(naive, [ALPHA / 2, 1 - ALPHA / 2])
    p = fn(theta)
    Z = np.sqrt(n) * (star - theta)
    D = np.array([(fn(theta + eps * z) - p) / eps for z in Z])
    return ((lo_n, hi_n),
            (p - np.quantile(D, 1 - ALPHA / 2) / np.sqrt(n),
             p - np.quantile(D, ALPHA / 2) / np.sqrt(n)),
            p, D, Z)


def run(name, ld, chis=(1.0, 3.0, 9.0, 30.0), seed=0):
    cyc, fail, acyc, afail, n = pieces(ld, seed)
    rng = np.random.default_rng(7)
    B = rng.integers(0, n, size=(NBOOT, n))
    print("=" * 98)
    print("%s  n=%d  %d resamples, %d+%d grid points" % (name, n, NBOOT, NGRID, NGRID))
    print("   chi    ratio   percentile width  ndm width  ndm/pct  "
          "closed form vs numerical   ties(K0,J0)")
    for chi in chis:
        theta = curves(cyc, fail, acyc, afail, chi, np.arange(n))
        star = np.array([curves(cyc, fail, acyc, afail, chi, b) for b in B])
        eps = n ** -0.25
        (pc, nd, p, D, Z) = intervals(theta, star, n, psi, eps)
        Dc = np.array([closed_form(theta, z) for z in Z])
        r = float(np.corrcoef(D, Dc)[0, 1])
        rel = float(np.mean(np.abs(D - Dc)) / np.mean(np.abs(Dc)))
        k0 = int((theta[:NGRID] <= theta[:NGRID].min() * (1 + 1e-12)).sum())
        j0 = int((theta[NGRID:] <= theta[NGRID:].min() * (1 + 1e-12)).sum())
        print("   %-6.1f %-7.4f %-17.5f %-10.5f %-8.2f corr %.4f  rel %.4f    %d,%d"
              % (chi, p, pc[1] - pc[0], nd[1] - nd[0],
                 (nd[1] - nd[0]) / (pc[1] - pc[0]), r, rel, k0, j0))

    print("   step sensitivity, ndm width as a multiple of the percentile width")
    print("   chi     eps=n^-1/6  n^-1/5   n^-1/4   n^-1/3   n^-1/2")
    for chi in chis:
        theta = curves(cyc, fail, acyc, afail, chi, np.arange(n))
        star = np.array([curves(cyc, fail, acyc, afail, chi, b) for b in B])
        row = []
        for e in (n ** -(1 / 6.), n ** -0.2, n ** -0.25, n ** -(1 / 3.), n ** -0.5):
            (pc, nd, _, _, _) = intervals(theta, star, n, psi, e)
            row.append((nd[1] - nd[0]) / (pc[1] - pc[0]))
        print("   %-7.1f " % chi + "  ".join("%8.2f" % v for v in row))

    print("   control: ratio at two FIXED grid points, where the map is differentiable")
    theta = curves(cyc, fail, acyc, afail, 9.0, np.arange(n))
    star = np.array([curves(cyc, fail, acyc, afail, 9.0, b) for b in B])
    kc, ka = int(theta[:NGRID].argmin()), int(theta[NGRID:].argmin())
    fixed = lambda t: t[kc] / t[NGRID + ka]                       # noqa: E731
    (pc, nd, _, _, _) = intervals(theta, star, n, fixed, n ** -0.25)
    print("      percentile %.5f   ndm %.5f   ratio %.3f  %s"
          % (pc[1] - pc[0], nd[1] - nd[0], (nd[1] - nd[0]) / (pc[1] - pc[0]),
             "control passes" if abs((nd[1] - nd[0]) / (pc[1] - pc[0]) - 1) < 0.12
             else "CONTROL FAILS"))


if __name__ == "__main__":
    for nm, ld in (("FD002", M.FD002), ("FD004", M.FD004)):
        run(nm, ld)
