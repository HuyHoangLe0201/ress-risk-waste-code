"""Does the riskwaste package compute what the paper's scripts computed?

On the paper's own pipeline -- the ridge predictor and seed-0 split of makefigs6.prep --
the package is run on the same grids as the scripts that produced the reported numbers,
and the two are compared: the per-unit statistic ell_u, the frontier, kappa*, and the
share of monitored cycles a threshold rule can see.

One difference is expected and is printed rather than hidden. The scripts normalise the
calibration half's wasted life by the mean life of the whole fleet; the package, following
Lemma 1, normalises by the mean life of the units it averages over. The frontier then
differs by the factor Tbar_cal / Tbar_fleet in (1 - omega), and every quantity the paper
compares -- kappa*, the optimal threshold, ratios on a common set -- is unchanged by it.
The check converts one to the other and requires agreement to machine precision.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "shared"), os.path.dirname(HERE)]
import makefigs6 as M          # noqa: E402
import riskwaste as rw         # noqa: E402

FLEETS = [("FD002", M.FD002), ("FD004", M.FD004), ("Severson", M.load_severson)]

worst = 0.0
for name, ld in FLEETS:
    seq, life = ld()
    R, rm, pa, cal, val, Tb = M.prep(seq, life)
    lu = np.array([pa[i][R[i] > 0].min() for i in cal])
    grid = np.linspace(np.quantile(lu, .02), np.quantile(lu, .999) + .3 * Tb, 170)
    om_paper, pf_paper = M.frontier(ld)
    z0 = int(np.nonzero(pf_paper == 0)[0][0])
    ks_paper = M.critical_ratio(om_paper, pf_paper, z0) - 1.0

    pol = rw.ThresholdPolicy([pa[i] for i in cal])
    _, om, pf = pol.frontier(grid)
    Tb_cal = life[cal].mean()
    om_as_paper = 1.0 - (1.0 - om) * Tb_cal / Tb

    idx = np.concatenate([cal, val])
    visible = rw.ThresholdPolicy([pa[i] for i in idx]).visible_fraction()

    d = [np.abs(pol.ell_u - lu).max(), np.abs(om_as_paper - om_paper).max(),
         np.abs(pf - pf_paper).max(), abs(rw.kappa_star(om, pf) - ks_paper)]
    worst = max(worst, max(d))
    print("=" * 76)
    print("%-9s %d calibration units, %d-point grid" % (name, len(cal), len(grid)))
    print("   max |difference|  ell_u %.1e   omega %.1e   P_f %.1e   kappa* %.1e"
          % tuple(d))
    print("   kappa*  paper %.4f   package, same grid %.4f   package, exact %.4f"
          % (ks_paper, rw.kappa_star(om, pf), pol.kappa_star()))
    print("   omega_0 at the boundary: paper's normalisation %.4f, Lemma 1's %.4f "
          "(Tbar_cal / Tbar_fleet = %.4f)" % (om_paper[z0], om[z0], Tb_cal / Tb))
    print("   visible cycles %.1f%%" % (100 * visible))
print("=" * 76)
print("largest difference over all fleets: %.1e -> %s"
      % (worst, "AGREE" if worst < 1e-9 else "DISAGREE"))
