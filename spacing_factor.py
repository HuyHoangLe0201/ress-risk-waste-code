r"""What does Equation (12) assume when it replaces a secant by the top spacing?

Theorem 5 gives an exact lower bound: evaluating the secant at the last point of the
failure branch, where exactly one calibration unit fails,

    kappa* >= (omega_0 - omega(ell_(n-1))) / ((1/n)(1 - omega_0)).

Equation (12) then writes that numerator as D_n/Tbar, with D_n the spacing between the two
largest ell_u, and obtains n D_n / (Tbar (1 - omega_0)). The two are not the same quantity.
D_n is a gap in PREDICTED LIFE; the numerator is the WASTE recovered by lowering the
threshold across that gap. They agree only when the running minimum descends at about one
unit of predicted life per cycle near the alarm, so that a drop of D_n in the threshold
delays the alarm by about D_n cycles.

That assumption is measurable, and this measures it. For each split, report

    factor = (omega_0 - omega(ell_(n-1))) * Tbar / D_n,

which is 1 exactly when the substitution is exact.
"""
import sys

import numpy as np

import os.path as _p; sys.path.insert(0, _p.join(_p.dirname(_p.abspath(__file__)), "shared"))
import makefigs6 as M                                            # noqa: E402

SPLITS = 40


def run(name, loader):
    seq, life = loader()
    facs, exact, approx = [], [], []
    for seed in range(SPLITS):
        R, rm, pa, cal, val, Tb = M.prep(seq, life, seed=seed)
        lu = np.sort(np.array([pa[i][R[i] > 0].min() for i in cal]))
        if len(lu) < 3:
            continue
        l0, l1 = lu[-1], lu[-2]
        Dn = l0 - l1
        if Dn <= 0:
            continue

        def om(e):
            per = np.array([M.unit_cost(i, e, R, rm, life) for i in cal])
            return 1 - per[:, 1].mean() / Tb

        w0, w1 = om(l0), om(l1)
        num = w0 - w1
        n = len(cal)
        e = num / ((1.0 / n) * (1 - w0))            # the exact secant at P_f = 1/n
        a = n * Dn / (Tb * (1 - w0))                # what Equation (12) writes instead
        facs.append(num * Tb / Dn)
        exact.append(e)
        approx.append(a)

    f = np.array(facs)
    r = np.array(approx) / np.array(exact)
    print("\n%s  (%d usable splits)" % (name, len(f)))
    print("   factor (omega_0-omega)*Tbar / D_n : median %.3f  [%.3f, %.3f]"
          % (np.median(f), *np.percentile(f, [10, 90])))
    print("   Eq.(12) / exact secant            : median %.3f  [%.3f, %.3f]"
          % (np.median(r), *np.percentile(r, [10, 90])))


if __name__ == "__main__":
    run("FD002", M.FD002)
    run("FD004", M.FD004)
    run("Severson", M.load_severson)
