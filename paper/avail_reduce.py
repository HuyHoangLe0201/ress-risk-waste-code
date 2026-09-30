"""Availability is not a second objective.  It is the cost at one particular ratio.

From (eq:avail),  A = (1-w) / [(1-w) + d_p + (d_f-d_p) P_f], so

    1/A = 1 + [d_p + (d_f-d_p)P_f]/(1-w) = 1 + d_p (1 + chi_A P_f)/(1-w),
    chi_A = (d_f - d_p)/d_p.

Maximising availability is therefore IDENTICAL to minimising the Lemma 1 cost at cost
ratio chi_A -- not approximately, but up to the positive constant d_p.  Three consequences
follow, and the third turns an empirical observation of the paper into a theorem:

  (i)   the availability optimum is a vertex of the same lower envelope;
  (ii)  its optimal rule is the hazard threshold of Theorem 3 taken at chi_A;
  (iii) cost and availability optima coincide exactly when chi and chi_A fall in a common
        interval of Proposition 5's partition.

With the paper's d_p = 0.02 and d_f = 0.30, chi_A = 14.  Since chi* is about 1.6 on FD002,
every chi above chi* puts both ratios in the same terminal interval and the two objectives
must agree; below chi* they need not, and at chi = 0.3 they should oppose completely.  The
paper reports exactly that, so what is being checked here is whether the mechanism is the
stated one.
"""
import sys

import numpy as np

import os.path as _p; sys.path.insert(0, _p.join(_p.dirname(_p.abspath(__file__)), "shared"))
import makefigs6 as M                                            # noqa: E402

DP, DF = 0.02, 0.30
CHI_A = (DF - DP) / DP


def achievable(ld, n_thr=260):
    seq, life = ld()
    R, rm, pa, cal, val, _ = M.prep(seq, life, seed=0)
    idx = np.concatenate([cal, val])
    paths = [np.asarray(pa[i], float) for i in idx]
    n_u = np.array([len(p) for p in paths], float)
    Tb = float(n_u.mean())
    lo = min(p.min() for p in paths)
    hi = max(p.max() for p in paths)
    out = []
    for e in np.linspace(lo, hi, n_thr):
        w, f = [], []
        for p in paths:
            h = np.nonzero(p <= e)[0]
            if len(h) == 0:
                w.append(0.0); f.append(1.0)
            else:
                w.append(float(len(p) - 1 - h[0])); f.append(0.0)
        out.append((float(np.mean(w)) / Tb, float(np.mean(f))))
    return np.array(out)


def run(name, ld):
    A = achievable(ld)
    om, pf = A[:, 0], A[:, 1]
    avail = (1 - om) / ((1 - om) + DP + (DF - DP) * pf)
    ja = int(np.argmax(avail))
    jc = int(np.argmin((1 + CHI_A * pf) / (1 - om)))

    print("=" * 70)
    print("%s   chi_A = %.1f" % (name, CHI_A))
    print("   argmax availability = %d | argmin cost at chi_A = %d | identical: %s"
          % (ja, jc, ja == jc))
    # the paper compares availability against the DURATION-ADJUSTED cost, which is a
    # different linear-fractional functional from the Lemma 1 one; report both
    D = (1 - om) + DP + (DF - DP) * pf
    print("%8s %12s %12s %10s %8s %8s" %
          ("chi", "Lemma1 P_f", "dur.adj P_f", "avail P_f", "L1=A", "Ld=A"))
    for chi in (0.3, 1.0, 2.0, 3.0, 5.0, 9.0, 30.0):
        j1 = int(np.argmin((1 + chi * pf) / (1 - om)))
        jd = int(np.argmin((1 + chi * pf) / D))
        print("%8.1f %12.4f %12.4f %10.4f %8s %8s"
              % (chi, pf[j1], pf[jd], pf[ja],
                 "yes" if j1 == ja else "NO", "yes" if jd == ja else "NO"))


if __name__ == "__main__":
    for nm, ld in (("FD002", M.FD002), ("FD004", M.FD004),
                   ("Severson", M.load_severson)):
        run(nm, ld)
