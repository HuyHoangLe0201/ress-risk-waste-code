"""Is the risk-waste plane a choice, or is it forced?

A referee can reasonably say: Lemma 1 is renewal-reward in new letters, Theorem 2
is Barlow-Proschan in new letters, and Propositions 2 and 5 are convexity.  Each
step is elementary and that charge has force.  The answer cannot be that the
theorems are hard.  It has to be that the coordinates are not chosen but forced,
and that is a provable statement rather than a rhetorical one.

  SUFFICIENCY (Lemma 1).  Cost depends on a rule only through (omega, P_f).

  MINIMALITY.  Conversely, if two rules cost the same at EVERY chi then their
  coordinates coincide.  Proof: (1+chi p)(1-w') = (1+chi p')(1-w) for all chi;
  both sides are affine in chi, so matching the constant term gives w = w' and
  matching the chi term gives p = p'.  So the map rule -> (omega, P_f) is exactly
  the quotient of the space of rules by cost-equivalence at all cost ratios.  The
  plane is not a convenient parameterisation, it is THE parameterisation.

  NO SCALAR SUFFICES.  Because the fibres of a scalar functional are
  codimension 1 while cost-equivalence classes are points, no single figure of
  merit can rank policies uniformly in chi.  Exhibited by construction below: two
  rules equal in cost at one chi and strictly ordered at another.

  IDENTIFICATION.  Two distinct cost ratios determine (omega, P_f) exactly:

      1 - omega = (chi1 - chi2) / (chi1 L2 - chi2 L1),
      P_f       = (L1 - L2)(1 - omega) / (chi1 - chi2).

  An operator who knows the cost rate under two price regimes recovers the whole
  operating point without ever measuring failures or wasted life.  That statement
  is invisible in the classical formulation and is checked numerically here.
"""
import sys

import numpy as np

import os.path as _p; sys.path.insert(0, _p.join(_p.dirname(_p.abspath(__file__)), "shared"))
import makefigs6 as M                                            # noqa: E402


def L(om, pf, chi):
    return (1.0 + chi * pf) / (1.0 - om)


def invert(L1, L2, c1, c2):
    """Recover (omega, P_f) from the cost at two distinct cost ratios."""
    one_minus_w = (c1 - c2) / (c1 * L2 - c2 * L1)
    w = 1.0 - one_minus_w
    p = (L1 - L2) * one_minus_w / (c1 - c2)
    return w, p


def real_policies(ld, seed=0):
    """Genuine (omega, P_f) points: threshold policies on a real fleet."""
    seq, life = ld()
    R, rm, pa, cal, val, Tb = M.prep(seq, life, seed=seed)
    lu = np.array([pa[i][R[i] > 0].min() for i in cal])
    g = np.linspace(np.quantile(lu, .05), lu.max() + .25 * Tb, 40)
    C = np.array([[M.unit_cost(i, e, R, rm, life) for i in cal] for e in g])
    return 1.0 - C[:, :, 1].mean(1) / Tb, C[:, :, 2].mean(1)


print("A. IDENTIFICATION from two price regimes, on real policies")
for nm, ld in (("FD002", M.FD002), ("FD004", M.FD004), ("Severson", M.load_severson)):
    om, pf = real_policies(ld)
    worst = 0.0
    for (c1, c2) in ((2.0, 9.0), (4.0, 30.0), (1.5, 100.0)):
        w, p = invert(L(om, pf, c1), L(om, pf, c2), c1, c2)
        worst = max(worst, float(np.max(np.abs(w - om))), float(np.max(np.abs(p - pf))))
    print("   %-9s %d policies, 3 regime pairs : max recovery error %.2e"
          % (nm, len(om), worst))

print()
print("B. MINIMALITY: equal cost at one chi does not mean equal policy")
om, pf = real_policies(M.FD002)
chi0 = 9.0
L0 = L(om, pf, chi0)
found = 0
for i in range(len(om)):
    for j in range(i + 1, len(om)):
        if abs(L0[i] - L0[j]) < 1e-3 and abs(om[i] - om[j]) > 1e-3:
            r = L(om, pf, 100.0)
            print("   two policies tie at chi=%g (cost %.4f) but differ at chi=100: "
                  "%.4f vs %.4f  [omega %.3f vs %.3f]"
                  % (chi0, L0[i], r[i], r[j], om[i], om[j]))
            found += 1
            break
    if found:
        break
if not found:
    print("   no near-tie on this grid; constructing one analytically instead")
    w1, p1 = 0.05, 0.02
    L1 = L(w1, p1, chi0)
    w2 = 0.12
    p2 = (L1 * (1 - w2) - 1) / chi0            # same cost at chi0 by construction
    print("   (%.3f,%.3f) and (%.3f,%.3f) both cost %.4f at chi=%g"
          % (w1, p1, w2, p2, L1, chi0))
    for c in (1.0, 50.0):
        print("      at chi=%5.1f : %.4f vs %.4f  -> order %s"
              % (c, L(w1, p1, c), L(w2, p2, c),
                 "flips" if (L(w1, p1, c) - L(w2, p2, c)) *
                            (L(w1, p1, 1.0) - L(w2, p2, 1.0)) < 0 else "holds"))

print()
print("C. how far apart can two policies be that tie at a given chi?")
for chi0 in (2.0, 9.0, 30.0):
    w1, p1 = 0.04, 0.03
    L1 = L(w1, p1, chi0)
    ws = np.linspace(0.04, 0.30, 200)
    ps = (L1 * (1 - ws) - 1) / chi0
    ok = ps >= 0
    print("   chi=%5.1f : the tie set is a segment with omega in [%.2f, %.2f], "
          "P_f in [%.3f, %.3f]" % (chi0, ws[ok].min(), ws[ok].max(),
                                   ps[ok].min(), ps[ok].max()))
