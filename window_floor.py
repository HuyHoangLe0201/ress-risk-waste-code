"""A theorem for the one policy family that lacks one.

Sections 3.2 and 3.3 give coordinates for age replacement and for threshold
policies.  Commit-and-forget -- observe k cycles, predict the total life, commit
to a replacement time, stop monitoring -- has none, although the paper defines it
and measures it.  One structural fact is immediate and has consequences.

  (i) To observe k cycles a unit must survive k cycles.  Units with T <= k fail
      before any commitment is made, whatever the predictor, so

          P_f(k)  >=  F(k),

      a failure floor set by the window alone.  No prediction quality removes it.

  (ii) By Proposition 5 a family whose achievable P_f is bounded below by a
       positive number has an UNBOUNDED cost in chi.  So for any window bounded
       away from zero the commit-and-forget cost grows linearly in chi while the
       continuous-monitoring cost saturates, and g_CBM/g_CF -> 0.

  (iii) Because the floor enters the cost multiplied by chi, a larger failure cost
        should push the optimal window down: k*(chi) non-increasing, and k* -> 0
        as chi -> infinity, at which point the family degenerates to age
        replacement.  That is the testable part, and it is tested here.
"""
import sys

import numpy as np

import os.path as _p; sys.path.insert(0, _p.join(_p.dirname(_p.abspath(__file__)), "shared"))
import makefigs6 as M                                            # noqa: E402

CHIS = np.array([1.0, 2.0, 4.0, 9.0, 19.0, 49.0, 99.0, 299.0])


def family_B(seq, life, ks, margins, seed=0):
    """Cost of commit-and-forget on a validation half, for each (k, margin)."""
    n = len(seq)
    rg = np.random.default_rng(seed)
    p = rg.permutation(n)
    cal, val = p[: n // 2], p[n // 2:]
    Tb = life[cal].mean()
    out = {}
    for k in ks:
        pool = cal[life[cal] > k]
        if len(pool) < 8:
            continue
        Xc = np.array([M.agg(seq[i], k) for i in pool])
        w = M.fit(Xc, life[pool].astype(float))
        Xv = np.array([M.agg(seq[i], k) for i in val])
        That = Xv @ w
        Tv = life[val].astype(float)
        for m in margins:
            tau = np.maximum(That - m, k)
            died_in_window = Tv <= k
            fail = died_in_window | (Tv <= tau)
            cost = np.where(fail, M.CF, 1.0)
            cyc = np.where(fail, Tv, tau)
            out[(k, m)] = (float(cost.mean()), float(cyc.mean()),
                           float(fail.mean()), Tb)
    return out, Tb


def run(name, ld):
    seq, life = ld()
    life = np.asarray(life)
    Tb = life.mean()
    ks = np.unique(np.round(np.arange(0.08, 0.85, 0.04) * Tb).astype(int))
    margins = np.round(np.arange(0.0, 0.35, 0.03) * Tb).astype(int)

    tab, _ = family_B(seq, life, ks, margins)
    print("=" * 78)
    print("%-9s Tbar=%.0f   windows %d..%d   %d (k,m) pairs"
          % (name, Tb, ks.min(), ks.max(), len(tab)))
    print("   chi :   " + "".join("%7.0f" % c for c in CHIS))
    best_k, best_pf = [], []
    for chi in CHIS:
        bk, bc, bpf = None, np.inf, None
        for (k, m), (cm, cy, pf, _) in tab.items():
            # normalised cost with c_p = 1 and c_f = 1 + chi
            c = ((1.0 - pf) + (1.0 + chi) * pf) / cy * Tb
            if c < bc:
                bc, bk, bpf = c, k, pf
        best_k.append(bk / Tb)
        best_pf.append(bpf)
    print("   k*/Tb:  " + "".join("%7.2f" % v for v in best_k))
    print("   P_f  :  " + "".join("%7.3f" % v for v in best_pf))
    mono = all(best_k[i + 1] <= best_k[i] + 1e-9 for i in range(len(best_k) - 1))
    print("   k*(chi) non-increasing? %s     k* falls %.2f -> %.2f"
          % (mono, best_k[0], best_k[-1]))
    # the floor F(k) at the largest and smallest window
    for k in (ks.min(), ks.max()):
        print("      F(k=%3d) = %.3f  (failure floor at that window)"
              % (k, float((life <= k).mean())))


if __name__ == "__main__":
    for nm, ld in (("FD002", M.FD002), ("FD004", M.FD004),
                   ("Severson", M.load_severson)):
        run(nm, ld)
