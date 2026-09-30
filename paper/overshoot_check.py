"""Why the stop-loss reading of Theorem 4 fails, and what is true instead.

The identity W = ell - e would hold if the alarm fired at predicted RUL exactly
ell.  It does not.  The alarm fires at the first cycle whose running minimum has
already fallen at or below ell, so the effective threshold is not ell but
rm(t_alarm) <= ell, and

    (ell - e) - W  =  ell - rm(t_alarm)  =:  S,   the OVERSHOOT.

S is what breaks the stop-loss reading, so it is worth knowing how big it is and
what governs it.  The paper already reports, from an adversarial audit, that
predictor roughness is the variable governing the candidate rule's behaviour, but
gives no mechanism.  Overshoot is a candidate mechanism: a jumpier path steps
further past the threshold before triggering.  That is testable at the unit
level -- rougher units should overshoot more -- and the test needs a negative
control, so it is run against a smoothed copy of the same paths, where roughness
is reduced and nothing else is.
"""
import sys

import numpy as np

import os.path as _p; sys.path.insert(0, _p.join(_p.dirname(_p.abspath(__file__)), "shared"))
import makefigs6 as M                                            # noqa: E402


def roughness(p):
    """Mean absolute second difference, the paper's roughness measure."""
    return float(np.mean(np.abs(np.diff(p, 2)))) if len(p) > 2 else np.nan


def overshoot_stats(idx, R, rm, ells, Tb):
    """Per-unit mean overshoot across thresholds where the unit is replaced."""
    S, rows = {}, []
    for i in idx:
        vals = []
        for ell in ells:
            h = np.nonzero(rm[i] <= ell)[0]
            if len(h) == 0 or R[i][h[0]] <= 0:
                continue
            vals.append(ell - rm[i][h[0]])
        if vals:
            S[i] = float(np.mean(vals))
            rows.append((i, S[i]))
    return S, rows


def run(name, ld):
    seq, life = ld()
    R, rm, pa, cal, val, Tb = M.prep(seq, life, seed=0)
    idx = np.concatenate([cal, val])
    lu = np.array([pa[i][R[i] > 0].min() for i in idx])
    ells = np.linspace(np.quantile(lu, .05), lu.max() + .20 * Tb, 9)

    S, rows = overshoot_stats(idx, R, rm, ells, Tb)
    s = np.array([r[1] for r in rows])
    rough = np.array([roughness(pa[r[0]]) for r in rows])

    print("=" * 76)
    print("%s   Tbar=%.1f   %d units contribute" % (name, Tb, len(rows)))
    print("   overshoot S : median %.1f cycles (%.1f%% of Tbar), "
          "p90 %.1f, max %.1f" % (np.median(s), 100 * np.median(s) / Tb,
                                  np.quantile(s, .90), s.max()))
    ok = np.isfinite(rough) & np.isfinite(s)
    r_p = np.corrcoef(rough[ok], s[ok])[0, 1]
    order_r = np.argsort(rough[ok]).argsort()
    order_s = np.argsort(s[ok]).argsort()
    r_s = np.corrcoef(order_r, order_s)[0, 1]
    print("   corr(roughness, overshoot) : Pearson %+.2f   Spearman %+.2f  (n=%d)"
          % (r_p, r_s, ok.sum()))

    # negative control: smooth the same paths, changing roughness and nothing else
    for w in (5, 15):
        pas = {}
        for i in idx:
            p = pa[i]
            k = np.ones(w) / w
            pas[i] = np.convolve(p, k, mode="same") if len(p) > w else p.copy()
        rms = {i: np.minimum.accumulate(pas[i]) for i in idx}
        S2, rows2 = overshoot_stats(idx, R, rms, ells, Tb)
        s2 = np.array([r[1] for r in rows2])
        rough2 = np.array([roughness(pas[r[0]]) for r in rows2])
        print("   control, moving average w=%-2d : roughness %.2f -> %.2f, "
              "median overshoot %.1f -> %.1f cycles"
              % (w, np.nanmedian(rough), np.nanmedian(rough2),
                 np.median(s), np.median(s2)))


if __name__ == "__main__":
    for nm, ld in (("FD002", M.FD002), ("FD004", M.FD004),
                   ("Severson", M.load_severson)):
        run(nm, ld)
