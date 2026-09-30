"""Coverage of the intervals this paper reports, with Algorithm 1 implemented properly.

Run 1 split the resampled multiset rather than the distinct units, so a unit could
land on both sides of the split.  That is exactly the leakage Algorithm 1 step 3
forbids, so run 1 did not measure Algorithm 1.  This run implements the guard,
and keeps the leaky variant as a fourth arm so the guard's worth is measured
rather than asserted.

Population = the empirical distribution of the whole fleet.  Truth = the minimum
of the population cost curve.  Four 95% intervals, 300 sub-fleets each.
"""
import sys

import numpy as np

import os.path as _p; sys.path.insert(0, _p.join(_p.dirname(_p.abspath(__file__)), "shared"))
import makefigs6 as M                                            # noqa: E402

NGRID = 90
REPS = 300
BOOT = 400
ALPHA = 0.05
ARMS = ("unsplit + percentile", "unsplit + numerical delta",
        "split, leaky (no guard)", "split + guard (Algorithm 1)")


def population(ld):
    seq, life = ld()
    R, rm, pa, cal, val, Tb = M.prep(seq, life, seed=0)
    allu = np.concatenate([cal, val])
    lu = np.array([pa[i][R[i] > 0].min() for i in allu])
    g = np.linspace(np.quantile(lu, .05), np.quantile(lu, .999) + .28 * Tb, NGRID)
    C = np.array([[M.unit_cost(i, e, R, rm, life) for i in allu] for e in g])
    return C[:, :, 0], C[:, :, 1], Tb, len(allu)


def curve(num, den, Tb, idx):
    return num[:, idx].mean(-1) / den[:, idx].mean(-1) * Tb


def select_and_score(num, den, Tb, draw, rng, guard):
    """One replicate of the paper's protocol on a resampled multiset of units."""
    if guard:
        # split over DISTINCT units, copies follow their unit (Algorithm 1, step 3)
        uniq = np.unique(draw)
        rng.shuffle(uniq)
        half = set(uniq[: len(uniq) // 2].tolist())
        mask = np.array([u in half for u in draw])
        a, b = draw[mask], draw[~mask]
    else:
        p = rng.permutation(len(draw))
        a, b = draw[p[: len(draw) // 2]], draw[p[len(draw) // 2:]]
    if len(a) < 5 or len(b) < 5:
        return np.nan
    k = int(curve(num, den, Tb, a).argmin())
    return curve(num, den, Tb, b)[k]


def one_rep(num, den, Tb, N, nprime, rng):
    sub = rng.integers(0, N, size=nprime)
    theta = curve(num, den, Tb, sub)
    phi_u = theta.min()

    B = rng.integers(0, nprime, size=(BOOT, nprime))
    star = curve(num, den, Tb, sub[B]).T

    ci0 = tuple(np.quantile(star.min(1), [ALPHA / 2, 1 - ALPHA / 2]))

    eps = nprime ** -0.25
    Z = np.sqrt(nprime) * (star - theta)
    Dv = (np.min(theta + eps * Z, axis=1) - phi_u) / eps
    ci1 = (phi_u - np.quantile(Dv, 1 - ALPHA / 2) / np.sqrt(nprime),
           phi_u - np.quantile(Dv, ALPHA / 2) / np.sqrt(nprime))

    out = [(phi_u, ci0), (phi_u, ci1)]
    for guard in (False, True):
        h = np.array([select_and_score(num, den, Tb,
                                       sub[rng.integers(0, nprime, size=nprime)],
                                       rng, guard) for _ in range(BOOT)])
        h = h[np.isfinite(h)]
        out.append((float(np.median(h)),
                    tuple(np.quantile(h, [ALPHA / 2, 1 - ALPHA / 2]))))
    return out


def run(name, ld, nprime):
    num, den, Tb, N = population(ld)
    truth = float((num.mean(1) / den.mean(1) * Tb).min())
    rng = np.random.default_rng(11)
    cov = np.zeros(4)
    wid = np.zeros(4)
    est = np.zeros(4)
    for _ in range(REPS):
        for j, (e, ci) in enumerate(one_rep(num, den, Tb, N, nprime, rng)):
            cov[j] += (ci[0] <= truth <= ci[1])
            wid[j] += ci[1] - ci[0]
            est[j] += e
    cov, wid, est = cov / REPS, wid / REPS, est / REPS
    print("=" * 80)
    print("%-9s N=%d  sub-fleet n=%d  truth=%.4f  %d reps x %d resamples  nominal 95%%"
          % (name, N, nprime, truth, REPS, BOOT))
    for j, tag in enumerate(ARMS):
        print("   %-28s coverage %5.1f%%  width %5.1f%% of truth  "
              "point est %+6.2f%%"
              % (tag, 100 * cov[j], 100 * wid[j] / truth,
                 100 * (est[j] - truth) / truth))


if __name__ == "__main__":
    for nm, ld, npr in (("FD002", M.FD002, 100), ("FD004", M.FD004, 100),
                        ("Severson", M.load_severson, 80)):
        run(nm, ld, npr)
