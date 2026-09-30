r"""The vertex count should be a function of one dimensionless number, not of fleet size.

The balance in hullscale.py gives the number of admissible policies as N ~ (c^2 n / v)^{1/3}
in the curvature-dominated regime and H_n in the flat one.  The dimensionless strength of
the curvature against the noise, measured as in hullreal.py by the sagitta over its own
resampling error, is Lambda ~ c sqrt(n / v).  Eliminating c between the two,

    N - H_n  =  kappa * Lambda^{2/3},

with the fleet size entering ONLY through Lambda.  That is a much stronger claim than the
exponent: it says one curve carries every fleet and every fleet size, and it is
parameter-free once kappa is calibrated.

Both quantities are computed here exactly as hullreal.py computes them on the fleets --
the same hull routine, the same sagitta, the same unit-level bootstrap -- so kappa
transfers.  The simulation's `units` are its increments: resampling them with replacement,
holding the population frontier fixed, is the simulation's counterpart of resampling the
fleet.

A fitted exponent away from 2/3, or fleets off the simulated curve, refutes the balance.
"""
import numpy as np

from hullscale import lower_hull, harmonic


def path(n, c, incr, v=1.0):
    """Frontier of curvature c with a prescribed set of unit increments."""
    p = np.arange(n + 1) / n
    bar = 0.5 * c * (1.0 - p) ** 2
    step = np.concatenate([[0.0], np.sqrt(v) / n * incr])
    return p, bar + np.cumsum(step)


def sagitta(p, w):
    chord = w[0] + (w[-1] - w[0]) * (p - p[0]) / (p[-1] - p[0])
    return float(np.max(chord - w))


def measure(n, c, rng, boots=200, v=1.0):
    """(faces, Lambda) for one simulated fleet, by the fleet recipe."""
    incr = rng.standard_normal(n)
    p, w = path(n, c, incr, v)
    N = len(lower_hull(p, w)) - 1
    s0 = sagitta(p, w)
    bs = []
    for _ in range(boots):
        b = rng.choice(incr, n, replace=True)          # resample the units
        bs.append(sagitta(*path(n, c, b, v)))
    se = float(np.std(bs, ddof=1))
    return N, (s0 / se if se > 0 else np.inf)


def main(reps=16, boots=120):
    """The collapse holds only in the continuum, and both edges have to be excluded.

    Below, a cell is used only if it is out of the flat regime (the curvature has to have
    produced vertices at all) and out of lattice saturation.  The second is not a
    convenience: the balance predicts a vertex spacing h ~ (v/(c^2 n))^{1/3}, and once h
    falls below the lattice spacing 1/n there is nowhere left to put a vertex, so the
    count saturates at n and no scaling law can hold.  At n=60, c=64 that ratio is
    n*h = 0.96 and 44.5 of the 60 points are vertices; such cells say nothing about the
    exponent, and keeping them was what bent the first fit to 0.83.
    """
    print("=" * 78)
    print("collapse: N - H_n against Lambda, over curvatures and fleet sizes")
    print("%8s %8s %9s %9s %11s %8s %7s" %
          ("n", "c", "faces", "H_n", "Lambda", "N/n", "used"))
    L, Y = [], []
    rng = np.random.default_rng(0)
    for n in (260, 600, 1500, 4000, 10000):
        for c in (0.25, 1.0, 4.0, 16.0):
            fs, ls = [], []
            for _ in range(reps):
                f, lam = measure(n, c, rng, boots=boots)
                if np.isfinite(lam):
                    fs.append(f); ls.append(lam)
            f, lam = float(np.median(fs)), float(np.median(ls))
            h = harmonic(n)
            use = (f - h > 2.0) and (f < n / 8.0)
            print("%8d %8.2f %9.1f %9.2f %11.2f %8.3f %7s"
                  % (n, c, f, h, lam, f / n, "yes" if use else "no"))
            if use:
                L.append(lam); Y.append(f - h)
    L, Y = np.array(L), np.array(Y)
    sl, ic = np.polyfit(np.log(L), np.log(Y), 1)
    pred = np.polyval([sl, ic], np.log(L))
    r2 = 1.0 - np.sum((np.log(Y) - pred) ** 2) / np.sum(
        (np.log(Y) - np.mean(np.log(Y))) ** 2)
    kappa = float(np.exp(ic))
    print()
    print("   fit over %d continuum cells, n=260..10000, c=0.25..16" % len(L))
    print("   N - H_n = %.2f * Lambda^%.3f      R2 %.3f   (theory: exponent 2/3)"
          % (kappa, sl, r2))
    return kappa, sl


def asymptotic(reps=16, boots=120):
    """The same fit, restricted to cells large enough for the continuum to hold.

    (N - H_n) / Lambda^{2/3} drifts upward with n -- 2.4 at n=260, 3.0 by n=10000 -- so a
    fit that pools fleet-sized cells with large ones reports an exponent that is neither
    the small-n behaviour nor the limit.
    """
    print("=" * 78)
    print("the same collapse, restricted to n >= 1500 and Lambda >= 20")
    L, Y = [], []
    rng = np.random.default_rng(3)
    for n in (1500, 4000, 10000, 25000):
        for c in (1.0, 4.0, 16.0):
            fs, ls = [], []
            for _ in range(reps):
                f, lam = measure(n, c, rng, boots=boots)
                if np.isfinite(lam):
                    fs.append(f); ls.append(lam)
            f, lam = float(np.median(fs)), float(np.median(ls))
            h = harmonic(n)
            if lam >= 20.0 and f < n / 8.0:
                L.append(lam); Y.append(f - h)
                print("   n=%6d c=%5.1f  N-H_n %8.2f  Lambda %8.1f   ratio %.2f"
                      % (n, c, f - h, lam, (f - h) / lam ** (2.0 / 3.0)))
    L, Y = np.array(L), np.array(Y)
    sl, ic = np.polyfit(np.log(L), np.log(Y), 1)
    pred = np.polyval([sl, ic], np.log(L))
    r2 = 1.0 - np.sum((np.log(Y) - pred) ** 2) / np.sum(
        (np.log(Y) - np.mean(np.log(Y))) ** 2)
    print("   N - H_n = %.2f * Lambda^%.3f      R2 %.3f   (theory: 2/3 = 0.667)"
          % (np.exp(ic), sl, r2))
    return float(np.exp(ic)), float(sl)


def fleet_scale_bias(reps=48):
    """What exponent does a fleet-sized sweep report when the truth is exactly 1/3?

    The fleets support n = 135 to 260, and Section 4 reads an exponent off sub-fleets of
    40 to 260.  Running that same sweep on a simulation whose asymptotic exponent is known
    to be 1/3 says how much of the reported number is the law and how much is the window.
    """
    print("=" * 78)
    print("what a fleet-sized sweep reports when the true exponent is 1/3")
    sizes = np.array([40.0, 70.0, 110.0, 170.0, 260.0])
    rng = np.random.default_rng(4)
    print("%10s %s %9s %8s" % ("curvature", "".join("%8d" % s for s in sizes), "exponent", "R2"))
    for c in (0.0, 0.25, 1.0, 4.0, 16.0):
        med = []
        for s in sizes:
            v = [measure(int(s), c, rng, boots=2)[0] for _ in range(reps)]
            med.append(float(np.median(v)))
        med = np.array(med)
        sl, ic = np.polyfit(np.log(sizes), np.log(med), 1)
        pr = np.polyval([sl, ic], np.log(sizes))
        r2 = 1.0 - np.sum((np.log(med) - pr) ** 2) / np.sum(
            (np.log(med) - np.log(med).mean()) ** 2)
        print("%10.2f %s %9.3f %8.4f"
              % (c, "".join("%8.1f" % m for m in med), sl, r2))
    print("   the asymptotic exponent is 1/3 for every c > 0 and 0 (log) for c = 0;")
    print("   over this window neither is what comes out.")


if __name__ == "__main__":
    kappa, sl = main()
    asymptotic()
    fleet_scale_bias()

    print("=" * 78)
    print("the fleets, against that curve")
    import hullreal as HR
    import makefigs6 as M
    print("%-10s %8s %10s %10s %12s %10s" %
          ("fleet", "N", "Lambda", "faces", "predicted", "H_N"))
    for nm, ld in (("FD002", M.FD002), ("FD004", M.FD004),
                   ("Severson", M.load_severson)):
        tab, lives = HR.prep(ld)
        n = len(tab[0])
        rng = np.random.default_rng(0)
        pf, om = HR.pareto(tab, lives, np.arange(n))
        f = HR.vertices(pf, om)
        s0 = HR.sagitta(pf, om)
        bs = []
        for _ in range(200):
            sub = rng.choice(n, n, replace=True)
            p2, o2 = HR.pareto(tab, lives, sub)
            if p2 is not None:
                bs.append(HR.sagitta(p2, o2))
        lam = s0 / float(np.std(bs, ddof=1))
        h = harmonic(n)
        print("%-10s %8d %10.1f %10d %12.1f %10.2f"
              % (nm, n, lam, f, h + kappa * lam ** sl, h))
