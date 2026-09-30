"""Does the two-regime structure survive competing failure modes?

The paper lists this as open.  It has an answer, and the answer is negative in a way that
bounds the paper's own headline.

Let T = min(T_deg, T_shock), with the degradation mode monitored and announced and the
shock mode unmonitored with support reaching zero.  A threshold on the degradation
predictor alarms at T_deg - l, so the unit fails exactly when T_shock <= T_deg - l.  For
the POPULATION failure probability to vanish we need l >= T_deg for every unit, that is an
alarm at time zero, so omega_0 = 1: there is no operating point with P_f = 0 at omega < 1.

By Proposition 5 that is precisely the condition for the optimal cost to be UNBOUNDED in
chi.  So a shock mode removes the ceiling that gives condition-based maintenance its
advantage at high cost ratios, the kink of Theorem 5 does not exist, and chi* is infinite.
The zero-failure regime is a property of fleets whose failures are announced, not of
monitoring as such.

On a finite fleet none of this is visible directly: P_f = 0 is reached at whatever
threshold happens to beat every OBSERVED shock, so the measured omega_0 is finite and
grows towards one only as n grows.  That is the same finite-fleet illusion the paper
already documents for the empirical age-replacement baseline, appearing here for the
condition-based family, and it is worth measuring.
"""
import numpy as np


def fleet(n, lam, rng, cv=0.35):
    """Degradation life lognormal with mean 1; shock exponential with rate lam."""
    sig = np.sqrt(np.log(1 + cv ** 2))
    Tdeg = rng.lognormal(-0.5 * sig ** 2, sig, n)
    Tsh = rng.exponential(1.0 / lam, n) if lam > 0 else np.full(n, np.inf)
    return Tdeg, Tsh


def coords(Tdeg, Tsh, ell):
    """Alarm at T_deg - ell; failure iff the shock beats it."""
    alarm = Tdeg - ell
    T = np.minimum(Tdeg, Tsh)
    fail = Tsh <= np.maximum(alarm, 0.0)
    waste = np.where(fail, 0.0, T - np.maximum(alarm, 0.0))
    return float(np.mean(waste)) / float(np.mean(T)), float(np.mean(fail))


def summarize(n, lam, rng, ngrid=400):
    Tdeg, Tsh = fleet(n, lam, rng)
    ells = np.linspace(0.0, Tdeg.max(), ngrid)
    om, pf = np.array([coords(Tdeg, Tsh, e) for e in ells]).T
    z = np.nonzero(pf <= 0)[0]
    if len(z) == 0:
        return None
    j0 = z[0]                       # smallest threshold with no observed failure
    om0 = om[j0]
    k = np.arange(j0)
    good = k[(pf[k] > 0) & (om[k] < om0)]
    if len(good) == 0:
        # nothing with failures is cheaper in waste than the zero-failure point, so that
        # point is optimal at every cost ratio: chi* = 0, not infinity
        return om0, 0.0
    return om0, float(np.max((om0 - om[good]) / (pf[good] * (1.0 - om0))))


def main():
    rng = np.random.default_rng(0)
    print("%8s %8s %12s %12s" % ("shock", "n", "omega_0", "chi*"))
    for lam in (0.0, 0.05, 0.2, 0.5):
        for n in (50, 200, 1000, 5000, 20000):
            reps = [summarize(n, lam, rng) for _ in range(5)]
            reps = [r for r in reps if r]
            o = float(np.median([r[0] for r in reps]))
            c = float(np.median([r[1] for r in reps]))
            print("%8.2f %8d %12.4f %12.2f" % (lam, n, o, c))
        print()


if __name__ == "__main__":
    main()
