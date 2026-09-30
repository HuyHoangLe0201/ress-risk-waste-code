"""Age replacement, the schedule a condition-based rule has to beat.

Age replacement renews a unit at age ``t_R`` or at failure, whichever comes first, and
ignores the condition signal. It is the fair baseline for a condition-based rule: the
comparison then prices what watching each unit degrade is worth.

Its coordinates are exact on any fleet: ``P_f = Pr(T <= t_R)`` and ``omega`` is the
stop-loss transform of the lifetime law at ``t_R`` over the mean life (Theorem 2). Its
optimal age needs a lifetime law. On a finite fleet, fit one parametrically: the empirical
optimum degenerates to an order statistic of the observed lifetimes.
"""
import numpy as np
from scipy import stats

from .coordinates import coordinates, cost_rate

FAMILIES = {
    "lognormal": stats.lognorm,
    "weibull": stats.weibull_min,
    "gamma": stats.gamma,
}


def age_coordinates(lifetimes, ages):
    """``(omega, P_f)`` of age replacement at each age in ``ages``, on the given fleet."""
    T = np.asarray(lifetimes, float)
    ages = np.atleast_1d(np.asarray(ages, float))
    om = np.empty(len(ages))
    pf = np.empty(len(ages))
    for j, t in enumerate(ages):
        om[j], pf[j] = coordinates(T, np.full_like(T, t))
    return om, pf


def optimal_age(lifetimes, kappa, family="lognormal", n_grid=5000):
    """Cost-optimal replacement age at cost ratio ``kappa`` under a fitted lifetime law.

    The law is fitted by maximum likelihood with its location fixed at zero, and the
    renewal cost rate ``(1 + kappa F(t)) / int_0^t S`` is minimised on a grid out to four
    times the longest observed life. Returns ``np.inf`` when running to failure is
    cheapest, as it is under a decreasing failure rate.

    ``family="empirical"`` minimises over the observed lifetimes instead; it is provided
    for comparison and is not recommended as a baseline.
    """
    T = np.asarray(lifetimes, float)
    if family == "empirical":
        ages = np.unique(T)
        om, pf = age_coordinates(T, ages)
        L = cost_rate(om, pf, kappa)
        j = int(np.argmin(L))
        return float(ages[j]) if L[j] < 1.0 + kappa else float("inf")
    if family not in FAMILIES:
        raise ValueError("family must be one of %s or 'empirical'"
                         % ", ".join(sorted(FAMILIES)))
    law = FAMILIES[family]
    params = law.fit(T, floc=0)
    # Geometric spacing near zero, where a decreasing-hazard survivor drops steeply and
    # the trapezoid rule would overstate int S -- enough to make the end of the grid
    # look cheaper than running to failure.
    top = T.max()
    t = np.unique(np.concatenate([np.geomspace(top * 1e-6, top, n_grid // 2),
                                  np.linspace(top, 4.0 * top, n_grid // 2)]))
    S = law.sf(t, *params)
    cum = np.concatenate([[t[0] * (1.0 + S[0]) / 2.0],
                          (S[1:] + S[:-1]) / 2.0 * np.diff(t)]).cumsum()
    rate = (1.0 + kappa * (1.0 - S)) / cum
    j = int(np.argmin(rate))
    if j == len(t) - 1:            # still falling where the grid ends
        return float("inf")
    run_to_failure = (1.0 + kappa) / law.mean(*params)
    return float(t[j]) if rate[j] < run_to_failure else float("inf")
