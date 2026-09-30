"""Threshold policies on a predicted remaining life, and where their optimum lies.

A condition-based rule replaces a unit the first time its predicted remaining useful life
(RUL) falls to a threshold ``ell``. For any predictor, the rule's coordinates depend on a
predicted path only through its running minimum (Proposition 3), and its failure
probability is the upper tail of one per-unit statistic,

    ell_u = min { predicted RUL at the cycles where the true RUL is still positive },

so that ``P_f(ell) = Pr(ell_u > ell)`` (Theorem 4). The zero-failure threshold
``ell_0 = max_u ell_u`` is the exact cost optimum if and only if the cost ratio is at least

    kappa* = max_{ell < ell_0} (omega_0 - omega(ell)) / (P_f(ell) (1 - omega_0))

(Theorem 5), computable from calibration data before any policy is committed to.

Convention: time is counted in inspection cycles. ``predicted_rul[u][k]`` is the
prediction made at cycle ``k + 1`` of unit ``u``, and the path runs to the failure cycle,
so the unit's lifetime is ``len(predicted_rul[u])``. An alarm at the failure cycle comes
too late and counts as a failure.
"""
import numpy as np

from .coordinates import coordinates, cost_rate

_TIE = 1e-12


class ThresholdPolicy:
    """Replace a unit at the first cycle its predicted RUL reaches ``ell``.

    Parameters
    ----------
    predicted_rul : sequence of 1-D arrays
        One predicted-RUL path per unit, observed up to and including its failure cycle.
        Any predictor may produce them; nothing here depends on how.
    """

    def __init__(self, predicted_rul):
        paths = [np.asarray(p, float).ravel() for p in predicted_rul]
        if not paths:
            raise ValueError("no units")
        for u, p in enumerate(paths):
            if len(p) < 2:
                raise ValueError("unit %d: a path needs at least two cycles" % u)
            if not np.all(np.isfinite(p)):
                raise ValueError("unit %d: the predicted path has non-finite values" % u)
        self.paths = paths
        self.lifetimes = np.array([len(p) for p in paths], float)
        self._runmin = [np.minimum.accumulate(p) for p in paths]
        # the running minimum over every cycle but the failure cycle
        self.ell_u = np.array([m[-2] for m in self._runmin])

    @property
    def n_units(self):
        return len(self.paths)

    @property
    def ell0(self):
        """The zero-failure threshold: the smallest at which no unit fails."""
        return float(self.ell_u.max())

    def candidates(self, units=None):
        """Every threshold at which the cost-rate optimum can sit, at any cost ratio.

        Between consecutive values of ``ell_u`` the failure probability is constant and
        the wasted life only grows with ``ell``, so the optimum is always at one of them,
        or at ``-inf`` (never replace preventively). That is at most ``n + 1``
        thresholds, and searching them is exact where a grid is an approximation.
        """
        ell = self.ell_u if units is None else self.ell_u[np.asarray(units)]
        return np.concatenate([[-np.inf], np.unique(ell)])

    def alarm_times(self, ell):
        """Cycle at which each unit is replaced at threshold ``ell``; ``inf`` if never."""
        out = np.full(self.n_units, np.inf)
        for u, m in enumerate(self._runmin):
            k = np.searchsorted(-m, -ell, side="left")
            if k < len(m):
                out[u] = k + 1.0
        return out

    def coordinates(self, ell, units=None):
        """``(omega, P_f)`` at threshold ``ell``, over all units or the given indices.

        ``units`` may repeat an index, as a bootstrap resample does.
        """
        tau = self.alarm_times(ell)
        if units is None:
            return coordinates(self.lifetimes, tau)
        units = np.asarray(units)
        return coordinates(self.lifetimes[units], tau[units])

    def frontier(self, thresholds=None, units=None):
        """``(thresholds, omega, P_f)`` along increasing thresholds.

        With ``thresholds=None`` the exact candidate set is used; pass an array to
        evaluate on a grid of your own.
        """
        th = self.candidates(units) if thresholds is None else np.sort(
            np.asarray(thresholds, float))
        om = np.empty(len(th))
        pf = np.empty(len(th))
        for j, ell in enumerate(th):
            om[j], pf[j] = self.coordinates(ell, units)
        return th, om, pf

    def kappa_star(self, units=None):
        """``kappa*`` of Theorem 5 on this fleet's exact frontier."""
        _, om, pf = self.frontier(units=units)
        return kappa_star(om, pf)

    def optimum(self, kappa, units=None, thresholds=None):
        """Cost-rate minimiser at cost ratio ``kappa``: ``(ell, omega, P_f, L)``."""
        th, om, pf = self.frontier(thresholds, units)
        return optimum(th, om, pf, kappa)

    def record_mask(self, u):
        """Cycles of unit ``u`` a threshold rule can react to (Proposition 3).

        A cycle is visible when the prediction reaches the running minimum. Changing the
        prediction at any other cycle leaves ``omega`` and ``P_f`` unchanged at every
        threshold, and so leaves the cost unchanged at every cost ratio.
        """
        p = self.paths[u]
        return p <= self._runmin[u] + _TIE

    def visible_fraction(self):
        """Share of all monitored cycles that are visible to any threshold rule."""
        rec = sum(int(self.record_mask(u).sum()) for u in range(self.n_units))
        return rec / float(self.lifetimes.sum())


def kappa_star(omega, pf):
    """The critical cost ratio of Theorem 5, from a frontier ordered by threshold.

    The zero-failure boundary is the first point with ``P_f == 0``; the frontier must
    reach it. Every earlier point is on the failure branch.
    """
    omega = np.asarray(omega, float)
    pf = np.asarray(pf, float)
    zero = np.nonzero(pf == 0)[0]
    if len(zero) == 0:
        raise ValueError("the frontier never reaches P_f = 0; extend it past ell_0")
    z0 = int(zero[0])
    if z0 == 0:
        return 0.0
    om0 = omega[z0]
    return float(((om0 - omega[:z0]) / (pf[:z0] * (1.0 - om0))).max())


def optimum(thresholds, omega, pf, kappa):
    """Minimise the normalised cost rate over a frontier: ``(ell, omega, P_f, L)``."""
    L = cost_rate(omega, pf, kappa)
    j = int(np.argmin(L))
    return float(thresholds[j]), float(omega[j]), float(pf[j]), float(L[j])


def optimal_intervals(omega, pf):
    """Which operating points are ever optimal, and over which cost ratios.

    Each point contributes the affine function ``kappa -> (1 + kappa P_f) / (1 - omega)``,
    and the optimal cost is their lower envelope (Proposition 4). A point is optimal at
    some ``kappa > 0`` exactly when it is a vertex of the lower-left convex envelope of
    the achievable set (Proposition 2).

    Returns
    -------
    list of (index, kappa_low, kappa_high)
        Points in the order they become optimal as ``kappa`` grows from 0; the last
        interval is open to ``inf``.
    """
    omega = np.asarray(omega, float)
    pf = np.asarray(pf, float)
    a = 1.0 / (1.0 - omega)
    b = pf / (1.0 - omega)
    cur = int(np.lexsort((b, a))[0])
    lo = 0.0
    out = []
    while True:
        steeper = np.nonzero(b < b[cur] - _TIE)[0]
        if len(steeper) == 0:
            out.append((cur, lo, np.inf))
            return out
        cross = (a[steeper] - a[cur]) / (b[cur] - b[steeper])
        cross = np.maximum(cross, lo)
        first = cross.min()
        tied = steeper[cross <= first + _TIE]
        nxt = int(tied[np.argmin(b[tied])])
        if first > lo + _TIE:          # a tie at one cost ratio is not an interval
            out.append((cur, lo, float(first)))
        cur, lo = nxt, float(first)
