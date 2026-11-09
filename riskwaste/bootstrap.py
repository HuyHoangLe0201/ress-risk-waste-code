"""Interval estimates for a policy comparison on a finite fleet, and a threshold rule.

``paired_bootstrap`` is Algorithm 1 of the paper. What varies between fleets of the same
size is not only which units fail, but which predictor and which threshold the procedure
learns from them, so each replicate redoes everything a practitioner would:

1. draw ``n`` units with replacement, and split the *distinct* units into halves, so that
   no unit both trains the predictor and scores it;
2. train the predictor and select every policy parameter on the calibration half;
3. score both policies on the same validation half, which makes the statistic a paired
   ratio.

A replicate holds about ``0.63 n`` distinct units, so the predictor is trained on less
information than in the original fleet and the interval describes a procedure run on a
smaller fleet. Its width therefore reflects that smaller fleet. Whether it is conservative
for the procedure at ``n`` depends on the learning curve, which we do not assume: the
measured comparison of a ``0.63 n`` and a ``0.85 n`` sub-fleet runs the *other* way on both
turbofan fleets (Supplementary Section S9.2).

``choose_threshold`` is Algorithm 2: adopt the zero-failure threshold only when the cost
ratio clears an upper bootstrap quantile of ``kappa*``, not the point estimate, which a
finite fleet biases downward.
"""
from dataclasses import dataclass, field

import numpy as np

from .age import optimal_age
from .coordinates import coordinates, cost_rate
from .threshold import ThresholdPolicy, kappa_star, optimum


@dataclass
class PairedResult:
    """Replicates of ``g_CBM / g_age`` and their summary."""
    ratios: np.ndarray
    alpha: float
    kappa_star: np.ndarray = field(default_factory=lambda: np.empty(0))

    @property
    def mean(self):
        return float(self.ratios.mean())

    @property
    def interval(self):
        q = np.quantile(self.ratios, [self.alpha / 2.0, 1.0 - self.alpha / 2.0])
        return float(q[0]), float(q[1])

    @property
    def wins(self):
        """Share of replicates in which condition-based replacement is cheaper."""
        return float(np.mean(self.ratios < 1.0))

    @property
    def excludes_parity(self):
        return self.interval[1] < 1.0 or self.interval[0] > 1.0

    def __str__(self):
        lo, hi = self.interval
        return ("g_CBM / g_age = %.3f  [%.3f, %.3f]  (%d replicates, %.0f%% interval), "
                "condition-based cheaper in %.0f%%"
                % (self.mean, lo, hi, len(self.ratios), 100 * (1 - self.alpha),
                   100 * self.wins))


def paired_bootstrap(lifetimes, fit_predict, kappa, n_boot=200, alpha=0.05,
                     age_family="lognormal", thresholds=None, seed=0, min_units=5):
    """Algorithm 1: a unit-level bootstrap for condition-based against age replacement.

    Parameters
    ----------
    lifetimes : array of shape (n,)
        Lifetime of each unit in cycles, equal to the length of its predicted path.
    fit_predict : callable ``(train_units, predict_units) -> {unit: predicted RUL path}``
        Trains your predictor on the units in ``train_units`` (distinct indices) and
        returns a predicted-RUL path for every unit in ``predict_units``. Any model works;
        it is refitted in every replicate.
    kappa : float
        Cost ratio ``(c_f - c_p) / c_p``.
    thresholds : None, array, or callable ``policy, cal -> array``
        Where to search for the threshold on the calibration half. ``None`` searches the
        exact candidate set.
    age_family : str
        Lifetime law fitted to the calibration half for the age-replacement baseline.

    Returns
    -------
    PairedResult
        Also carries the ``kappa*`` of each replicate's calibration half, which
        ``choose_threshold`` accepts as its resampling distribution.
    """
    T = np.asarray(lifetimes, float)
    n = len(T)
    rng = np.random.default_rng(seed)
    ratios, kstars = [], []
    for _ in range(n_boot):
        draw = rng.integers(0, n, size=n)
        units = np.unique(draw)
        order = units.copy()
        rng.shuffle(order)
        cal_units = np.sort(order[: len(order) // 2])
        val_units = np.sort(order[len(order) // 2:])
        if len(cal_units) < min_units or len(val_units) < min_units:
            continue
        in_cal = np.isin(draw, cal_units)
        cal, val = draw[in_cal], draw[~in_cal]

        paths = fit_predict(cal_units, units)
        pol = ThresholdPolicy([paths[u] for u in units])
        local = np.searchsorted(units, np.arange(n))
        if not np.allclose(pol.lifetimes, T[units]):
            raise ValueError("a predicted path does not match its unit's lifetime")
        cal_l, val_l = local[cal], local[val]

        th = (None if thresholds is None else
              thresholds(pol, cal_l) if callable(thresholds) else thresholds)
        th, om, pf = pol.frontier(th, units=cal_l)
        ell = optimum(th, om, pf, kappa)[0]
        g_cbm = cost_rate(*pol.coordinates(ell, units=val_l), kappa)

        t_R = optimal_age(T[cal], kappa, family=age_family)
        g_age = cost_rate(*coordinates(T[val], np.full(len(val), t_R)), kappa)

        ratios.append(g_cbm / g_age)
        if np.any(pf == 0):
            kstars.append(kappa_star(om, pf))
    if not ratios:
        raise ValueError("no replicate had %d units in each half" % min_units)
    return PairedResult(np.array(ratios), alpha, np.array(kstars))


@dataclass
class ThresholdChoice:
    """What Algorithm 2 decided, and the numbers it decided on."""
    threshold: float
    rule: str
    kappa: float
    kappa_star: float
    kappa_star_upper: float
    omega: float
    pf: float

    def __str__(self):
        return ("%s: threshold %.4g (omega %.3f, P_f %.3f); kappa %.3g against "
                "kappa* %.3g, upper quantile %.3g"
                % (self.rule, self.threshold, self.omega, self.pf, self.kappa,
                   self.kappa_star, self.kappa_star_upper))


def choose_threshold(predicted_rul, kappa, alpha=0.05, n_boot=500, seed=0,
                     kappa_star_draws=None):
    """Algorithm 2: choose the threshold at a given cost ratio, conservatively.

    ``kappa*`` is estimated on the calibration fleet and resampled over whole units. If
    ``kappa`` clears the ``(1 - alpha)`` quantile, the zero-failure threshold ``ell_0`` is
    adopted; otherwise the empirical cost minimiser is returned, and its interval should
    be reported with ``paired_bootstrap``.

    By default the predicted paths are held fixed while units are resampled. Pass the
    ``kappa_star`` of a ``paired_bootstrap`` result as ``kappa_star_draws`` to use
    replicates in which the predictor was refitted as well, as the paper does; that
    distribution is wider, so the rule is more conservative.
    """
    pol = ThresholdPolicy(predicted_rul)
    th, om, pf = pol.frontier()
    k_hat = kappa_star(om, pf)
    if kappa_star_draws is None:
        rng = np.random.default_rng(seed)
        n = pol.n_units
        draws = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            _, om_b, pf_b = pol.frontier(units=idx)
            draws.append(kappa_star(om_b, pf_b))
        kappa_star_draws = np.array(draws)
    upper = float(np.quantile(np.asarray(kappa_star_draws, float), 1.0 - alpha))
    if kappa > upper:
        o, p = pol.coordinates(pol.ell0)
        return ThresholdChoice(pol.ell0, "zero-failure threshold", kappa, k_hat,
                               upper, o, p)
    ell, o, p, _ = optimum(th, om, pf, kappa)
    return ThresholdChoice(ell, "empirical minimiser (report its interval)", kappa,
                           k_hat, upper, o, p)
