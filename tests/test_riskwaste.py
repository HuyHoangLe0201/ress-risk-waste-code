"""Each result the package relies on, checked against brute force.

Run with ``python -m pytest tests``. Nothing here needs data beyond what is generated.
"""
import numpy as np
import pytest

import riskwaste as rw
from riskwaste.datasets import synthetic_fleet
from riskwaste.predictors import ridge_fit_predict


def random_paths(n=60, seed=0):
    """Noisy predicted-RUL paths that trend down to failure, as a real predictor's do."""
    rng = np.random.default_rng(seed)
    paths = []
    for T in rng.integers(30, 120, n):
        true = np.arange(T - 1, -1, -1.0)
        paths.append(true + rng.normal(0, 8, T) + np.cumsum(rng.normal(0, 1, T)))
    return paths


# ---- Lemma 1 and Theorem 1 ---------------------------------------------------

def test_identity_equals_the_cost_rate_computed_directly():
    rng = np.random.default_rng(1)
    T = rng.gamma(4.0, 25.0, 500)
    tau = np.where(rng.random(500) < 0.2, np.inf, rng.uniform(10, 150, 500))
    for kappa in (0.5, 4.0, 29.0):
        c_p, c_f = 1.0, 1.0 + kappa
        direct = np.where(T <= tau, c_f, c_p).sum() / np.minimum(T, tau).sum()
        omega, pf = rw.coordinates(T, tau)
        assert rw.cost_rate(omega, pf, kappa) * c_p / T.mean() == pytest.approx(direct)


def test_two_price_regimes_recover_both_coordinates():
    omega, pf = 0.137, 0.042
    L1, L2 = rw.cost_rate(omega, pf, 3.0), rw.cost_rate(omega, pf, 40.0)
    assert rw.coordinates_from_costs(L1, L2, 3.0, 40.0) == pytest.approx((omega, pf))


# ---- threshold policies: Theorem 4, Proposition 3 ----------------------------

def test_alarm_times_match_a_cycle_by_cycle_scan():
    pol = rw.ThresholdPolicy(random_paths())
    for ell in (-5.0, 3.0, 17.5, 60.0):
        want = []
        for p in pol.paths:
            m = np.minimum.accumulate(p)
            hit = np.nonzero(m <= ell)[0]
            want.append(hit[0] + 1.0 if len(hit) else np.inf)
        assert np.array_equal(pol.alarm_times(ell), want)


def test_failure_probability_is_the_upper_tail_of_ell_u():
    pol = rw.ThresholdPolicy(random_paths())
    for ell in np.linspace(pol.ell_u.min() - 5, pol.ell0 + 5, 41):
        assert pol.coordinates(ell)[1] == pytest.approx(np.mean(pol.ell_u > ell))
    assert pol.coordinates(pol.ell0)[1] == 0.0


def test_changing_invisible_cycles_changes_nothing():
    paths = random_paths(seed=3)
    pol = rw.ThresholdPolicy(paths)
    rng = np.random.default_rng(4)
    moved = []
    for u, p in enumerate(paths):
        q = p.copy()
        hidden = ~pol.record_mask(u)
        q[hidden] = np.minimum.accumulate(p)[hidden] + rng.uniform(0, 500, hidden.sum())
        moved.append(q)
    grid = np.linspace(pol.ell_u.min() - 10, pol.ell0 + 10, 300)
    a, b = pol.frontier(grid), rw.ThresholdPolicy(moved).frontier(grid)
    assert np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2])
    assert 0.0 < pol.visible_fraction() < 1.0


# ---- where the optimum lies: Proposition 4, Theorem 5 ------------------------

def test_the_candidate_set_is_exact():
    """The optimum over every threshold at which some alarm time changes equals the
    optimum over the n + 1 candidates."""
    pol = rw.ThresholdPolicy(random_paths(seed=5))
    every = np.unique(np.concatenate([np.minimum.accumulate(p) for p in pol.paths]))
    every = np.concatenate([[-np.inf], every])
    for kappa in (0.1, 1.0, 3.0, 12.0, 80.0):
        assert pol.optimum(kappa)[3] == pytest.approx(pol.optimum(kappa, thresholds=every)[3])


@pytest.mark.parametrize("seed", range(6))
def test_zero_failure_threshold_is_optimal_exactly_above_kappa_star(seed):
    pol = rw.ThresholdPolicy(random_paths(seed=seed))
    ks = pol.kappa_star()
    assert ks > 0
    assert pol.optimum(ks * (1 + 1e-6) + 1e-9)[0] == pol.ell0
    assert pol.optimum(ks * (1 - 1e-3))[0] != pol.ell0


def test_optimal_intervals_agree_with_a_scan_over_cost_ratios():
    th, om, pf = rw.ThresholdPolicy(random_paths(seed=7)).frontier()
    pieces = rw.optimal_intervals(om, pf)
    assert pieces[0][1] == 0.0 and pieces[-1][2] == np.inf
    for kappa in np.geomspace(1e-3, 1e4, 400):
        j = next(i for i, lo, hi in pieces if lo <= kappa <= hi)
        L = rw.cost_rate(om, pf, kappa)
        assert L[j] == pytest.approx(L.min(), rel=1e-12)


# ---- the age baseline --------------------------------------------------------

def test_optimal_age_is_finite_under_wear_out_and_infinite_otherwise():
    rng = np.random.default_rng(8)
    wear_out = rng.weibull(3.0, 400) * 100
    early = rng.weibull(0.7, 400) * 100
    assert np.isfinite(rw.optimal_age(wear_out, 9.0, family="weibull"))
    assert rw.optimal_age(early, 9.0, family="weibull") == np.inf


# ---- Algorithms 1 and 2 ------------------------------------------------------

def test_bootstrap_trains_only_on_the_calibration_units():
    seqs, life = synthetic_fleet(80, seed=2)
    inner = ridge_fit_predict(seqs)
    seen = []

    def spy(train, predict):
        seen.append((np.asarray(train), np.asarray(predict)))
        return inner(train, predict)

    res = rw.paired_bootstrap(life, spy, kappa=9.0, n_boot=12, seed=3)
    assert len(res.ratios) == len(seen) == 12
    for train, predict in seen:
        assert np.all(np.diff(train) > 0)                 # distinct units
        assert np.all(np.isin(train, predict))
        assert len(train) == len(predict) // 2           # half of the distinct units
    assert np.all(np.isfinite(res.ratios)) and len(res.kappa_star) == 12
    again = rw.paired_bootstrap(life, inner, kappa=9.0, n_boot=12, seed=3)
    assert np.array_equal(res.ratios, again.ratios)


def test_threshold_rule_takes_the_boundary_only_at_high_cost_ratios():
    paths = random_paths(seed=9)
    high = rw.choose_threshold(paths, kappa=1e4, n_boot=100)
    assert high.rule.startswith("zero-failure") and high.pf == 0.0
    low = rw.choose_threshold(paths, kappa=0.0, n_boot=100)
    assert low.rule.startswith("empirical") and low.threshold == -np.inf
