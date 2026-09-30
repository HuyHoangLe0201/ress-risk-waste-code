# riskwaste

**Risk–waste geometry for condition-based replacement.** A condition-based rule replaces a
unit when its predicted remaining useful life (RUL) falls to a threshold. For any predictor,
`riskwaste` tells you

1. **what a rule costs** — every replacement rule is a point `(omega, P_f)`: how often it
   is late, and what fraction of life it discards when early. The long-run cost rate is an
   exact function of that point, `Tbar g / c_p = (1 + kappa P_f) / (1 - omega)` with
   `kappa = (c_f - c_p) / c_p`, whatever the lifetime law and however the rule was built;
2. **where the optimal threshold lies** — the zero-failure threshold `ell_0` is the exact
   cost optimum if and only if `kappa >= kappa*`, a constant of the fleet and the predictor
   computable in closed form from calibration data before any policy is committed to;
3. **whether your fleet can resolve a comparison** — a unit-level bootstrap that refits the
   predictor and reselects every policy parameter in each replicate, for condition-based
   replacement against age replacement under a fitted lifetime law.

It is companion code to *Risk–waste geometry for condition-based replacement: Optimality
and finite-fleet estimation* by Huy Hoang Le and Kim-Anh Nguyen.

## Install

```bash
pip install git+https://github.com/HuyHoangLe0201/ress-risk-waste-code
```

Python 3.9 or later, with numpy and scipy.

## Quick start

```python
import riskwaste as rw

# One predicted-RUL path per calibration unit, observed up to its failure cycle,
# from any model you like.
policy = rw.ThresholdPolicy(predicted_paths)

policy.kappa_star()                  # the critical cost ratio kappa*
policy.optimum(kappa=9.0)            # (threshold, omega, P_f, cost rate)
rw.choose_threshold(predicted_paths, kappa=9.0)   # Algorithm 2: a conservative choice

# Algorithm 1: condition-based against age replacement, with an interval.
result = rw.paired_bootstrap(lifetimes, fit_predict, kappa=9.0, n_boot=200)
print(result)   # g_CBM / g_age = 0.701 [0.595, 0.953] ... on C-MAPSS FD002
```

`examples/quickstart.py` runs the whole method on a synthetic fleet in about a minute, with
nothing to download. `examples/cmapss.py` runs it on a C-MAPSS subset.

## Bring your own predictor

The method never looks inside the predictor. `ThresholdPolicy` takes predicted paths, and
`paired_bootstrap` takes a callable

```python
def fit_predict(train_units, predict_units):
    model = train_your_model([sequences[u] for u in train_units])
    return {u: model.predict(sequences[u]) for u in predict_units}
```

which it calls once per replicate with the calibration half of that replicate, so the
variability of learning the predictor is part of the interval. `riskwaste.predictors`
provides the ridge baseline used in the paper; a CNN, an LSTM or a gradient-boosted model
plugs in the same way.

## What is in the package

| | |
|---|---|
| `coordinates(T, tau)` | `(omega, P_f)` of any rule, from lifetimes and replacement times |
| `cost_rate(omega, pf, kappa)` | the normalised long-run cost rate |
| `coordinates_from_costs(L1, L2, k1, k2)` | recover `(omega, P_f)` from costs seen under two price regimes |
| `ThresholdPolicy(paths)` | a threshold rule on predicted RUL: `ell_u`, `ell0`, `frontier()`, `kappa_star()`, `optimum()`, `visible_fraction()` |
| `kappa_star(omega, pf)` | the critical cost ratio, from any frontier ordered by threshold |
| `optimal_intervals(omega, pf)` | which operating points are ever optimal, and over which cost ratios |
| `age_coordinates`, `optimal_age` | the age-replacement baseline, with a fitted lognormal, Weibull or gamma law |
| `paired_bootstrap(...)` | Algorithm 1, the unit-level bootstrap for a paired comparison |
| `choose_threshold(...)` | Algorithm 2, the conservative threshold rule |

## Conventions

- Time is counted in inspection cycles. `paths[u][k]` is the prediction at cycle `k + 1`,
  and a path runs to the failure cycle, so its length is the unit's lifetime. An alarm at
  the failure cycle comes too late and counts as a failure.
- `omega` is normalised by the mean life of the units it is averaged over. Two rules
  scored on the same units share that constant, so their cost ratio does not depend on it.
- The threshold search is exact by default. Between consecutive values of `ell_u` the
  failure probability is constant and the wasted life only grows, so the optimum at any
  cost ratio is one of at most `n + 1` candidates. Pass `thresholds=` to search a grid.
- Intervals from `paired_bootstrap` describe a procedure run on about `0.63 n` distinct
  units, so they are conservative: read them as a floor on what the fleet can resolve.

## Does it reproduce the paper?

`paper/check_package.py` runs the package on the paper's own pipeline and grids. It
matches the scripts that produced the reported numbers to within 1e-15 on FD002, FD004 and
the battery fleet: `ell_u`, the frontier, `kappa*`, and the 16.4%, 15.7% and 30.3% of
monitored cycles visible to a threshold rule. With the package's own conventions,
`examples/cmapss.py` gives `0.701 [0.595, 0.953]` on FD002 against the paper's
`0.701 [0.601, 0.960]`.

## Tests

```bash
python -m pytest tests
```

Each result the package relies on is checked against brute force: the cost identity
against the cost rate computed directly, the exact candidate set against every threshold
at which an alarm moves, `kappa*` on both sides of the if-and-only-if, and the bootstrap's
split over distinct units.

## Data

Nothing is redistributed. C-MAPSS is data set 6 of the
[NASA Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/);
the battery cells are those of Severson et al., *Nature Energy* 4:383–391, 2019.

## Reproducing the paper

`paper/` holds the scripts that produced every figure, table and reported number, kept
unchanged as the record of what was run. See `paper/README.md`.

## License

MIT, see `LICENSE`. The benchmark datasets carry their own terms.
