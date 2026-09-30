"""A baseline RUL predictor, so the method runs end to end out of the box.

Nothing in the method depends on the predictor. ``ThresholdPolicy`` takes predicted paths
from any model, and ``paired_bootstrap`` takes any ``fit_predict`` callable. This one is
the ridge regression the paper uses for most of its results: each cycle is described by
its sensor channels, their trailing means over the last 5 and 20 cycles, and an intercept,
and the target is the true remaining life.

Channels are standardised on the training units only, so no information from the units
being scored enters the fit.
"""
import numpy as np


def _features(seq, windows):
    seq = np.asarray(seq, float)
    n = len(seq)
    csum = np.vstack([np.zeros((1, seq.shape[1])), seq]).cumsum(0)
    t = np.arange(1, n + 1)
    cols = [seq]
    for w in windows:
        lo = np.maximum(t - w, 0)
        cols.append((csum[t] - csum[lo]) / np.minimum(t, w)[:, None])
    cols.append(np.ones((n, 1)))
    return np.hstack(cols)


class RidgeRUL:
    """Ridge regression of true remaining life on sensor channels and trailing means."""

    def __init__(self, windows=(5, 20), lam=0.1):
        self.windows = tuple(windows)
        self.lam = lam

    def fit(self, sequences):
        """``sequences``: list of (cycles, channels) arrays, each run to failure."""
        S = np.vstack(sequences)
        sd = S.std(0)
        self.keep_ = sd > 1e-8
        self.mean_, self.std_ = S[:, self.keep_].mean(0), sd[self.keep_]
        X = np.vstack([self._design(s) for s in sequences])
        y = np.concatenate([np.arange(len(s) - 1, -1, -1.0) for s in sequences])
        A = X.T @ X + self.lam * np.eye(X.shape[1])
        self.coef_ = np.linalg.solve(A, X.T @ y)
        return self

    def predict(self, seq):
        """Predicted remaining life at every cycle of one sequence."""
        return self._design(seq) @ self.coef_

    def _design(self, seq):
        z = (np.asarray(seq, float)[:, self.keep_] - self.mean_) / self.std_
        return _features(z, self.windows)


def ridge_fit_predict(sequences, **kwargs):
    """A ``fit_predict`` callable for ``paired_bootstrap``, built on ``RidgeRUL``."""
    def fit_predict(train_units, predict_units):
        model = RidgeRUL(**kwargs).fit([sequences[u] for u in train_units])
        return {u: model.predict(sequences[u]) for u in predict_units}
    return fit_predict
