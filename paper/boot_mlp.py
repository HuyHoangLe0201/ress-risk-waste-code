"""Blocking item 2: is the central claim tied to my hand-built feature set?

Every result so far used ridge/RFF/kNN over [raw sensors, MA(5), MA(20)].  The
published C-MAPSS predictors instead feed a RAW SLIDING WINDOW of sensor readings to a
network (Li 2018 CNN, Zheng 2017 LSTM).  Here: a 20-cycle raw window flattened into an
MLP -- same input convention as the published models, different function class from
everything used so far, and no hand-built features at all.

Re-runs headline 5.1 (g_CBM / g_age) under the unit-level bootstrap.  If the ratio
holds, the central claim is not an artefact of the feature pipeline.

Writes to D: drive C has 0.1 GB free.
"""
import numpy as np, sys
from scipy.stats import lognorm
from sklearn.neural_network import MLPRegressor

CM = r"D:\bách khoa Đà Nẵng\nghiencuukhoahoc\baotri\data\CMAPSSData"
W = 20            # window length, cycles
STRIDE = 2        # subsample rows for speed
CF = 10.0


def load(fn):
    d = np.loadtxt(CM + "\\" + fn)
    unit, cyc, X = d[:, 0].astype(int), d[:, 1].astype(int), d[:, 2:]
    units = np.unique(unit)
    life = np.array([float(cyc[unit == u].max()) for u in units])
    X = X[:, X.std(0) > 1e-8]
    X = (X - X.mean(0)) / X.std(0)
    seq = [X[unit == u] for u in units]
    # raw sliding window, left-padded with the first observation
    win, R = [], []
    for s in seq:
        n = len(s)
        pad = np.vstack([np.repeat(s[:1], W - 1, axis=0), s])
        idx = np.arange(W)[None, :] + np.arange(n)[:, None]
        win.append(pad[idx].reshape(n, -1))
        R.append(np.arange(n - 1, -1, -1.0))
    return win, R, life


def fit_predict(Xtr, ytr, Xte, seed):
    m = MLPRegressor(hidden_layer_sizes=(64,), max_iter=60, random_state=seed,
                     early_stopping=True, n_iter_no_change=6, batch_size=256,
                     learning_rate_init=3e-3).fit(Xtr, ytr)
    return [m.predict(x) for x in Xte]


def draw(win, R, life, n, rg, boot=True):
    idx = rg.choice(n, size=n, replace=True) if boot else rg.permutation(n)
    uq = np.unique(idx)
    if len(uq) < 20:
        return None
    pm = rg.permutation(len(uq))
    calu = set(uq[pm[:len(uq) // 2]])
    cal = np.array([i for i in idx if i in calu])
    val = np.array([i for i in idx if i not in calu])
    if len(cal) < 12 or len(val) < 12:
        return None
    cu = np.unique(cal)
    Xtr = np.vstack([win[i][::STRIDE] for i in cu])
    ytr = np.concatenate([R[i][::STRIDE] for i in cu])
    allu = np.unique(idx)
    pred = fit_predict(Xtr, ytr, [win[i] for i in allu], int(rg.integers(1e6)))
    pa = {i: p for i, p in zip(allu, pred)}
    rm = {i: np.minimum.accumulate(pa[i]) for i in pa}
    Tb = life[idx].mean()

    def cbm(ix, ell):
        c, cl = [], []
        for i in ix:
            h = np.nonzero(rm[i] <= ell)[0]
            if len(h) == 0 or R[i][h[0]] <= 0:
                c.append(CF); cl.append(life[i])
            else:
                c.append(1.0); cl.append(life[i] - R[i][h[0]])
        return np.mean(c) / np.mean(cl)
    lu = np.array([pa[i][R[i] > 0].min() for i in cu])
    grid = np.linspace(np.quantile(lu, .02), np.quantile(lu, .999) + 0.3 * Tb, 45)
    ell = grid[int(np.argmin([cbm(cal, e) for e in grid]))]
    g_c = cbm(val, ell)
    Lc, Lv = life[cal], life[val]
    s, _, sc = lognorm.fit(Lc, floc=0)
    t = np.linspace(1.0, 4 * Lc.max(), 5000)
    S = lognorm.sf(t, s, scale=sc)
    cum = np.concatenate([[0.0], np.cumsum((S[1:] + S[:-1]) / 2 * np.diff(t))])
    tR = t[10 + int(np.argmin(((S + CF * (1 - S)) / np.maximum(cum, 1e-12))[10:]))]
    g_a = np.where(Lv <= tR, CF, 1.0).mean() / np.minimum(Lv, tR).mean()
    err = np.concatenate([pa[i] - R[i] for i in val])
    return g_c / g_a, np.sqrt((err ** 2).mean()) / Tb


print("=" * 78)
print(f"  HEADLINE 5.1 WITH A PUBLISHED-STYLE PREDICTOR (raw {W}-cycle window -> MLP)")
print("=" * 78)
print(f"  {'fleet':>7} {'n':>5} {'RMSE/Tbar':>10} {'point':>7} {'boot mean':>10} "
      f"{'95% CI':>18} {'wins':>6}", flush=True)
for nm in sys.argv[1:] or ["FD002", "FD004"]:
    win, R, life = load(f"train_{nm}.txt")
    n = len(win)
    pt = [draw(win, R, life, n, np.random.default_rng(s), False) for s in range(6)]
    pt = [v for v in pt if v]
    bs = [draw(win, R, life, n, np.random.default_rng(7000 + s), True)
          for s in range(45)]
    bs = [v for v in bs if v]
    r = np.array([v[0] for v in bs])
    rm_ = np.mean([v[1] for v in bs])
    print(f"  {nm:>7} {n:5d} {rm_:10.3f} {np.mean([v[0] for v in pt]):7.3f} "
          f"{r.mean():10.3f}   [{np.percentile(r,2.5):.3f}, {np.percentile(r,97.5):.3f}]"
          f" {100*np.mean(r<1):5.0f}%", flush=True)
