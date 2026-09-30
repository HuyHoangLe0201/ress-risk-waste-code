"""Extract per-cycle multi-channel summaries from the Severson MIT/Stanford
fast-charging dataset (Nature Energy 2019) -- the knee-type battery data.
Reads only the 'summary' group (tiny) so the 3 GB HDF5 files stay on disk.
"""
import h5py, numpy as np, os
# The README documents SEVERSON_DIR as the way to point this at the release;
# the fallback is a sibling directory, so the script names no machine of its own.
SD = os.environ.get("SEVERSON_DIR",
                    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "severson"))
FILES = ["2017-05-12_batchdata_updated_struct_errorcorrect.mat",
         "2017-06-30_batchdata_updated_struct_errorcorrect.mat",
         "2018-04-12_batchdata_updated_struct_errorcorrect.mat"]
CH = ['QDischarge', 'QCharge', 'IR', 'Tavg', 'Tmax', 'Tmin', 'chargetime']

cells, names, lives = [], [], []
for bi, fn in enumerate(FILES):
    with h5py.File(os.path.join(SD, fn), 'r') as f:
        b = f['batch']; n = b['summary'].shape[0]
        for i in range(n):
            s = f[b['summary'][i, 0]]
            try:
                A = np.stack([np.array(s[c]).ravel().astype(float) for c in CH], 1)
                life = float(np.array(f[b['cycle_life'][i, 0]]).ravel()[0])
            except Exception:
                continue
            A = A[1:-1]                                  # drop first/last cycle (known artefacts)
            if len(A) < 100 or not np.isfinite(A).all():
                continue
            q = A[:, 0]
            if q[0] <= 0 or q.max() > 1.5 or q.min() < 0.5:   # nominal 1.1 Ah LFP
                continue
            # require genuine fade: end capacity below 95% of the early plateau
            if q[-1] > 0.95 * np.median(q[:20]):
                continue
            cells.append(A); names.append(f"b{bi+1}c{i}"); lives.append(life)
        print(f"{fn}: {n} raw -> running total kept {len(cells)}")

lens = np.array([len(c) for c in cells])
print(f"\nkept {len(cells)} cells; cycle counts: min={lens.min()} med={int(np.median(lens))} max={lens.max()}")
q0 = np.array([c[0, 0] for c in cells]); qE = np.array([c[-1, 0] for c in cells])
print(f"capacity start median={np.median(q0):.3f}  end median={np.median(qE):.3f}  "
      f"fade median={100*(1-np.median(qE/q0)):.1f}%")

# knee diagnostic: normalised capacity curve, second derivative concentration
g = np.linspace(0, 1, 101)
Q = np.array([np.interp(g, np.linspace(0, 1, len(c)), c[:, 0] / c[0, 0]) for c in cells])
mq = Q.mean(0)
d2 = np.gradient(np.gradient(mq, g), g)
print(f"mean normalised capacity: tau=0 -> {mq[0]:.3f}, 0.5 -> {mq[50]:.3f}, 1 -> {mq[-1]:.3f}")
print(f"|d2Q/dtau2| peak at tau = {g[np.argmax(np.abs(d2))]:.2f}  (knee location)")
print(f"linearity R^2 of mean curve = "
      f"{np.corrcoef(mq, g)[0,1]**2:.4f}   (NASA constant-load cells report 0.94-0.98)")

np.savez_compressed(os.path.join(os.path.dirname(os.path.abspath(__file__)), "severson_cells.npz"),
                    data=np.array(cells, dtype=object), names=np.array(names),
                    lives=np.array(lives), channels=np.array(CH), allow_pickle=True)
print("\nsaved severson_cells.npz")
