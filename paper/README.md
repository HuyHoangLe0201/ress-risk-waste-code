# The scripts behind the paper

These are the scripts that produced every figure, table and reported number in the paper,
kept unchanged as the record of what was run. To use the method on your own fleet, use the
`riskwaste` package at the top of the repository instead; `check_package.py` here shows it
reproduces these scripts exactly on their own inputs.

## Layout

    *.py              one result or family of results each; the docstring says which
    makefigs7.py      writes fig1.pdf
    makefigs6.py      writes fig2.pdf, fig3.pdf and fig4.pdf
    makefigs_ress.py  writes fig6.pdf and fig7.pdf
    shared/           figstyle.py and makefigs6.py as the scripts resolve them
    check_package.py  the package against these scripts, on the same fleets and grids
    paths.txt         the scripts that still name an absolute path

`shared/` exists because most scripts put that directory at the front of `sys.path`. Two
modules have the same name in both places and differ; the copies in `shared/` are the ones
the reported numbers were produced with.

## Data

The scripts reach the data by absolute path: edit the constants listed in `paths.txt`, or
place the data at the same locations.

The battery scripts do not read the Severson release directly. `extract_severson.py` reads
the three `.mat` batch files -- set `SEVERSON_DIR` to the directory holding them -- and
writes `severson_cells.npz` beside itself. It applies the screening of Supplementary
Section S8: drop the first and last cycle as known artefacts, then keep a cell only if it
has at least 100 cycles with finite summaries, a discharge capacity inside the nominal
1.1 Ah range throughout, and an end capacity below 95% of the median of its first twenty.
That leaves 135 cells. Point the `TSP` constant of `shared/makefigs6.py` at the directory
holding the `.npz`.

Python 3.11 with numpy, scipy, matplotlib and scikit-learn; `extract_severson.py` also needs
h5py.

## One difference from the package

`makefigs6.frontier` normalises the calibration half's wasted life by the mean life of the
whole fleet, where the package, following Lemma 1, uses the mean life of the units it
averages over. `kappa*`, the optimal threshold and every ratio on a common set are
unchanged by the choice; the absolute value of `omega` is not. `check_package.py` prints
both.
