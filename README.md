# Risk--waste geometry for condition-based replacement

Code for *Risk--waste geometry for condition-based replacement: Optimality and
finite-fleet estimation*, by Huy Hoang Le and Kim-Anh Nguyen.

Every figure, table and reported number in the paper is produced by a script here.

## Layout

    *.py        analysis scripts, one result or family of results each
    shared/     figstyle.py and makefigs6.py as the scripts resolve them
    README.md   what each script is for, and the data it needs
    paths.txt   the files that still name an absolute path

`shared/` exists because most scripts put that directory at the front of `sys.path`. Two
modules have the same name in both places and differ; the copies in `shared/` are the ones
the reported numbers were produced with.

## Data

The benchmarks are public and are not redistributed here: C-MAPSS (NASA prognostics data
repository), the battery cells of Severson et al., *Nature Energy* 4:383--391, 2019, and
the PRONOSTIA and XJTU-SY bearing sets. Scripts reach the data by absolute path; edit the
constants listed in `paths.txt`, or place the data at the same locations.

Python 3.11 with numpy, scipy, matplotlib and scikit-learn.

## License

MIT, see `LICENSE`. It covers the code in this repository. The benchmark datasets are not
redistributed here and carry their own terms from the sources named above.
