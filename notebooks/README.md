# Notebooks

Exploratory only. Anything that produces a number or figure appearing in the
dissertation must be moved into `src/` and committed as a script — a REPL
session is not reproducible six months later when a marker queries a figure.

Planned:
- `01_eda.ipynb`      Week 11-12. Class distribution, Amount by class, fraud
                      rate by hour, V1-V28 separability, correlation structure.
                      Target: 10-15 figures for Ch.1 and Ch.3.
- `02_baselines.ipynb` Week 15-16. Sanity-check baselines before the sweep.
- `03_results_analysis.ipynb`  Ch.5 tables and figures. All computation is
                      in src/results_analysis.py; the notebook calls it,
                      displays results and saves them to dissertation/.
                      Needs `make prepare`, `make sweep` and `make profile`
                      first.

Start each with:

    import sys; sys.path.insert(0, "..")
    from src.data_loader import FraudDataset, DataConfig
    ds = FraudDataset(DataConfig(csv_path="../data/raw/creditcard.csv"))
    ds.validate(); splits = ds.prepare()
