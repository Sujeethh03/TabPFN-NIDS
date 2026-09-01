# TabPFN-NIDS

**Scaling Tabular Foundation Models for Network Intrusion Detection**

[![Repository](https://img.shields.io/badge/GitHub-TabPFN--NIDS-181717?logo=github)](https://github.com/Sujeethh03/TabPFN-NIDS)
[![Python](https://img.shields.io/badge/python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-163%20passing-success)](tests/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

Applying TabPFN v2 (Hollmann et al., *Nature* 2025) to network intrusion
detection, and extending it past its 10,000-sample context limit with a
stratified chunked ensemble.

**Repository:** <https://github.com/Sujeethh03/TabPFN-NIDS>

## Quick start

With Python 3.11 and an activated virtualenv (full setup in [§1](#1-setup)):

```bash
pip install -r requirements.txt && pip install -e .
kaggle datasets download -d hassan06/nslkdd -p data/raw/nsl-kdd --unzip
python scripts/run_baseline.py --seed 42 --test-size 5000 --n-estimators 2
```

Writes a timestamped CSV to `reports/`. Expect **F1 ≈ 0.75, ROC-AUC ≈ 0.95**.
Budget the runtime first — see [§3](#3-run-the-experiments); the same command
without `--test-size` and `--n-estimators` runs for hours.

For the full reproduction guide — hardware requirements, dataset acquisition,
per-experiment expected outputs, tolerance bands and troubleshooting — see
**[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)**.

> **On terminology:** TabPFN takes labelled rows as in-context examples at
> inference time, so this is *in-context learning*, not zero-shot. The term
> "zero-shot" is used only for the cross-dataset transfer experiment, where no
> target-domain labels are involved.

---

## 1. Setup

```bash
git clone https://github.com/Sujeethh03/TabPFN-NIDS.git
cd TabPFN-NIDS

python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .          # required: notebooks and tests import the package by name
```

`pip install -e .` is **not optional**. This is a src-layout project: the
importable package is `tabpfn_nids`, living in `src/`. Without the editable
install nothing can import it.

```python
from tabpfn_nids.data_pipeline import load_nsl_kdd     # correct
from src.data_pipeline.loader import load_nsl_kdd      # wrong - there is no `src` package
```

If you hit `ModuleNotFoundError`, the fix is `pip install -e .`, not
`PYTHONPATH=.`. Setting `PYTHONPATH` to the repo root makes `import src.*`
appear to work and then breaks the moment you run from another directory.

Verify the environment:

```bash
python smoke_test.py
```

Runs TabPFN on a small sklearn dataset and prints versions, device and the
pinned checkpoint. **~14s**, or a few minutes the first time while the
checkpoint downloads. Exits 1 on failure.

```
  python         3.11.9 (arm64)
  torch          2.13.0
  tabpfn         8.5.0
  checkpoint     tabpfn-v2-classifier.ckpt
  device         mps
  ...
  Accuracy       0.9649
Smoke test PASSED. Environment is ready.
```

---

## 2. Get the data

Datasets are gitignored, so a fresh clone has none. NSL-KDD is the only one
needed to reproduce every result in this repo.

```bash
kaggle datasets download -d hassan06/nslkdd -p data/raw/nsl-kdd --unzip
```

This needs a Kaggle API token in `~/.kaggle/kaggle.json`. You can equally
download the archive by hand — all that matters is that `KDDTrain+.txt` and
`KDDTest+.txt` end up in `data/raw/nsl-kdd/`.

Check it landed:

```bash
python -c "from tabpfn_nids.data_pipeline import load_nsl_kdd; \
           tr, te = load_nsl_kdd(); print(tr.shape, te.shape)"
# (125973, 43) (22544, 43)
```

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) §3 for the other two
datasets and the data-quality defects in CIC-IDS-2018.

---

## 3. Run the experiments

Every script appends a timestamped CSV to `reports/`, carrying its own
provenance: seed, hardware, device, library versions, checkpoint and git commit.

```bash
# Baseline - vanilla TabPFN, one 10,000-row context
python scripts/run_baseline.py --seed 42 --test-size 5000 --n-estimators 2

# Enhancement 1 - stratified chunked ensemble
python scripts/run_enhanced.py --seed 42 --test-size 5000 --n-estimators 2

# Enhancement 2 - engineered-feature ablation, with/without
python scripts/run_feature_ablation.py --seed 42 --max-chunks 3 --test-size 1000 \
       --n-estimators 2 --max-service-flag-categories 40

# Roll results up into the comparison table
python scripts/summarize_runs.py --tag baseline
python scripts/generate_comparison_table.py   # -> reports/comparison_table.{md,csv}
python scripts/build_comparison_table.py      # -> reports/tables/comparison_table.{md,csv}
```

The two table scripts are **not** interchangeable and write to different
places. `generate_comparison_table.py` builds the headline three-arm table
(vanilla / chunked ensemble / +engineered) at `reports/comparison_table.md`.
`build_comparison_table.py` emits a per-experiment breakdown with mean ± std
across seeds at `reports/tables/comparison_table.md`.

`--n-estimators`, `--test-size` and the seed **must match across arms** or the
comparison measures the wrong thing.

### Budget the runtime before you start

Prediction cost is linear in `--test-size` and **superlinear** in context size.
Measured on an M1, `n_estimators=2`, seconds per 1,000 test rows:

| context rows | 1,000 | 2,500 | 5,000 | 10,000 |
|---|---|---|---|---|
| s / 1,000 test rows | 15 | 49 | 144 | **459** |

So `run_baseline.py --test-size 5000` at the full 10,000-row context is a
**~40 minute** job, not a 3-minute one. For a quick check use:

```bash
python scripts/run_baseline.py --context-size 5000 --test-size 500 --n-estimators 2
```

which finishes in about 75s. The chunked ensemble multiplies this by the chunk
count — full 51-chunk coverage of NSL-KDD costs roughly 2.5s *per test row*.

---

## 4. See the results

### The notebooks

Four notebooks, meant to be read top to bottom. Runtimes are measured on an M1.

| notebook | what it shows | runtime |
|---|---|---|
| [`01_reproduction.ipynb`](notebooks/01_reproduction.ipynb) | the baseline end to end: load, dataset stats, preprocessing, fit, metrics, confusion matrix | **46s** |
| [`02_enhancement.ipynb`](notebooks/02_enhancement.ipynb) | single context vs chunked ensemble vs +engineered features, on one fixed test set | **4.4 min** |
| [`03_ablations.ipynb`](notebooks/03_ablations.ipynb) | context-size and stratification ablations, from cached CSVs | **3s** |
| [`04_results_summary.ipynb`](notebooks/04_results_summary.ipynb) | every result CSV, the comparison table, all figures, written findings | **4s** |

```bash
jupyter lab notebooks/01_reproduction.ipynb
```

03 and 04 re-run no inference — they only read `reports/` — so they execute in
seconds and are safe to run anywhere. 01 and 02 call TabPFN live.

Notebooks 01 and 02 run at **reduced scale** so they finish in minutes; each one
says so at the top and prints its own numbers next to the full-scale runs of
record. The pipeline is identical, only the row counts differ.

Execute them all headlessly:

```bash
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=3600 \
    --inplace notebooks/0[1-4]*.ipynb
```

### The raw results

```
reports/
├── baseline_<timestamp>_seed<N>.csv    one row per run, with full provenance
├── feature_ablation_*.csv              Enhancement 2, with/without arms
├── ablation_*.csv                      context-size and stratification sweeps
├── comparison_table.{csv,md}           all arms side by side
└── figures/                            confusion matrices, ROC, PR curves
```

`reports/comparison_table.md` (from `generate_comparison_table.py`) is the
headline summary artifact. Arms that have not
been run are printed as *not run* rather than omitted.

**Read the noise floor before reading any delta.** The baseline's F1 standard
deviation across three seeds is **0.0229**. A difference smaller than that is
seed-to-seed variation, not an effect — the notebooks apply this rule explicitly
rather than leaving it to the reader.

---

## 5. Run the tests

```bash
pytest                # fast suite, ~10s
pytest -v             # per-test names
pytest -m slow        # adds real TabPFN inference (minutes)
pytest tests/test_chunker.py -v      # one module
```

Expected: **163 passed, 4 deselected**. The 4 deselected carry the `slow`
marker — they run real TabPFN inference and are excluded from the default suite
by `addopts = "-q -m 'not slow'"` in `pyproject.toml`. That is configuration,
not failure.

The suite covers the loader, preprocessor, chunker, ensemble, feature
engineering, metrics, evaluation and repo structure. It also asserts the
checkpoint pin never reverts to `"auto"` — see the warning below for why that
matters.

---

## Results snapshot

Vanilla TabPFN on NSL-KDD: 10,000-row stratified context, 5,000-row stratified
test sample, `n_estimators=2`, MPS backend, 3 seeds.

| metric | mean ± std |
|---|---|
| accuracy | 0.7653 ± 0.0212 |
| precision | 0.9347 ± 0.0269 |
| recall | 0.6318 ± 0.0213 |
| **f1_score** | **0.7539 ± 0.0229** |
| **roc_auc** | **0.9555 ± 0.0040** |

Two things this table is really saying:

**The error is almost all recall.** Precision 0.93 against recall 0.63 — the
model is right when it flags an attack but misses about a third of them. That is
the expected consequence of NSL-KDD's 17 attack types that appear only in the
test split, not a bug.

**ROC-AUC 0.9555 versus F1 0.7539.** The model *ranks* attacks well; the default
0.5 threshold just places the operating point poorly. Threshold tuning is the
cheapest available improvement.

The F1 std of **0.0229** is the noise floor: a single-seed delta below ~2.3
percentage points is not evidence of an effect.

---

## Project layout

```
tabpfn-nids/
├── src/tabpfn_nids/          # the installed package
│   ├── config.py             # paths, seed, TabPFN limits, provenance capture
│   ├── data_pipeline/        # loader.py, preprocessor.py
│   ├── features/             # engineered.py  (Enhancement 2)
│   ├── models/               # tabpfn_wrapper.py, chunker.py,
│   │                         # chunked_ensemble.py  (Enhancement 1)
│   ├── evaluation/           # metrics, significance, reporter, plots
│   └── utils/
├── scripts/                  # experiment runners (thin CLI wrappers)
├── notebooks/                # 01 reproduction, 02 enhancement,
│                             # 03 ablations, 04 results summary
├── tests/                    # 163 tests (+4 slow, deselected by default)
├── data/raw/                 # gitignored; .gitkeep keeps the tree
├── reports/                  # results CSVs, figures, tables
└── docs/                     # REPRODUCIBILITY, CONTRIBUTIONS, FUTURE_WORK
```

## Two things that will bite you

**The default checkpoint is gated and is the wrong model.** `tabpfn` 8.5.0
defaults to `model_path="auto"`, which resolves to the gated HuggingFace repo
`Prior-Labs/tabpfn_3` — a *different, newer* model than the Nature 2025
publication. This project pins `tabpfn-v2-classifier.ckpt` from the ungated
`Prior-Labs/TabPFN-v2-clf`, and a test asserts it never reverts to `"auto"`.

**MPS memory is the binding constraint.** Metal does not release memory when a
Python reference is dropped, so a sequential chunked run accumulates
allocations until it dies — ours did, at chunk 4 of 13. `TabPFNWrapper.free()`
fixes it. Separately, engineered features at full cardinality (463 columns)
exhaust MPS regardless of batch size; use `--max-service-flag-categories`.

Both are documented in detail in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

---

## Status

| | Enhancement | Status |
|---|---|---|
| 1 | Stratified chunked ensemble | implemented and tested; full 13-chunk run not yet completed |
| 2 | Domain-aware feature engineering | implemented and tested; ablation at reduced scope (memory) |
| 3 | Cross-dataset transfer | not implemented |

Classical baselines (XGBoost, LightGBM) are not yet implemented.

## Datasets

| dataset | rows | used for |
|---|---|---|
| [NSL-KDD](https://www.kaggle.com/datasets/hassan06/nslkdd) | 125,973 / 22,544 | main benchmark |
| [UNSW-NB15](https://www.kaggle.com/datasets/dhoogla/unswnb15) | 175,341 / 82,332 | scaling experiments |
| [CIC-IDS-2018](https://www.kaggle.com/datasets/solarmainframe/ids-intrusion-csv) | ~16M | cross-dataset transfer |

Gitignored. See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) §3 for
download instructions and the data-quality defects in CIC-IDS-2018.

## Documentation

- [Reproducibility guide](docs/REPRODUCIBILITY.md) — environment, data, runtime measurements, results, limitations
- [Contribution statement](docs/CONTRIBUTIONS.md) — what is ours versus prior work
- [Future work](docs/FUTURE_WORK.md) — what to do next, ordered by value
- [Design notes](docs/feature_design/design-setup.md) — architecture decisions and planning corrections

## Environment

Python 3.11, PyTorch 2.13 with MPS, tabpfn 8.5.0. CPU or Apple Silicon only —
no CUDA required. Exact pins in `requirements.txt`.

## Reference

Hollmann, N., Müller, S., Purucker, L., Krishnakumar, A., Körfer, M., Hoo, S. B.,
Schirrmeister, R. T., & Hutter, F. (2025). Accurate predictions on small data
with a tabular foundation model. *Nature, 637*(8045), 319–326.
https://doi.org/10.1038/s41586-024-08328-6

## Team

Sujeeth G (2320090080) · P. Mahitha (2320090056)
Supervisor: P. Krishnanjaneyulu · KL University, Dept. of CSE

## License

MIT — see [LICENSE](LICENSE).
