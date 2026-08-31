# TabPFN-NIDS

**Scaling Tabular Foundation Models for Network Intrusion Detection**

Applying TabPFN v2 (Hollmann et al., *Nature* 2025) to network intrusion
detection, and extending it past its 10,000-sample context limit with a
stratified chunked ensemble.

> **On terminology:** TabPFN takes labelled rows as in-context examples at
> inference time, so this is *in-context learning*, not zero-shot. The term
> "zero-shot" is used only for the cross-dataset transfer experiment, where no
> target-domain labels are involved.

---

## Quick start

```bash
git clone https://github.com/Sujeethh03/TabPFN-NIDS.git
cd TabPFN-NIDS

python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .          # required: notebooks and tests import the package by name

python smoke_test.py      # verifies the environment; exits 1 on failure
```

Then download NSL-KDD and run the baseline:

```bash
kaggle datasets download -d hassan06/nslkdd -p data/raw/nsl-kdd --unzip
python scripts/run_baseline.py --seed 42 --test-size 5000 --n-estimators 2
```

Takes about 3 minutes on an M1 and writes a timestamped CSV to `reports/`.

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
├── tests/                    # 152 tests
├── data/raw/                 # gitignored; .gitkeep keeps the tree
├── reports/                  # results CSVs, figures, tables
└── docs/                     # REPRODUCIBILITY, CONTRIBUTIONS, FUTURE_WORK
```

## Reproduction commands

```bash
python scripts/run_baseline.py --seed 42 --test-size 5000 --n-estimators 2
python scripts/run_enhanced.py --seed 42 --test-size 5000 --n-estimators 2
python scripts/run_feature_ablation.py --seed 42 --max-chunks 3 --test-size 1000 \
       --n-estimators 2 --max-service-flag-categories 40
python scripts/summarize_runs.py --tag baseline
python scripts/build_comparison_table.py

pytest                # fast suite
pytest -m slow        # includes real TabPFN inference
```

`--n-estimators`, `--test-size` and the seed **must match across arms** for any
comparison to be valid.

---

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
