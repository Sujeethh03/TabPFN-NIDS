# Reproducibility Guide

**Project:** Scaling Tabular Foundation Models for Network Intrusion Detection
**Repository:** https://github.com/Sujeethh03/TabPFN-NIDS
**Base paper:** Hollmann et al., *Accurate predictions on small data with a tabular foundation model*, Nature 637(8045):319–326, 2025

This document is written so that someone with the repository, a Kaggle account
and an Apple Silicon Mac can reproduce every number reported here, and can tell
exactly which numbers are solid and which are constrained by the hardware.

---

## 1. What this project does

TabPFN v2 is a transformer pretrained once on synthetic tabular tasks. It makes
predictions on a new table by taking labelled rows as *in-context examples* at
inference time — there is no per-dataset training. Its published limits are
10,000 in-context samples, 500 features and 10 classes.

This project applies it to network intrusion detection and extends it in three
directions:

| | Enhancement | Status |
|---|---|---|
| 1 | Stratified chunked ensemble to exceed the 10,000-sample context limit | implemented, partially measured |
| 2 | Domain-aware feature engineering | implemented, ablation constrained by memory |
| 3 | Cross-dataset transfer (UNSW-NB15 → CIC-IDS-2018) | not implemented |

A note on terminology, because it matters for how the work is described. TabPFN
receives labelled examples in its context window, so this is **in-context
learning, not zero-shot**. The phrase "zero-shot" is accurate only for
Enhancement 3, where no target-domain labels are used.

---

## 2. Environment

### 2.1 Hardware this was run on

| | |
|---|---|
| Machine | MacBook Air M1, 16 GB unified memory |
| OS | macOS 26.6.2 (arm64) |
| Backend | MPS (Metal). No CUDA anywhere in this project. |
| Disk used | ~6.5 GB of datasets, ~1.2 GB of virtualenv |

### 2.2 Setup

```bash
git clone https://github.com/Sujeethh03/TabPFN-NIDS.git
cd TabPFN-NIDS

python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

python smoke_test.py
```

`pip install -e .` is required, not optional: the package lives at
`src/tabpfn_nids/` and the notebooks and tests import it by name. Without the
editable install they fail with `ModuleNotFoundError`.

Expected smoke-test output:

```
  checkpoint     tabpfn-v2-classifier.ckpt
  device         mps
  Accuracy       0.9649
  Runtime        7.42s
Smoke test PASSED. Environment is ready.
```

Exit status is 0 on pass and 1 on failure, so it can gate CI.

### 2.3 Pinned versions

`requirements.txt` holds an exact `pip freeze` of 149 packages. The ones that
matter:

| package | version |
|---|---|
| python | 3.11.9 (arm64) |
| tabpfn | 8.5.0 |
| torch | 2.13.0 |
| scikit-learn | 1.9.0 |
| pandas | 3.0.5 |
| numpy | 2.4.6 |
| pyarrow | 25.0.1 |

### 2.4 The checkpoint pin — read this before changing anything

`tabpfn` 8.5.0 defaults to `model_path="auto"`, which resolves to the **gated**
HuggingFace repository `Prior-Labs/tabpfn_3`. On a clean machine that fails:

```
TabPFNHuggingFaceGatedRepoError: This model is gated and requires you to
accept its terms.
```

This project pins `tabpfn-v2-classifier.ckpt` from the ungated
`Prior-Labs/TabPFN-v2-clf` instead, in `tabpfn_nids/config.py`:

```python
TABPFN_CHECKPOINT: str = "tabpfn-v2-classifier.ckpt"
```

There are two reasons, and the second is the important one. It removes the
authentication requirement, and — more significantly — `tabpfn_3` is a
**different, newer model** than the one published in Nature 2025. Loading it
would invalidate the claim that this work reproduces and extends TabPFN v2.
`tests/test_config.py` asserts the constant is never `"auto"` again.

---

## 3. Datasets

All three are free on Kaggle and are gitignored. `data/` keeps its directory
tree in a fresh clone via `.gitkeep` sentinels.

```bash
mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

kaggle datasets download -d hassan06/nslkdd            -p data/raw/nsl-kdd      --unzip
kaggle datasets download -d dhoogla/unswnb15           -p data/raw/unsw-nb15    --unzip
kaggle datasets download -d solarmainframe/ids-intrusion-csv -p data/raw/cic-ids-2018 --unzip
```

### 3.1 NSL-KDD — the main benchmark

| | |
|---|---|
| Files | `KDDTrain+.txt`, `KDDTest+.txt` |
| Rows | 125,973 train / 22,544 test |
| Columns | 43 = 41 features + attack label + difficulty |
| After preprocessing | 122 features |

The `.txt` files ship without a header. The 41 feature names are taken verbatim
from the `@attribute` declarations in `KDDTrain+.arff`, which is distributed in
the same archive — they are read from the dataset, not reproduced from memory.

One discrepancy that a guess would get wrong: the `.arff` declares **42**
attributes, ending in a binary `class {normal, anomaly}`, while the `.txt` files
carry **43** fields. Field 42 of the `.txt` is the *specific attack name*
(`normal`, `neptune`, `satan`, … 23 values in train), and field 43 is a
difficulty level (integer 0–21) absent from the `.arff` entirely.

**`difficulty` is metadata, not a feature.** It is dropped before encoding.
Feeding it to a model would leak.

Class balance after binarisation (0 = normal, 1 = attack):

| split | rows | normal | attack |
|---|---|---|---|
| train | 125,973 | 67,343 (53.46%) | 58,630 (46.54%) |
| test | 22,544 | 9,711 (43.08%) | 12,833 (56.92%) |

The distribution shift is intentional in NSL-KDD. **Do not read a train/test
gap as a bug.**

The benchmark's defining property is that the test split contains **17 attack
types absent from training** (`apache2`, `httptunnel`, `mailbomb`, `mscan`,
`named`, `processtable`, `ps`, `saint`, `sendmail`, `snmpgetattack`,
`snmpguess`, `sqlattack`, `udpstorm`, `worm`, `xlock`, `xsnoop`, `xterm`) —
23 attack labels in train, 38 in test.

A correction to a common assumption, verified here: the test split contains
**no unseen categorical values**. `protocol_type`, `service` and `flag` in test
are a strict *subset* of training's; six services appear only in training. The
`handle_unknown="ignore"` setting is retained as a guard but never fires on this
release. Claiming otherwise in a write-up would be an easy thing to disprove.

### 3.2 UNSW-NB15

Ships as **parquet**, not CSV, so `pyarrow` is required.

| | claimed in early planning | actual |
|---|---|---|
| Rows | ~2.5M | **175,341 train / 82,332 test** |
| Features | 49 | **36 columns** |
| Classes | 9 attacks | **10 including Normal** |

The `dhoogla/unswnb15` release is the cleaned standard partition, not the full
2.54M-record capture. Its `attack_cat` has exactly 10 classes, sitting precisely
on TabPFN's `MAX_NUMBER_OF_CLASSES` with no headroom.

### 3.3 CIC-IDS-2018

10 daily CSVs, 6.4 GB, ~80 columns. Four data-quality problems that will break a
naïve `pd.concat`:

1. **`02-20-2018.csv` has 84 columns, not 80** — four extra: `Flow ID`,
   `Src IP`, `Src Port`, `Dst IP`. It is also the 3.8 GB file, over half the
   dataset.
2. **Repeated header rows embedded mid-file**: 33 in `02-28-2018.csv`, 25 in
   `03-01-2018.csv`, 1 in `02-16-2018.csv`. Read naïvely these become string
   rows that coerce whole numeric columns to `object`.
3. **`02-20-2018.csv` uses CRLF line endings** — labels are `Benign\r`. Any
   label mapping keyed on clean strings silently misses every row in the
   largest file.
4. **Infinity values are present.** In the first 200,000 rows of
   `02-14-2018.csv` alone: 5 `Inf` and 5 `NaN` cells, all in `Flow Pkts/s`.
   TabPFN's `PASSTHROUGH_INF` is `False`, so infinities are *rejected at input
   validation* and must be handled explicitly before inference.

---

## 4. Running the experiments

```bash
python scripts/run_baseline.py --seed 42 --test-size 5000 --n-estimators 2
python scripts/run_enhanced.py --seed 42 --test-size 5000 --n-estimators 2
python scripts/run_feature_ablation.py --seed 42 --max-chunks 3 --test-size 1000 \
       --n-estimators 2 --max-service-flag-categories 40
python scripts/summarize_runs.py --tag baseline
python scripts/build_comparison_table.py
```

Every run writes a timestamped CSV to `reports/` carrying seed, hardware,
library versions, the checkpoint filename, git commit and runtime alongside the
metrics.

**The three settings that must match between arms** for any comparison to be
valid: `--n-estimators`, `--test-size` and the seed. If they differ, the measured
delta reflects those knobs rather than the enhancement.

---

## 5. Runtime — measure before you plan

This is the section that most changes what is feasible.

TabPFN's `fit` merely caches the context and takes under a second. **Essentially
all compute is in `predict`.** Measured on the M1 Air, synthetic data, 2,000 test
rows:

| context rows | features | device | accuracy | predict |
|---|---|---|---|---|
| 1,000 | 40 | MPS | 0.908 | 22.7 s |
| 1,000 | 40 | CPU | 0.907 | 86.0 s |
| 5,000 | 40 | MPS | 0.969 | 97.8 s |
| 10,000 | 40 | MPS | 0.961 | 279.9 s |
| 10,000 | 40 | CPU | 0.962 | 1,493.1 s |
| 10,000 | 80 | MPS | 0.953 | 544.4 s |

Four conclusions:

1. **MPS is 3.8× faster than CPU at 1,000 rows and 5.3× at 10,000.** The gap
   widens with size. `tabpfn` also refuses CPU inference above 1,000 samples by
   default (`MAX_CPU_SAMPLES`).
2. **Cost grows super-linearly in context size.** 10× the context costs 12.3×
   the time.
3. **Doubling features nearly doubles predict time** (280 s → 544 s).
4. **Accuracy saturates well before the limit.** 5,000 context rows scored 0.969
   against 10,000 rows' 0.961 — within noise, at a third of the cost. Do not
   default `chunk_size` to 10,000 without measuring.

### 5.1 Consequences

The full-scale experiment described in the original plan is **not runnable on
this hardware**:

| experiment | measured or projected |
|---|---|
| Baseline, 3 seeds, full 22,544-row test | 17.5 hours |
| Baseline, 3 seeds, 5,000-row test, `n_estimators=2` | ~15 min |
| Chunked ensemble, 13 chunks, 5,000-row test | ~37 min |
| Chunked ensemble, 13 chunks, full test | ~2.8 hours |

`n_estimators="auto"` resolved to **8** for 122 features, which is where the
17.5-hour figure comes from. Fixing it to 2 is the single largest lever. A sweep
at 500 test rows found `n=1` scored *higher* than `n=8` (F1 0.8016 vs 0.7727)
while being 4× faster — within sampling noise, but clear evidence that the
auto-scaled ensemble buys nothing measurable here.

### 5.2 MPS memory — two distinct failures

**Failure 1: accumulation across chunks.** The chunked ensemble died at chunk 4
of 13 with `TabPFNMPSOutOfMemoryError`. MPS does not release memory when a Python
reference is dropped, so allocations accumulate. Fixed by
`TabPFNWrapper.free()`, which calls `torch.mps.empty_cache()` between chunks;
verified over 6 sequential chunks.

**Failure 2: context too large.** With engineered features on, the feature count
is 463 and a 9,691 × 463 context exhausts MPS memory regardless of test batch
size — it OOMs at batch 1,000, 250 and 100. The memory goes to the *context*,
not the test rows, so batching cannot fix it. The workaround is
`max_service_flag_categories`, which caps the interaction feature's cardinality.

---

## 6. Results

### 6.1 Baseline — vanilla TabPFN on NSL-KDD

10,000-row stratified context, 5,000-row stratified test sample,
`n_estimators=2`, MPS.

| seed | accuracy | precision | recall | f1_score | roc_auc |
|---|---|---|---|---|---|
| 42 | 0.7452 | 0.9159 | 0.6082 | 0.7310 | 0.9515 |
| 123 | 0.7632 | 0.9227 | 0.6374 | 0.7539 | 0.9555 |
| 2024 | 0.7874 | 0.9655 | 0.6497 | 0.7767 | 0.9595 |
| **mean ± std** | 0.7653 ± 0.0212 | 0.9347 ± 0.0269 | 0.6318 ± 0.0213 | **0.7539 ± 0.0229** | **0.9555 ± 0.0040** |

**The F1 standard deviation of 0.0229 is the noise floor.** Any enhancement
delta smaller than roughly 2.3 percentage points, measured on a single seed, is
not evidence of an effect.

Two findings worth more than the headline number:

**The error is almost entirely recall.** Precision 0.93 against recall 0.63 —
the model is right when it flags an attack but misses about 37% of them. This is
the expected consequence of the 17 unseen attack types, not a defect, and it is
the gap Enhancement 1 should narrow.

**ROC-AUC 0.9555 with std 0.0040.** The model *ranks* attacks well; the default
0.5 decision threshold simply places the operating point poorly for this
distribution shift. Threshold tuning would likely lift F1 substantially with no
model change. This is the most actionable result in the project.

### 6.2 What is not yet measured

- **Enhancement 1 end-to-end**: the 13-chunk run crashed on the MPS memory bug
  before completing. The fix is in and verified on a smaller configuration, but
  the full run has not been repeated.
- **Enhancement 2 ablation**: runnable only at reduced scope (3 of 13 chunks,
  1,000 test rows, 40 capped categories).
- **Enhancement 3**: not implemented.
- **Classical baselines** (XGBoost, LightGBM): not implemented.

---

## 7. Honest limitations

1. **Test set is subsampled.** Headline numbers use a stratified 5,000-row
   sample of the 22,544-row test split. Not directly comparable to published
   full-test-set results.
2. **`n_estimators=2`, not TabPFN's auto-scaled 8.** Chosen for runtime. It must
   be held identical across arms or comparisons measure ensemble size instead.
3. **Three seeds cannot support a significance test.** Wilcoxon's smallest
   attainable two-sided p is 0.25 at n=3. `significance.py` refuses to return a
   p-value it could never have rejected. Report mean ± std.
4. **The Enhancement 2 category cap changes what is being measured.** At 40
   categories `common_service_flag` covers ~88% of training rows instead of
   100%. This is a hardware constraint, not a design choice, and should be
   presented as such.
5. **Binary classification only.** The 5-class taxonomy (normal/dos/probe/r2l/u2r)
   is not implemented. NSL-KDD's U2R class is ~0.04% of training rows (about 52
   of 125,973), so macro-F1 over five classes would be dominated by a class with
   almost no support.
6. **Single hardware configuration.** Every timing is from one 16 GB M1 Air.

---

## 8. Test suite

```bash
pytest                 # fast suite
pytest -m slow         # real TabPFN inference, deselected by default
```

152 tests. Those needing the datasets skip automatically when `data/raw/` is
empty, so a fresh clone passes without any download.

| file | covers |
|---|---|
| `test_structure.py` | package layout, imports from any working directory, no absolute paths |
| `test_config.py` | checkpoint pin, limits mirrored from the installed library, seeding |
| `test_loader.py` | column names, row counts, error handling |
| `test_preprocessor.py` | encoding, scaling, leakage, binarisation |
| `test_features.py` | engineered features, finiteness, edge cases, toggle |
| `test_chunker.py` | chunk counts, disjointness, stratification within ±2% |
| `test_ensemble.py` | aggregation maths, validity of probabilities |
| `test_metrics.py` | metric correctness, degenerate inputs |
| `test_evaluation.py` | significance floors, reporting round-trip, figures |
| `test_tabpfn_wrapper.py` | context-limit guard, device selection |

---

## 9. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `TabPFNHuggingFaceGatedRepoError` | `TABPFN_CHECKPOINT` was changed to `"auto"`. Restore `"tabpfn-v2-classifier.ckpt"`. |
| `ModuleNotFoundError: tabpfn_nids` | `pip install -e .` was not run. |
| `TabPFNMPSOutOfMemoryError` mid-ensemble | Lower `--predict-batch-size`; ensure `TabPFNWrapper.free()` is called between chunks. |
| Same error with engineered features on | The context is too wide. Set `--max-service-flag-categories 40`. |
| "Running on CPU with more than 200 samples may be slow" | MPS unavailable. Check `torch.backends.mps.is_available()`. |
| Run takes hours | `n_estimators` is `"auto"` (resolves to 8). Pass `--n-estimators 2` and `--test-size`. |
| `FileNotFoundError` on NSL-KDD | Datasets not downloaded; see section 3. |
| Parquet read fails | `pip install pyarrow`. |

---

## 10. Provenance

Every results CSV records: timestamp, experiment, dataset, seed, hardware
string, device, tabpfn version, checkpoint filename, torch/sklearn/pandas/python
versions, git commit, context rows, test rows, feature count, `n_estimators`,
runtime split into fit and predict, all five metrics, the four confusion-matrix
cells and the matrix as JSON.

That is enough for a reader to tell whether two rows are comparable without
having to trust the surrounding prose.
