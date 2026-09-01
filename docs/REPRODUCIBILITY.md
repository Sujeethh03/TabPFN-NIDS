# Reproducibility Guide

**Project:** Scaling Tabular Foundation Models for Network Intrusion Detection
**Repository:** <https://github.com/Sujeethh03/TabPFN-NIDS>
**Base paper:** Hollmann et al., *Accurate predictions on small data with a tabular foundation model*, Nature 637(8045):319–326, 2025
**Authors:** Sujeeth G (2320090080), P. Mahitha (2320090056) — KL University, Dept. of CSE
**Supervisor:** P. Krishnanjaneyulu

This document is written so that a reader with the repository, a Kaggle account
and an Apple Silicon Mac can reproduce every number reported in this project,
and can tell precisely which numbers are solid, which are constrained by the
hardware, and which have not been produced at all.

Every figure quoted here comes from a CSV in `reports/` or from a command run on
the reference machine. Where a result does not exist, this guide says so rather
than leaving a gap the reader has to discover.

---

## 1. Introduction

### 1.1 What this project does, in plain language

Network intrusion detection is a tabular classification problem. A sensor
records one row per network connection — duration, protocol, bytes sent and
received, error rates, flag states — and something has to decide whether that
row is ordinary traffic or an attack. The conventional approach trains a model
(a random forest, a gradient-boosted tree, a small neural network) on labelled
historical traffic, and retrains it whenever the traffic or the attacks change.

This project asks whether a **tabular foundation model** can do that job without
being trained at all.

### 1.2 What TabPFN v2 is, and why we use it

TabPFN v2 is a transformer that was pretrained once, by its authors, on millions
of *synthetic* tabular prediction tasks. It never sees your dataset during
training. Instead, at inference time you hand it labelled rows directly in its
context window — the same way you would paste examples into a language-model
prompt — and it predicts labels for new rows in a single forward pass.

Three consequences matter for intrusion detection:

1. **There is no training step.** In this codebase `fit()` completes in about
   0.3 seconds, because all it does is cache the context. Every bit of the
   compute is in `predict()`. This inverts the usual cost profile and is the
   single most important thing to understand before planning any experiment.
2. **Adapting to a new attack family means swapping the context, not
   retraining.** That is operationally attractive in a domain where novel
   attacks are the whole problem.
3. **It has hard published limits: 10,000 in-context samples, 500 features, 10
   classes.** The sample limit is architectural, not a tuning knob, and it is
   the constraint this entire project is built around.

That last point deserves emphasis. NSL-KDD's training split has 125,973 rows. A
single TabPFN context can hold 10,000 of them. **The vanilla baseline therefore
discards 92% of the available training data before it starts.**

A note on terminology, because it affects how the work should be described.
TabPFN receives labelled examples in its context window, so this is **in-context
learning, not zero-shot learning**. The term "zero-shot" is accurate only for
Enhancement 3, where no target-domain labels are involved. Describing the main
experiments as zero-shot would be wrong and is easy to disprove.

### 1.3 The three enhancements

| | Enhancement | Idea | Status |
|---|---|---|---|
| 1 | **Stratified chunked ensemble** | Partition training data into chunks that each fit one context; predict with every chunk; aggregate by confidence-weighted voting | Implemented and tested; **full-scale run not completed** |
| 2 | **Domain-aware feature engineering** | Add six features a security analyst reads directly (byte ratios, throughput, error-rate composites, protocol/service/flag interaction) | Implemented and tested; ablation results **conflict across scales** |
| 3 | **Cross-dataset transfer** | Train on one dataset, evaluate on another with no target labels | **Not implemented** |

Section 10 states plainly what each enhancement has and has not demonstrated.
Readers who want that conclusion first should skip there.

---

## 2. Hardware and Software Requirements

### 2.1 The reference machine

Every timing in this document was measured on one machine. Timings elsewhere
will differ, sometimes by a factor of four (see §10.3).

| | |
|---|---|
| Machine | MacBook Air M1, 16 GB unified memory, 8 cores |
| OS | macOS 26.6.2 (arm64) |
| Backend | MPS (Metal Performance Shaders). No CUDA anywhere in this project. |
| Disk | ~6.5 GB datasets + ~1.2 GB virtualenv |

### 2.2 Minimum requirements

| Component | Minimum | Notes |
|---|---|---|
| Python | **3.11** | Enforced by `requires-python = ">=3.11"`. Not tested on 3.12+. |
| RAM | 16 GB | 8 GB will OOM on the chunked ensemble. MPS shares system memory. |
| Disk | 2 GB | NSL-KDD only. Add 6.4 GB for CIC-IDS-2018. |
| Accelerator | Apple Silicon (MPS) strongly preferred | See below. |

### 2.3 Platform compatibility

**macOS on Apple Silicon** is the tested configuration and the one all results
come from.

**CPU-only** works but is impractical. Measured on synthetic data at 2,000 test
rows, CPU was **3.8× slower at a 1,000-row context and 5.3× slower at 10,000**
(1,493s vs 280s). Separately, `tabpfn` refuses CPU inference above 1,000 samples
by default via its `MAX_CPU_SAMPLES` guard, so CPU runs require deliberately
overriding a safety limit.

**Linux with CUDA** should work — nothing in this codebase is macOS-specific and
`config.resolve_device()` handles backend selection — but **it has never been
run here**, so no CUDA timing or result is claimed. A reader reproducing on CUDA
should expect different runtimes and should re-measure §10.3's tolerance bands
rather than assuming them.

**Windows** is untested.

---

## 3. Environment Setup

Expected time: **5–10 minutes**, plus 2–5 minutes on first run while TabPFN
downloads its checkpoint (~100 MB).

### 3.1 Step by step

```bash
git clone https://github.com/Sujeethh03/TabPFN-NIDS.git
cd TabPFN-NIDS

python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### 3.2 Why `pip install -e .` is mandatory

This is a **src-layout** project. The importable package is `tabpfn_nids` and it
lives at `src/tabpfn_nids/`. There is no top-level `src` package. Without the
editable install, every notebook, script and test fails with
`ModuleNotFoundError`.

```python
from tabpfn_nids.data_pipeline import load_nsl_kdd     # correct
from src.data_pipeline.loader   import load_nsl_kdd    # wrong — no `src` package exists
```

If you hit an import error, the fix is `pip install -e .`. **Do not set
`PYTHONPATH=.`** as a workaround. Putting the repository root on the path makes
`import src.*` appear to work, which masks the real problem and then breaks the
moment anything runs from another working directory. `pyproject.toml` already
declares the correct layout:

```toml
[tool.setuptools.packages.find]
where = ["src"]
```

### 3.3 Verify the installation

```bash
python smoke_test.py
```

This runs TabPFN end-to-end on a small scikit-learn dataset. It exits 0 on pass
and 1 on failure, so it can gate CI. Actual output from the reference machine:

```
TabPFN-NIDS environment smoke test
----------------------------------------------------
  python         3.11.9 (arm64)
  torch          2.13.0
  tabpfn         8.5.0
  scikit-learn   1.9.0
  checkpoint     tabpfn-v2-classifier.ckpt
  device         mps
  seed           42
----------------------------------------------------
Loading breast-cancer dataset...
  train 398 rows, test 171 rows, 30 features
Running TabPFN on mps (first run downloads the checkpoint)...
----------------------------------------------------
  Accuracy       0.9649
  Runtime        9.67s
----------------------------------------------------

Smoke test PASSED. Environment is ready.
```

Wall-clock is **~14s** on a warm cache; the first run takes several minutes
while the checkpoint downloads. Check three lines specifically: `device` should
read `mps` (not `cpu`), `checkpoint` must read `tabpfn-v2-classifier.ckpt` (see
§3.5), and accuracy should be ≈0.96.

### 3.4 Pinned versions

`requirements.txt` is an exact `pip freeze` of 149 pinned packages. The ones that
determine results:

| package | version |
|---|---|
| python | 3.11.9 (arm64) |
| tabpfn | 8.5.0 |
| torch | 2.13.0 |
| scikit-learn | 1.9.0 |
| pandas | 3.0.5 |
| numpy | 2.4.6 |
| pyarrow | 25.0.1 |

### 3.5 The checkpoint pin — read this before changing anything

This is the subtlest reproducibility trap in the project.

`tabpfn` 8.5.0 defaults to `model_path="auto"`, which resolves to the **gated**
HuggingFace repository `Prior-Labs/tabpfn_3`. On a clean machine that fails
outright:

```
TabPFNHuggingFaceGatedRepoError: This model is gated and requires you to
accept its terms.
```

This project instead pins `tabpfn-v2-classifier.ckpt` from the ungated
`Prior-Labs/TabPFN-v2-clf`, in `src/tabpfn_nids/config.py`:

```python
TABPFN_CHECKPOINT: str = "tabpfn-v2-classifier.ckpt"
```

There are two reasons and the second is the important one. It removes the
authentication requirement — and, far more significantly, **`tabpfn_3` is a
different, newer model than the one published in Nature 2025.** Silently
benchmarking `tabpfn_3` while citing the Nature paper would invalidate the
central claim that this work reproduces and extends TabPFN v2. A test in
`tests/test_config.py` asserts the constant is never `"auto"` again.

---

## 4. Dataset Acquisition

All three datasets are free on Kaggle. All are gitignored, so **a fresh clone
contains no data**; `data/` keeps its directory tree via `.gitkeep` sentinels.

### 4.1 Download

```bash
mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

kaggle datasets download -d hassan06/nslkdd                   -p data/raw/nsl-kdd        --unzip
kaggle datasets download -d dhoogla/unswnb15                  -p data/raw/unsw-nb15      --unzip
kaggle datasets download -d solarmainframe/ids-intrusion-csv  -p data/raw/cic-ids-2018   --unzip
```

Downloading by hand works equally well — only the final file locations matter.

| dataset | Kaggle slug | destination | on-disk | needed for |
|---|---|---|---|---|
| NSL-KDD | `hassan06/nslkdd` | `data/raw/nsl-kdd/` | 107 MB | **every result in this repo** |
| UNSW-NB15 | `dhoogla/unswnb15` | `data/raw/unsw-nb15/` | 14 MB | scaling experiments (not yet run) |
| CIC-IDS-2018 | `solarmainframe/ids-intrusion-csv` | `data/raw/cic-ids-2018/` | 6.4 GB | cross-dataset transfer (not implemented) |

**NSL-KDD alone is sufficient to reproduce every number in this document.** The
other two are only needed for work that has not been completed.

Required files:

- `data/raw/nsl-kdd/` — `KDDTrain+.txt`, `KDDTest+.txt` (the `.arff` files ship
  in the same archive and are the source of the column names)
- `data/raw/unsw-nb15/` — `UNSW_NB15_training-set.parquet`,
  `UNSW_NB15_testing-set.parquet`
- `data/raw/cic-ids-2018/` — ten daily CSVs, `02-14-2018.csv` … `03-02-2018.csv`

### 4.2 Verify the download

> **Note.** Earlier drafts of this guide referred to a `scripts/verify_datasets.py`
> helper. **That script does not exist in the repository.** Use the commands
> below, which require nothing beyond the installed package.

NSL-KDD:

```bash
python -c "from tabpfn_nids.data_pipeline import load_nsl_kdd; \
           tr, te = load_nsl_kdd(); print(tr.shape, te.shape)"
```

```
(125973, 43) (22544, 43)
```

Anything else means the files are missing, truncated, or in the wrong
directory. UNSW-NB15:

```bash
python -c "import pandas as pd; \
           print(pd.read_parquet('data/raw/unsw-nb15/UNSW_NB15_training-set.parquet').shape)"
# (175341, 36)
```

The test suite also self-verifies: tests that need data **skip automatically**
when `data/raw/` is empty, so a fresh clone passes `pytest` with no download at
all.

### 4.3 NSL-KDD structure, and one discrepancy a guess would get wrong

| | |
|---|---|
| Files | `KDDTrain+.txt`, `KDDTest+.txt` |
| Rows | 125,973 train / 22,544 test |
| Columns | 43 = 41 features + attack label + difficulty |
| After preprocessing | 122 features (38 numeric + 84 one-hot levels) |

The `.txt` files ship without a header. The 41 feature names are read verbatim
from the `@attribute` declarations in `KDDTrain+.arff` — from the dataset, not
reproduced from memory.

The `.arff` declares **42** attributes, ending in a binary
`class {normal, anomaly}`, while the `.txt` files carry **43** fields. Field 42
of the `.txt` is the *specific attack name* (`normal`, `neptune`, `satan`, … 23
values in train) and field 43 is a difficulty level (integer 0–21) that the
`.arff` omits entirely.

**`difficulty` is metadata, not a feature.** It is dropped before encoding.
Feeding it to a model would leak.

Class balance after binarisation (0 = normal, 1 = attack):

| split | rows | normal | attack |
|---|---|---|---|
| train | 125,973 | 67,343 (53.46%) | 58,630 (46.54%) |
| test | 22,544 | 9,711 (43.08%) | 12,833 (56.92%) |

**The train/test distribution shift is intentional. Do not read it as a bug.**
The benchmark's defining property is that the test split contains **17 attack
types absent from training** — `apache2`, `httptunnel`, `mailbomb`, `mscan`,
`named`, `processtable`, `ps`, `saint`, `sendmail`, `snmpgetattack`,
`snmpguess`, `sqlattack`, `udpstorm`, `worm`, `xlock`, `xsnoop`, `xterm` — 23
attack labels in train against 38 in test, covering **3,750 test rows (16.6%)**.
That is what makes NSL-KDD a test of detecting novel attacks rather than of
memorisation, and it is the direct cause of the low recall reported in §6.

Re-splitting the union of the two files would destroy this property, so the
published split is preserved exactly.

One correction to a common assumption, verified here: the test split contains
**no unseen categorical values**. `protocol_type`, `service` and `flag` in test
are a strict *subset* of training's; six services appear only in training. The
`handle_unknown="ignore"` setting is kept as a guard but never fires on this
release.

### 4.4 CIC-IDS-2018 data-quality defects

Four problems will break a naïve `pd.concat`:

1. **`02-20-2018.csv` has 84 columns, not 80** — four extra (`Flow ID`,
   `Src IP`, `Src Port`, `Dst IP`). It is also the 3.8 GB file, over half the
   dataset.
2. **Repeated header rows embedded mid-file**: 33 in `02-28-2018.csv`, 25 in
   `03-01-2018.csv`, 1 in `02-16-2018.csv`. Read naïvely these become string
   rows that coerce entire numeric columns to `object`.
3. **`02-20-2018.csv` uses CRLF line endings** — labels arrive as `Benign\r`.
   Any label mapping keyed on clean strings silently misses every row in the
   largest file.
4. **Infinity values are present.** In the first 200,000 rows of
   `02-14-2018.csv` alone: 5 `Inf` and 5 `NaN` cells, all in `Flow Pkts/s`.
   TabPFN's `PASSTHROUGH_INF` is `False`, so infinities are rejected at input
   validation and must be handled explicitly.

---

## 5. Code Structure

### 5.1 Folder tree

```
tabpfn-nids/
├── src/tabpfn_nids/          the installed package
│   ├── config.py             paths, seed, TabPFN limits, device resolution,
│   │                         provenance capture
│   ├── data_pipeline/
│   │   ├── loader.py         reads raw files into DataFrames; nothing else
│   │   └── preprocessor.py   encoding, scaling, binarisation; fitted on train only
│   ├── features/
│   │   └── engineered.py     Enhancement 2 — six domain features behind one flag
│   ├── models/
│   │   ├── tabpfn_wrapper.py TabPFN + context guard, device handling, free()
│   │   ├── chunker.py        stratified and random partitioning
│   │   └── chunked_ensemble.py  Enhancement 1 — chunk, predict, aggregate
│   ├── evaluation/
│   │   ├── metrics.py        the five metrics + confusion matrix
│   │   ├── significance.py   Wilcoxon, and the floor below which it refuses
│   │   ├── reporter.py       results CSV read/write with provenance
│   │   └── plots.py          confusion matrix, ROC, PR, comparison bars
│   └── utils/
├── scripts/                  thin CLI wrappers; no logic lives here
├── notebooks/                01 reproduction, 02 enhancement,
│                             03 ablations, 04 results summary
├── tests/                    163 tests (+4 slow, deselected by default)
├── data/raw/                 gitignored; .gitkeep preserves the tree
├── reports/                  results CSVs, figures, tables
└── docs/                     this guide, CONTRIBUTIONS, FUTURE_WORK
```

The separation is deliberate: **no pipeline logic lives in `scripts/` or
`notebooks/`.** Both call the same tested package code, so what a reviewer reads
in a notebook is what produced the CSVs.

### 5.2 Which script runs which experiment

| Experiment | Script | Section |
|---|---|---|
| Environment check | `smoke_test.py` | §3.3 |
| Baseline — vanilla TabPFN | `scripts/run_baseline.py` | §6 |
| Enhancement 1 — chunked ensemble | `scripts/run_enhanced.py` | §7 |
| Enhancement 2 — feature ablation | `scripts/run_feature_ablation.py` | §8 |
| Context-size + stratification ablations | `scripts/run_ablation.py` | §9 |
| Per-tag summary | `scripts/summarize_runs.py` | §10 |
| Headline three-arm table → `reports/comparison_table.md` | `scripts/generate_comparison_table.py` | §10 |
| Per-experiment table, mean ± std → `reports/tables/comparison_table.md` | `scripts/build_comparison_table.py` | §10 |

### 5.3 Where results are saved

Every run appends a timestamped CSV to `reports/`, named
`<tag>_<timestamp>_seed<N>.csv`:

```
reports/
├── baseline_20260830_114259_seed42.csv
├── baseline_20260830_114555_seed123.csv
├── baseline_20260830_114850_seed2024.csv
├── feature_ablation_20260831_141146_seed42.csv
├── ablation_20260831_153107_seed42.csv
├── comparison_table.{csv,md}
└── figures/*.png
```

Each row carries its own provenance: timestamp, experiment, dataset, seed,
hardware string, device, tabpfn version, checkpoint filename, torch / sklearn /
python versions, git commit, context rows, test rows, feature count,
`n_estimators`, runtime split into fit and predict, all five metrics, the four
confusion-matrix cells, and the matrix as JSON. That is enough for a reader to
decide whether two rows are comparable **without trusting the surrounding
prose**.

Note that `.gitignore` contains a blanket `*.csv` rule, so results files are not
tracked by default.

---

## 6. Reproducing the Baseline

### 6.1 Command

```bash
python scripts/run_baseline.py --seed 42  --test-size 5000 --n-estimators 2
python scripts/run_baseline.py --seed 123 --test-size 5000 --n-estimators 2
python scripts/run_baseline.py --seed 2024 --test-size 5000 --n-estimators 2
```

> **`--test-size` and `--n-estimators` are not optional in practice.**
> `run_baseline.py --seed 42` alone defaults to the full 22,544-row test split
> and `n_estimators="auto"` (which resolves to **8** at 122 features). That
> combination is a multi-hour job. The three commands above are what produced
> the recorded results.

### 6.2 Expected runtime

| configuration | runtime |
|---|---|
| `--test-size 5000 --n-estimators 2` (recorded) | 172–542 s per seed |
| `--context-size 5000 --test-size 500 --n-estimators 2` (quick check) | ~75 s |
| `--seed 42` with defaults | **hours — do not run casually** |

The wide 172–542 s spread across three identically-configured seeds is real and
is machine thermal state, not variance in the method. Treat runtime as
indicative, never as a result.

### 6.3 Expected metrics

Recorded results, 10,000-row stratified context, 5,000-row stratified test
sample, `n_estimators=2`, MPS:

| seed | accuracy | precision | recall | f1_score | roc_auc |
|---|---|---|---|---|---|
| 42 | 0.7452 | 0.9159 | 0.6082 | 0.7310 | 0.9515 |
| 123 | 0.7632 | 0.9227 | 0.6374 | 0.7539 | 0.9555 |
| 2024 | 0.7874 | 0.9655 | 0.6497 | 0.7767 | 0.9595 |
| **mean ± std** | 0.7653 ± 0.0212 | 0.9347 ± 0.0269 | 0.6318 ± 0.0213 | **0.7539 ± 0.0229** | **0.9555 ± 0.0040** |

Acceptance bands (±5% relative, per §10.2):

| metric | expected | acceptable range |
|---|---|---|
| accuracy | 0.7653 | 0.727 – 0.804 |
| precision | 0.9347 | 0.888 – 0.982 |
| recall | 0.6318 | 0.600 – 0.663 |
| **f1_score** | **0.7539** | **0.716 – 0.792** |
| **roc_auc** | **0.9555** | **0.908 – 1.000** |

### 6.4 How to know it succeeded

1. A new `reports/baseline_<timestamp>_seed42.csv` exists with one data row.
2. F1 lands in 0.716–0.792.
3. **Precision substantially exceeds recall** (≈0.93 vs ≈0.63). If they are
   close, or recall exceeds precision, something is structurally wrong —
   suspect a label inversion or a preprocessing mismatch, not bad luck.
4. ROC-AUC ≥ 0.90.
5. The CSV's `checkpoint` column reads `tabpfn-v2-classifier.ckpt`.

Sample confusion matrix (seed 42, 5,000 test rows):

```
                        predicted
                     normal    attack
    actual normal      1,995       159
    actual attack      1,115     1,731
```

---

## 7. Reproducing Enhancement 1: Chunked Ensemble

### 7.1 What it does

TabPFN caps its context at 10,000 rows. The chunked ensemble partitions the
training set into stratified chunks that each fit one context, runs TabPFN once
per chunk over the whole test set, and aggregates the per-chunk probabilities by
confidence-weighted voting. `fit()` performs no training — it only partitions
and stores.

### 7.2 Command

```bash
python scripts/run_enhanced.py --seed 42 --test-size 5000 --n-estimators 2
```

`--n-estimators` **must** match the baseline's value, or the comparison measures
TabPFN ensemble size rather than chunking.

### 7.3 Status: this run has not been completed

> **There are no `enhanced_*.csv` files in `reports/`.** The full-scale
> 13-chunk run crashed on the MPS memory bug described in §11.3 before it
> finished. That bug is fixed and verified, but the run has not been repeated.
> **This is the largest open gap in the project**, because Enhancement 1 is its
> central claim.

`reports/comparison_table.md` prints this arm as *not run* rather than omitting
it.

### 7.4 What has been measured — a bounded run

`notebooks/02_enhancement.ipynb` runs a deliberately bounded version that
completes in **261 s**: 8 chunks × 2,470 rows = 19,763 training rows, scored on
a fixed 400-row test subset, everything else held equal.

| arm | training rows | features | F1 | ROC-AUC | runtime |
|---|---|---|---|---|---|
| A. single context | 2,500 | 122 | 0.7835 | 0.9647 | 14 s |
| B. chunked ensemble | 19,763 (8 chunks) | 122 | 0.7624 | 0.9653 | 103 s |
| C. ensemble + engineered | 19,763 | 168 | 0.7980 | 0.9598 | 142 s |

Chunk diagnostics from that run:

```
  n_chunks                 8
  min_chunk_size           2,470
  max_chunk_size           2,471
  total_rows               19,763
  mean_positive_rate       0.465263
  max_positive_rate_drift  0.000216
  population attack rate   0.465417
```

### 7.5 How to interpret

**The stratification machinery works exactly as specified.** Chunk sizes differ
by one row and the attack rate drifts by 0.0002 against a population rate of
0.4654. That is a mechanical guarantee, and it holds.

**Chunking did not improve accuracy here.** Arm B used 8× the training data of
arm A and scored **2.11 pp lower F1**. That delta is *inside* the 2.29 pp noise
floor (§10.1), so the correct statement is "no measurable effect", not "chunking
hurts". It is nonetheless the second independent signal in this direction — the
context-size ablation in §9 points the same way.

**The honest reading of Enhancement 1 today:** it removes a hard architectural
ceiling *by construction*, which is a real contribution and is verifiable from
the chunk diagnostics above. It has **not** been shown to raise accuracy on
NSL-KDD. That is unsurprising: NSL-KDD is small enough that 10,000 rows already
captures most of the available signal. The enhancement's real case is
UNSW-NB15 (175,341 rows) and CIC-IDS-2018 (~16M), where a single context is a
far more severe constraint — and neither has been run.

---

## 8. Reproducing Enhancement 2: Feature Engineering

### 8.1 The six features

Vanilla TabPFN treats every column as an unrelated variable. Network flow
records are not unrelated.

| feature | definition | rationale |
|---|---|---|
| `bytes_ratio` | `src_bytes / (dst_bytes + 1)` | upload/download asymmetry; exfiltration skews high |
| `total_bytes` | `src_bytes + dst_bytes` | overall flow volume |
| `bytes_per_second` | `total_bytes / (duration + 1)` | throughput; volumetric floods sit at the extreme |
| `is_short_session` | `1` when `duration < 5s` | scans and probes are short |
| `error_rate_composite` | `serror_rate × srv_serror_rate` | high only when host- *and* service-level SYN errors coincide — a SYN flood, not one broken service |
| `common_service_flag` | `protocol_type + service + flag` | interactions a per-column encoding cannot capture: `tcp/private/S0` (half-open scan) vs `tcp/http/SF` (ordinary web request) |

Every denominator has `+1` added, so division by zero is impossible and output
is finite by construction — asserted, not trusted, because TabPFN rejects
infinities at input validation and a silent `Inf` would surface much later and
far less clearly.

### 8.2 Command

```bash
python scripts/run_feature_ablation.py --seed 42 --max-chunks 3 --test-size 1000 \
       --n-estimators 2 --max-service-flag-categories 40
```

**`--max-service-flag-categories 40` is required on 16 GB.** `common_service_flag`
has 336 distinct values in training; one-hot encoded alongside the existing
columns that totals **463 features**, against TabPFN's 500-feature pretraining
limit. A 9,691 × 463 context exhausts MPS memory regardless of test batch size
(§11.3). Capping at 40 yields 168 features.

### 8.3 Recorded results

From `reports/feature_ablation_20260831_141146_seed42.csv` — 3 chunks, 29,073
context rows, 1,000 test rows, seed 42:

| arm | features | accuracy | precision | recall | f1_score | roc_auc | runtime |
|---|---|---|---|---|---|---|---|
| baseline | 122 | 0.7800 | 0.9418 | 0.6538 | **0.7718** | 0.9651 | 922.7 s |
| engineered | 168 | 0.7610 | 0.9253 | 0.6309 | **0.7503** | 0.9617 | 952.8 s |
| **delta** | +46 | −0.0190 | −0.0165 | −0.0229 | **−0.0215** | −0.0034 | +30 s |

### 8.4 The two measurements disagree

This is the most important thing in this section, and it must not be smoothed
over.

| source | geometry | engineered-feature delta |
|---|---|---|
| `reports/feature_ablation_*.csv` | 3 chunks × 9,691 rows, 1,000 test rows | **−2.15 pp F1** |
| `notebooks/02_enhancement.ipynb` | 8 chunks × 2,470 rows, 400 test rows | **+3.56 pp F1** |

Same features, same cap of 40, same seed, **opposite signs**. The two runs
differ in chunk geometry and test subset, and both are single-seed. Neither is
conclusive.

**The defensible claim is that Enhancement 2's effect is unresolved.** It should
not be reported as an improvement on the strength of the notebook run, nor as a
regression on the strength of the CSV. Settling it requires multiple seeds at
matched scale — the single highest-value cheap experiment remaining.

---

## 9. Reproducing the Ablation Study

### 9.1 Command

```bash
python scripts/run_ablation.py --seed 42 --chunk-sizes 1000 2500 5000 10000 \
       --max-chunks 3 --test-size 1000 --n-estimators 2 \
       --max-service-flag-categories 40
```

Runtime on the reference machine: **~80 minutes** for all five configurations.
Results land in `reports/ablation_<timestamp>_seed42.csv`.

### 9.2 Study A — context size

Chunk *count* is held at 3 and the test set at 1,000 rows, so the only variable
is rows per chunk. Holding count fixed is what isolates context size; if the
whole partition were used, a smaller chunk size would also mean more chunks and
the two effects could not be separated. The cost of that choice, stated plainly:
a larger chunk size also means more total training data reaches the model.

| chunk size | context rows | f1_score | roc_auc | runtime | vs 1,000 |
|---|---|---|---|---|---|
| 1,000 | 3,000 | 0.7437 | 0.9562 | 39.3 s | — |
| 2,500 | 7,413 | 0.7721 | 0.9569 | 110.5 s | 2.8× |
| 5,000 | 14,538 | **0.7846** | 0.9602 | 290.1 s | 7.4× |
| 10,000 | 29,073 | 0.7503 | 0.9617 | **2,540.2 s** | **64.7×** |

**The cost curve is the finding.** F1 moves within 4.1 pp across a 10× change in
context size and ROC-AUC within 0.0055, while runtime grows **64.7×**. The
largest context is not even the best F1. Since all four points sit inside the
noise floor, the defensible claim is *"no measurable accuracy gain at large
cost"* — not "smaller is better".

Concretely: the 10,000-row context cost **2,501 s more than the 1,000-row one
for +0.66 pp F1.**

### 9.3 Study B — stratification

Both arms use `chunk_size=10,000`, 3 chunks, the same seed and the same test
rows. The only difference is whether chunks preserve the population class
balance.

| chunking | balance drift | mean confidence | f1_score | roc_auc | runtime |
|---|---|---|---|---|---|
| stratified | **0.000037** | 0.9427 | 0.7503 | 0.9617 | 2,540.2 s |
| random | 0.011904 | 0.9442 | **0.7746** | 0.9570 | 1,752.8 s |

Per-chunk attack rates:

```
  stratified   [0.46538, 0.46538, 0.46538]
  random       [0.46879, 0.46321, 0.45351]
```

Stratification reduces balance drift **322×**, exactly as designed. It shows no
accuracy benefit — random chunking scored 2.4 pp *higher* F1 on this single
seed, again inside the noise floor.

**Interpretation: on NSL-KDD, stratification is insurance rather than an
accuracy lever.** The dataset is near-balanced at 46.5% attack, so random chunks
are already nearly balanced by luck. The insurance is still worth keeping: on
UNSW-NB15 and CIC-IDS-2018, which are far more skewed, a random chunk can starve
of positives, and the drift column is what would catch it. One seed on one
near-balanced dataset is not grounds for dropping a guarantee.

---

## 10. Interpreting Results

### 10.1 The noise floor comes first

**The baseline's F1 standard deviation across three seeds is 0.0229 (2.29
percentage points).** This is the single most important number for reading any
comparison in this project.

A difference between arms smaller than 2.29 pp, measured on a single seed, is
seed-to-seed variation. It is not evidence of an effect and must not be reported
as one. Applying that rule consistently:

| comparison | delta | verdict |
|---|---|---|
| Chunked ensemble vs single context (§7.4) | −2.11 pp | within noise — no effect |
| Engineered features, recorded CSV (§8.3) | −2.15 pp | within noise — no effect |
| Engineered features, notebook run (§8.4) | +3.56 pp | exceeds floor, but contradicted |
| Context size 1,000 → 10,000 (§9.2) | +0.66 pp | within noise — no effect |
| Stratified vs random chunking (§9.3) | −2.43 pp | at the floor — no effect |

**No enhancement in this project has yet demonstrated a reproducible accuracy
gain.** Reporting that plainly is the correct scientific outcome, and it is
stronger than claiming a 1.8 pp win that a reviewer could dismantle by opening
one CSV.

No significance test is reported, deliberately. Wilcoxon's smallest attainable
two-sided *p* at n=3 is 0.25, so a *p*-value here would be uninformative by
construction. `significance.py` refuses to return one it could never have
rejected.

### 10.2 Tolerance for a reproduction attempt

Treat a reproduction as successful when each metric lands within **±5% relative**
of the recorded mean (the bands in §6.3). This is deliberately wider than the
2.29 pp noise floor: the floor measures seed variation on *one* machine, while
±5% must also absorb hardware, driver and library differences.

The two checks that matter more than the exact numbers:

1. **Precision should substantially exceed recall.** ≈0.93 vs ≈0.63. This
   ordering is a structural property of NSL-KDD's unseen attack types, and it
   should hold on any hardware. If it inverts, you have a bug.
2. **ROC-AUC should be ≥ 0.90** and noticeably higher than F1. The AUC/F1 gap is
   the signature described in §10.4.

### 10.3 Variation across hardware

Timings are the least portable thing here. Measured on the reference machine at
`n_estimators=2`, seconds per 1,000 test rows:

| context rows | 1,000 | 2,500 | 5,000 | 10,000 |
|---|---|---|---|---|
| s / 1,000 test rows | 15 | 49 | 144 | **459** |

Three cautions:

1. **Cost is superlinear in context size.** Attention over the in-context set is
   quadratic in context length, and past ~5,000 rows on MPS you also begin
   paying memory-pressure costs. Do not extrapolate linearly.
2. **The same machine varies by ~4×.** A 10,000-row context measured 108 s per
   1,000 test rows during the recorded baseline runs and 459 s during later
   measurement, purely from thermal state and background load. Three
   identically-configured baseline seeds recorded 542 s, 173 s and 172 s.
3. **Accuracy is far more portable than runtime.** Metrics should reproduce
   within the §10.2 bands on any correct installation; runtimes should not be
   compared across machines at all.

### 10.4 What the headline numbers actually say

**The error is almost entirely recall.** Precision 0.9347 against recall 0.6318
— the model is right when it flags an attack and misses about 37% of them. That
is the expected consequence of the 17 unseen attack types covering 16.6% of test
rows, not a defect. It is also precisely the gap Enhancement 1 was meant to
narrow, and has not yet narrowed.

**ROC-AUC 0.9555 against F1 0.7539 is a threshold problem, not a ranking
problem.** A model that ranks at 0.96 AUC while scoring 0.75 F1 at the default
0.5 cut is telling you the operating point is badly placed, not that the model
is weak. Threshold tuning on a validation split is the cheapest available
improvement and would likely lift F1 substantially with no model change.

It is deliberately **not** done here. Tuning a threshold on the test set is the
same category of leak as retaining the `difficulty` column, and would invalidate
the result. It is recorded in `docs/FUTURE_WORK.md` as the highest-value next
step.

### 10.5 Limitations to state before anyone asks

1. **The test set is subsampled.** Headline numbers use a stratified 5,000-row
   sample of the 22,544-row split; ablations use 1,000. Not directly comparable
   to published full-test-set results.
2. **Arms are not scale-matched.** The baseline is 3 seeds on 5,000 test rows;
   every ensemble and ablation arm is **1 seed** on 400–1,000 rows. Cross-arm
   deltas are indicative only.
3. **`n_estimators=2`, not TabPFN's auto-scaled 8.** Chosen for runtime, held
   identical across arms.
4. **The Enhancement 2 category cap changes what is measured.** At 40
   categories `common_service_flag` covers ~88% of training rows instead of
   100%. A hardware constraint, not a design choice.
5. **Binary classification only.** The 5-class taxonomy is not implemented;
   NSL-KDD's U2R class is ~0.04% of training rows (about 52 of 125,973), so
   macro-F1 over five classes would be dominated by a class with almost no
   support.
6. **NSL-KDD only.** UNSW-NB15 and CIC-IDS-2018 are downloaded but unused — and
   they are the datasets where chunking should matter most.
7. **No classical baselines.** XGBoost and LightGBM are not implemented, so
   there is no non-TabPFN reference point.
8. **Single hardware configuration.** Every timing is from one 16 GB M1 Air.

---

## 11. Troubleshooting

### 11.1 Quick reference

| Symptom | Cause and fix |
|---|---|
| `ModuleNotFoundError: tabpfn_nids` | `pip install -e .` was not run. Do **not** set `PYTHONPATH`. |
| `ModuleNotFoundError: No module named 'src.data_pipeline'` | Wrong import path. Use `from tabpfn_nids...`; there is no `src` package. |
| `TabPFNHuggingFaceGatedRepoError` | `TABPFN_CHECKPOINT` was changed to `"auto"`. Restore `"tabpfn-v2-classifier.ckpt"`. |
| `TabPFNMPSOutOfMemoryError` mid-ensemble | Lower `--predict-batch-size`; ensure `TabPFNWrapper.free()` runs between chunks. |
| Same error with engineered features on | Context too wide (463 features). Set `--max-service-flag-categories 40`. |
| "Running on CPU with more than 200 samples may be slow" | MPS unavailable. Check `torch.backends.mps.is_available()`. |
| Run takes hours | `n_estimators` is `"auto"` (resolves to 8) and/or `--test-size` is unset. |
| `FileNotFoundError` on NSL-KDD | Datasets not downloaded; see §4. |
| Parquet read fails | `pip install pyarrow`. |
| `pytest` reports "4 deselected" | Expected. Those carry the `slow` marker. Not a failure. |

### 11.2 TabPFN installation

The most common first-run problem is not an error at all: TabPFN downloads a
~100 MB checkpoint on first use, so `smoke_test.py` can appear to hang for
several minutes. Let it finish; subsequent runs use the cache.

If the download fails behind a proxy or on a read-only filesystem, set the cache
location explicitly:

```bash
export TABPFN_MODEL_CACHE_DIR=~/.cache/tabpfn
```

If you see a **gated repo** error, you are loading `tabpfn_3` rather than the
pinned v2 checkpoint — see §3.5. Do not resolve this by authenticating to
HuggingFace: that loads a different model than the one this project claims to
reproduce.

### 11.3 MPS memory — two distinct failures

These are different problems with different fixes, and conflating them wastes
time.

**Failure 1: accumulation across chunks.** The chunked ensemble died at chunk 4
of 13 with `TabPFNMPSOutOfMemoryError`. Metal does not release memory when a
Python reference is dropped, so allocations accumulate across sequential chunks.
Fixed by `TabPFNWrapper.free()`, which calls `torch.mps.empty_cache()` between
chunks; verified over 6 sequential chunks. **Symptom: the run succeeds for
several chunks, then dies.**

**Failure 2: the context itself is too large.** With engineered features
uncapped the feature count is 463, and a 9,691 × 463 context exhausts MPS
memory regardless of test batch size — it OOMs at batch 1,000, 250 and 100
alike. The memory goes to the *context*, not the test rows, so batching cannot
fix it. Use `--max-service-flag-categories 40`. **Symptom: the run dies
immediately on the first chunk, and lowering the batch size does not help.**

### 11.4 Memory on a 16 GB M1

MPS shares system memory, so other applications compete with the model. If runs
are unstable:

1. Reduce `--test-size` first — prediction cost is linear in it.
2. Reduce `--chunk-size`. Per §9.2 this costs little or no accuracy and saves
   enormous time.
3. Cap `--max-service-flag-categories` when engineered features are on.
4. Close browsers and other memory-heavy applications. Their footprint is not
   incidental at this margin.

### 11.5 Dataset loading errors

| Error | Meaning |
|---|---|
| `FileNotFoundError: KDDTrain+.txt` | Files are not in `data/raw/nsl-kdd/`. Check for a nested directory created by `--unzip`. |
| Shape is not `(125973, 43)` | Wrong file (e.g. `KDDTrain+_20Percent.txt`) or a truncated download. |
| Whole numeric columns are `object` | CIC-IDS-2018 embedded header rows — §4.4, problem 2. |
| Labels never match | CIC-IDS-2018 CRLF endings — §4.4, problem 3. `Benign\r` is not `Benign`. |
| TabPFN rejects input as non-finite | Infinities in CIC-IDS-2018 — §4.4, problem 4. |

---

## 12. References

Hollmann, N., Müller, S., Purucker, L., Krishnakumar, A., Körfer, M., Hoo, S. B.,
Schirrmeister, R. T., & Hutter, F. (2025). Accurate predictions on small data
with a tabular foundation model. *Nature, 637*(8045), 319–326.
https://doi.org/10.1038/s41586-024-08328-6

Moustafa, N., & Slay, J. (2015). UNSW-NB15: A comprehensive data set for network
intrusion detection systems (UNSW-NB15 network data set). In *2015 Military
Communications and Information Systems Conference (MilCIS)* (pp. 1–6). IEEE.
https://doi.org/10.1109/MilCIS.2015.7348942

Tavallaee, M., Bagheri, E., Lu, W., & Ghorbani, A. A. (2009). A detailed
analysis of the KDD CUP 99 data set. In *2009 IEEE Symposium on Computational
Intelligence for Security and Defense Applications* (pp. 1–6). IEEE.
https://doi.org/10.1109/CISDA.2009.5356528

Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). Toward generating a
new intrusion detection dataset and intrusion traffic characterization. In
*Proceedings of the 4th International Conference on Information Systems Security
and Privacy (ICISSP)* (pp. 108–116). SciTePress.
https://doi.org/10.5220/0006639801080116
