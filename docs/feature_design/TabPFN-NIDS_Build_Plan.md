# TabPFN-NIDS — Build Plan

Full end-to-end build sequence from empty folder to working project ready for Review-II.

**Project:** TabPFN-NIDS: Zero-Shot Network Intrusion Detection using a Tabular Foundation Model
**Repo:** https://github.com/Sujeethh03/TabPFN-NIDS
**Hardware:** MacBook Air M1, 16 GB memory, 100 GB free disk
**Review-II date:** 29 August 2026

Each section below is one buildable unit — finish it, test it, commit it, move to the next.

---

## Phase 0 — Foundation setup

### 0.1 — Project structure
Create the folder tree: `src/`, `notebooks/`, `tests/`, `scripts/`, `data/`, `reports/`, `docs/`. Each `src/` subfolder gets an empty `__init__.py`. This makes Python treat them as packages so you can import cleanly.

### 0.2 — Virtual environment and dependencies
Create a Python 3.11 venv. Install `tabpfn`, `scikit-learn`, `pandas`, `numpy`, `torch`, `matplotlib`, `seaborn`, `tqdm`, `pyyaml`, `pytest`. Freeze exact versions into `requirements.txt`. This locks your project to a known-working state.

### 0.3 — Smoke test
Write `smoke_test.py` that runs TabPFN on the sklearn breast-cancer dataset and prints accuracy. If this passes, your environment works. If it fails, you fix it before writing any other code.

### 0.4 — Git initialization and first commit
`git init`, add `.gitignore` (venv, data, `__pycache__`), commit the empty scaffold. Push to GitHub. Every phase from here gets its own commit.

---

## Phase 1 — Data pipeline

### 1.1 — Dataset download and verification
Download NSL-KDD from Kaggle into `data/nsl-kdd/`. Write a small verification script that checks the files exist and prints row counts. If wrong number of rows, you know something's off before writing code.

### 1.2 — NSL-KDD loader
Write `src/data_pipeline/loader.py` with a `load_nsl_kdd()` function that returns a pandas DataFrame with proper column names (NSL-KDD ships without headers — you assign the 41 known feature names manually). Handle both training and test files.

### 1.3 — Data preprocessing
Write `src/data_pipeline/preprocessor.py` that handles: missing values, categorical columns (protocol, service, flag → one-hot or label encoded), numerical scaling (StandardScaler), and binary label conversion (multi-class attack labels → normal vs attack).

### 1.4 — Train/test splitter
Write `src/data_pipeline/splitter.py` using `StratifiedShuffleSplit`. Preserves the attack/normal ratio in both sets. Configurable test size (default 20%). Fixed random seed for reproducibility.

### 1.5 — Tests for data pipeline
Write `tests/test_data_pipeline.py` — verifies loader returns correct shape, preprocessor handles missing values, splitter maintains class balance.

---

## Phase 2 — Baseline model (vanilla TabPFN)

### 2.1 — TabPFN wrapper
Write `src/models/tabpfn_wrapper.py` — a thin class around `TabPFNClassifier` that handles the 10K-sample limit gracefully (raises a clear error if exceeded), configures MPS or CPU backend, and returns both predictions and confidence scores.

### 2.2 — Baseline runner
Write `scripts/run_baseline.py` — subsamples NSL-KDD to under 10K rows, splits, fits vanilla TabPFN, predicts, saves metrics to `reports/baseline.csv`. This is your reproduction baseline — the number your enhancement must beat.

### 2.3 — Verify baseline results
Run the baseline. Expect F1 around 0.75–0.85 on NSL-KDD binary classification. If wildly off, something's wrong. Compare against Ferrag 2020 survey numbers to sanity check.

---

## Phase 3 — Enhancement 1: Chunked ensemble

### 3.1 — Stratified chunking function
Write `src/models/chunker.py` — takes a DataFrame and chunk size (default 10,000), returns a list of DataFrames where each chunk preserves the class balance of the full dataset. Use `StratifiedShuffleSplit` iteratively.

### 3.2 — Parallel inference
Write `src/models/chunked_ensemble.py` — takes chunks and a test set, runs `TabPFNClassifier` on each chunk against the same test set, collects predictions and confidence scores. Sequential for now; parallelize later if time allows.

### 3.3 — Weighted vote aggregation
Extend `chunked_ensemble.py` with an aggregator function. For each test row, collect predictions from all chunks along with confidences, compute weighted average of predicted probabilities, output final class. Configurable: simple majority, confidence-weighted, or learned weights.

### 3.4 — End-to-end enhancement runner
Write `scripts/run_enhanced.py` — loads full NSL-KDD (all 150K rows), chunks it, runs chunked ensemble, evaluates. This is where Enhancement 1 gets its numbers.

### 3.5 — Tests for chunked ensemble
Write `tests/test_ensemble.py` — verify chunker preserves stratification, verify aggregator produces reasonable outputs on toy data.

---

## Phase 4 — Enhancement 2: Domain-aware feature engineering

### 4.1 — Feature engineering module
Write `src/features/engineered.py` — takes a preprocessed DataFrame and adds engineered columns: `packet_rate = packets / duration`, `byte_ratio = src_bytes / dst_bytes`, `is_short_session`, protocol-specific interaction features.

### 4.2 — Toggle for feature engineering
Add a boolean flag `use_engineered_features` to your pipeline. This lets you run controlled experiments: same data, same model, only feature difference. Isolates Enhancement 2's contribution.

### 4.3 — Enhancement 2 ablation script
Write `scripts/run_feature_ablation.py` — runs the pipeline twice, once with engineered features off, once on, saves comparative metrics to `reports/feature_ablation.csv`.

### 4.4 — Tests
Verify engineered features have no NaN/Inf, verify shapes match expected.

---

## Phase 5 — Evaluation layer

### 5.1 — Metrics module
Write `src/evaluation/metrics.py` — computes F1 (macro and per-class), precision, recall, ROC-AUC, confusion matrix. Returns as a dict for easy CSV export.

### 5.2 — Statistical significance
Add a `wilcoxon_test` function — takes two lists of scores from multiple random seeds and returns a p-value. Used when comparing vanilla vs enhanced.

### 5.3 — Report writer
Write `src/evaluation/reporter.py` — writes metrics dicts to timestamped CSVs in `reports/`. Logs hardware info, seed, dataset used, runtime.

### 5.4 — Plot generator
Write `src/evaluation/plots.py` — generates confusion matrix heatmap (Seaborn), ROC curve, PR curve, saves as PNG in `reports/figures/`.

---

## Phase 6 — UNSW-NB15 support

### 6.1 — UNSW-NB15 loader
Download UNSW-NB15 from Kaggle. Extend `loader.py` with `load_unsw_nb15()`. Column names and preprocessing differ from NSL-KDD, so you'll have a separate preprocessor branch.

### 6.2 — UNSW-specific preprocessing
Handle UNSW's 49 features, its categorical columns (proto, service, state), and its 10-class attack labels. Convert to binary (normal vs attack) for consistency with NSL-KDD experiments.

### 6.3 — Enhancement 1 on UNSW-NB15
Run `scripts/run_enhanced.py` pointed at UNSW-NB15. This is where the chunked ensemble really matters (250 chunks on 2.5M rows). Expect longer runtime (20–40 minutes).

### 6.4 — Compare vanilla vs chunked on UNSW-NB15
Vanilla TabPFN on 10K sample vs full chunked ensemble. Show the delta. This is the killer table for your review.

---

## Phase 7 — Enhancement 3: Cross-dataset test

### 7.1 — CIC-IDS 2018 loader
Download CIC-IDS 2018 (start with a subset if disk is tight). Write `load_cic_ids_2018()`. Handle the Infinity/NaN issues explicitly — log rows removed.

### 7.2 — Feature alignment
UNSW-NB15 and CIC-IDS 2018 have different feature sets. Write an `align_features()` function that maps their common features and drops the rest. This is a non-trivial data engineering task — expect a few days of debugging.

### 7.3 — Cross-dataset runner
Write `scripts/run_cross_dataset.py` — trains context from UNSW-NB15 (via chunked ensemble), tests on CIC-IDS 2018. Compares transfer F1 vs in-domain F1.

### 7.4 — Interpret results honestly
Whatever the transfer gap is, document it. Small gap = TabPFN generalizes well. Large gap = it doesn't. Both are valid findings.

---

## Phase 8 — Baseline comparisons

### 8.1 — Classical ML baselines
Add `scripts/run_classical_baselines.py` — XGBoost and LightGBM on the same train/test splits. This gives you a comparison anchor: "TabPFN F1 vs XGBoost F1 on the same data." Reviewers love this.

### 8.2 — Comparison table
Write a script that reads all `reports/*.csv` and generates one combined comparison table (Vanilla TabPFN, +Enhancement 1, +Enhancement 2, +Both, XGBoost baseline). Save as `reports/comparison_table.csv` and `reports/comparison_table.md`.

---

## Phase 9 — Notebooks for the review demo

### 9.1 — Reproduction notebook
`notebooks/01_reproduction.ipynb` — walks through data loading, preprocessing, vanilla TabPFN on NSL-KDD, results. Runs top to bottom in under 5 minutes.

### 9.2 — Enhancement notebook
`notebooks/02_enhancement.ipynb` — chunked ensemble + feature engineering on UNSW-NB15. Shows the improvement over baseline.

### 9.3 — Cross-dataset notebook
`notebooks/03_cross_dataset.ipynb` — the transfer experiment.

### 9.4 — Results summary notebook
`notebooks/04_results_summary.ipynb` — pulls all CSVs, generates the final comparison table and figures. This is the notebook you demo live during the review.

---

## Phase 10 — Documentation

### 10.1 — README polish
Rewrite `README.md` with: project overview, quick start (3 commands to run reproduction), results snapshot table, folder structure, license.

### 10.2 — 10-page reproducibility document
Write `docs/REPRODUCIBILITY.md` — the detailed guide anyone can follow to reproduce your work. Covers environment, data, code structure, running each experiment, expected outputs, troubleshooting.

### 10.3 — Contribution statement
Write `docs/CONTRIBUTIONS.md` — clearly states what's original in your work vs what's from the TabPFN paper. Enhancement 1, 2, 3 each get their own subsection.

### 10.4 — Future work document
Write `docs/FUTURE_WORK.md` — Phase-II candidates: real-time streaming, explainability (SHAP), adversarial robustness, multi-class deep dive.

---

## Phase 11 — GitHub cleanup

### 11.1 — Verify no hardcoded paths
Grep the codebase for `/Users/`, `/home/`, absolute paths. Replace with `pathlib.Path` relative to project root.

### 11.2 — Verify no secrets or personal data
No API keys, no `.env` files committed, no experimental output CSVs with your local machine info in them (unless that's intentional for the review).

### 11.3 — Tag a release
`git tag v0.1-review-ii`, `git push --tags`. Reviewers can see this specific snapshot even if you push more changes later.

### 11.4 — Update GitHub repo description and topics
Add repo description: "Phase-I project applying TabPFN v2 to network intrusion detection." Add topics: `tabpfn`, `nids`, `foundation-models`, `cybersecurity`, `machine-learning`.

---

## Phase 12 — Presentation

### 12.1 — Update Review-II slide deck
Refresh the Phase-I deck with actual numbers, real results tables, updated architecture reflecting what you built.

### 12.2 — Add new slides for Review-II specifically
Slide for Implementation Details (code structure, GitHub link). Slide for Results (the comparison table). Slide for Contributions (what's yours vs what's from the paper).

### 12.3 — Rehearse full walkthrough
Time yourself. Aim for 10 minutes total. Practice the live demo of running one notebook end-to-end.

---

## Phase 13 — Final verification

### 13.1 — Fresh-clone test
On a friend's machine (or on your Mac after `rm -rf ~/Documents/TabPFN-NIDS`), clone from GitHub, follow the README, verify everything works. This catches "works on my machine" bugs before the reviewer catches them.

### 13.2 — Supervisor sign-off
Send the deck and documentation to Prof. Krishnanjaneyulu at least 3 days before the review. Address any feedback.

### 13.3 — Backup plan
Have your notebooks pre-run and saved with output cells visible. If the live demo fails during the review, you can still show pre-run results.

---

## Order and dependencies

Must do phases in this order:

**Foundation (Phase 0) → Data (Phase 1) → Baseline (Phase 2) → Enhancement 1 (Phase 3)**

After that, Phase 4 (Enhancement 2) and Phase 5 (Evaluation) can happen in parallel.

Phase 6 (UNSW-NB15) requires Phases 1–5 done first.

Phase 7 (Cross-dataset) requires Phase 6 done.

Phase 8 (Baselines) can happen anytime after Phase 2.

Phases 9–13 (Notebooks, Docs, Cleanup, Presentation, Verification) are polish — Phase 9 needs all experiments done first.

---

## Rough time allocation (24 days available)

| Days | Phase |
|---|---|
| Days 1–3 | Phase 0 + Phase 1 (foundation and data pipeline) |
| Days 4–6 | Phase 2 + Phase 3 (baseline and chunked ensemble) |
| Days 7–9 | Phase 4 + Phase 5 (feature engineering and evaluation) |
| Days 10–13 | Phase 6 (UNSW-NB15 support and experiments) |
| Days 14–16 | Phase 7 (cross-dataset test) — SKIP if running late |
| Days 17–18 | Phase 8 (baseline comparisons) |
| Days 19–20 | Phase 9 (notebooks) |
| Days 21–23 | Phase 10 + 11 (documentation and cleanup) |
| Day 24 | Phase 12 + 13 (presentation, verification, submission) |

---

## Which features to cut if running late

If behind schedule by Day 15:

- **Cut Phase 7** (cross-dataset test) — mention as Phase-II future work
- **Cut Phase 8's LightGBM** — keep only XGBoost as the classical baseline
- **Cut Phase 5.4** (plots) — keep only the metrics table

Never cut: Phases 0–4, 9 (at least one notebook), 10 (README + reproducibility doc), 11 (GitHub cleanup), 12 (presentation), 13 (rehearsal).

---

## What "done" means for the whole project

Done when a reviewer can:

1. Open the GitHub repo and understand what it does in 30 seconds
2. Clone it, follow README, run reproduction script, get numbers within ±5% of ours
3. Open one notebook and see the results end-to-end
4. Read the reproducibility doc and understand every experiment
5. Watch the presentation for 10 minutes and understand the contributions

If all five are true, we're ready for August 29.

---

*Start with Phase 0 today. Move phase by phase. Commit after each. Ask for help at each checkpoint.*
