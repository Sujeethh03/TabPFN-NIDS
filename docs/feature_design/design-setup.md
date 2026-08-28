# TabPFN-NIDS — Design & Setup

**Status:** Draft for sign-off
**Date:** 28 August 2026
**Author:** Generated during Phase 0.1 review
**Supersedes:** Section 0.1 of `TabPFN-NIDS_Build_Plan.md`; Section 7 "Repository Structure" of `TabPFN-NIDS_Project_Guide.md`

This document does three things:

1. Records **verified facts** about the environment, measured on this machine rather than assumed.
2. Lists **contradictions and errors** found across the four planning documents, with a recommended resolution for each.
3. Specifies the **repository structure and module contracts** the build should follow.

Everything in Section 2 was checked against the installed library or the filesystem. Where a number could not be verified it is marked `TODO` rather than guessed.

---

## 1. Source documents reviewed

| Document | Role | Key content |
|---|---|---|
| `TabPFN-NIDS_Project_Guide.md` | Master plan, 16-week | Papers, datasets, 3 enhancements, repo structure, grading rubric |
| `docs/feature_design/TabPFN-NIDS_Build_Plan.md` | Execution sequence, 24-day | Phases 0–13, module-by-module build order |
| `Project-Phase-I-template.pptx.pdf` | Review-II slide deck, 11 slides | Problem statement, related work, datasets, expected outcomes |
| `architecture_diagram.svg` | Pipeline diagram | Raw data → preprocess → chunk → TabPFN ×N → weighted vote → evaluate |

The four documents agree on the scientific substance. They disagree on scope, target variable, timeline and library version. Those disagreements are itemised in Section 3.

---

## 2. Verified environment facts

Measured on this machine on 28 August 2026. These replace the assumptions in Guide §7 and Build Plan §0.2.

### 2.1 Toolchain

| Item | Planned | Actual | Note |
|---|---|---|---|
| Python | 3.11 | 3.11.9 (arm64, python.org build) | `venv/` created and working |
| tabpfn | `>=2.0.0` | **8.5.0 installed** | Major version drift — see §3.1 |
| torch | `>=2.2.0` | 2.13.0 | MPS available: **yes** |
| scikit-learn | `>=1.4.0` | 1.9.0 | |
| pandas | `>=2.2.0` | 3.0.5 | pandas 3.x — some pandas-2 idioms in the EDA notebook are deprecated |
| Free disk | 100 GB | **81 GB** | Still sufficient for ~30 GB of data |

`requirements.txt` currently lists floors only (`>=`) and has **not** been frozen. Build Plan §0.2 requires exact pins; this is outstanding.

### 2.2 TabPFN pretraining limits — confirmed by reading the installed source

From `tabpfn/inference_config.py`:

| Constant | Value | Consequence for this project |
|---|---|---|
| `MAX_NUMBER_OF_SAMPLES` | `10_000` | **Enhancement 1's premise is correct and verified.** |
| `MAX_NUMBER_OF_FEATURES` | `500` | Never binding: NSL-KDD 41, UNSW 49, CIC ~80 |
| `MAX_NUMBER_OF_CLASSES` | `10` | Binding for UNSW-NB15, which has exactly 10 classes (9 attacks + Normal). No headroom. |
| `MAX_CPU_SAMPLES` | `1_000` | **CPU inference is disallowed above 1,000 samples by default.** MPS or `ignore_pretraining_limits=True` required. |
| `PASSTHROUGH_INF` | `False` | **Infinities are rejected at input validation**, not silently passed. Directly relevant to RULE 6 and CIC-IDS-2018. |

These are soft limits: exceeding them raises unless `ignore_pretraining_limits=True` is passed. They are *not* hard architectural caps, but going past them degrades accuracy, which is precisely the effect Enhancement 1 exists to avoid.

### 2.3 Blocker found: the default model is gated

`tabpfn` 8.5.0 defaults to downloading `Prior-Labs/tabpfn_3` from HuggingFace, which is a **gated repository**. A clean install fails with:

```
TabPFNHuggingFaceGatedRepoError: HuggingFace authentication error
downloading from 'Prior-Labs/tabpfn_3'. This model is gated and
requires you to accept its terms.
```

This breaks `smoke_test.py`, breaks the "clone and run in 30 minutes" claim in Guide §2, and breaks the fresh-clone test in Build Plan §13.1.

**Verified fix:** pass `model_path="tabpfn-v2-classifier.ckpt"`, which resolves to the **ungated** `Prior-Labs/TabPFN-v2-clf` repository. Confirmed working:

```
model_path=tabpfn-v2-classifier.ckpt  →  breast-cancer accuracy 0.9649 in 15.3 s (CPU)
```

This is also the *scientifically correct* choice: `tabpfn_3` is a newer model than the one described in the Nature 2025 paper. A project whose stated contribution is reproducing and extending TabPFN v2 must load the v2 checkpoint, or the reproduction claim is void. **The model checkpoint must be pinned explicitly in code, never left as `"auto"`.**

### 2.4 Measured runtime on this hardware

Synthetic data, 5 classes, 2,000 test rows, v2 checkpoint, `ignore_pretraining_limits=True`.

| Context rows | Features | Device | Accuracy | Fit | Predict (2,000 rows) | Total |
|---|---|---|---|---|---|---|
| 1,000 | 40 | MPS | 0.908 | 0.6 s | 22.7 s | **23.3 s** |
| 1,000 | 40 | CPU | 0.907 | 0.2 s | 86.0 s | **86.2 s** |
| 5,000 | 40 | MPS | 0.969 | 0.5 s | 97.8 s | **98.3 s** |
| 10,000 | 40 | MPS | 0.961 | 0.8 s | 279.9 s | **280.7 s** |
| 10,000 | 40 | CPU | *running* | | | |
| 10,000 | 80 | MPS | *running* | | | |

Four conclusions, and the third one reshapes the experiment plan.

1. **MPS is ~3.7× faster than CPU** at equal accuracy. MPS is the correct default; the `MAX_CPU_SAMPLES = 1000` guard exists for good reason.
2. **Cost is dominated by prediction, not fitting.** TabPFN's `fit` only caches the context; the transformer forward pass happens in `predict`. Fit time is negligible (< 1 s) at every size tested.
3. **Prediction cost grows super-linearly in context size.** 1K → 5K → 10K context rows cost 22.7 s → 97.8 s → 279.9 s for a *fixed* 2,000-row test set: a 10× larger context costs **12.3×** more. Full-size chunks are disproportionately expensive.
4. **Accuracy saturates well before the limit.** 5,000 context rows scored 0.969 against 10,000 rows' 0.961 on this synthetic task — within noise, at roughly one third of the cost.

### 2.5 Runtime budget — implication for Enhancement 1

The chunked ensemble runs *the same test set* through *N* chunks, so total cost is `N × predict(chunk_size, n_test)`. Extrapolating from the measured 10,000-row figure at 2,000 test rows:

| Experiment | Chunks | Test rows | Estimated runtime |
|---|---|---|---|
| NSL-KDD, 125K train, 22.5K test | 13 | 22,544 | ~11 hours |
| UNSW-NB15, 175K train, 82K test | 18 | 82,332 | ~57 hours |

**These are not runnable in the time available, and the deck's "20–40 minutes" estimate in Build Plan §6.3 is wrong by two orders of magnitude.** Three levers bring it back into range, and they should be applied together:

- **Subsample the test set.** A stratified 5,000-row test sample gives metric standard errors under ±0.01 and cuts cost ~4.5× on NSL-KDD and ~16× on UNSW. This is the single biggest lever and costs almost nothing in statistical power.
- **Use a smaller chunk size.** `chunk_size=5000` is ~2.9× cheaper per chunk than 10,000 while doubling chunk count — net ~1.4× cheaper overall, with no measured accuracy loss (conclusion 4). Chunk size should be selected by measurement, not set to the 10,000 ceiling by default.
- **Cap chunk count.** Sampling *M* chunks rather than exhausting the partition turns runtime into a directly chosen budget; the ensemble's benefit saturates well before every chunk is used.

**Recommended starting configuration: `chunk_size=5000`, `max_chunks=10`, stratified test sample of 5,000 rows** — roughly 20 minutes per experiment arm on MPS, which makes the full comparison table achievable. The full-partition run stays available as a single overnight job if time allows.

---

## 3. Contradictions across the planning documents

Ten issues, ordered by how much damage they cause if left unresolved.

### 3.1 — TabPFN version drift *(blocking)*

Guide §7, the PPT slide 8 and `requirements.txt` all say TabPFN v2 (`>=2.0.0`). The installed package is 8.5.0, whose default weights are a different model entirely (§2.3).

**Recommendation:** keep `tabpfn==8.5.0` but pin `model_path` to the v2 checkpoint, and state this explicitly in the reproducibility document. The alternative — downgrading to `tabpfn==2.2.1` — is also defensible and arguably cleaner for a reproduction, but 2.x is older and may not build against torch 2.13. Pinning the checkpoint on a current library is the lower-risk path.
**Either way, `requirements.txt` must pin an exact version, and the checkpoint filename must appear in the results CSV as provenance.**

### 3.2 — Binary vs 5-class target *(blocking, decided but inconsistent)*

- Build Plan §1.3: "binary label conversion (multi-class attack labels → normal vs attack)"
- Build Plan §6.2: convert UNSW "to binary for consistency with NSL-KDD experiments"
- Guide §11 lists "multi-class deep dive" as *future work*, i.e. explicitly out of scope
- **The decision taken in this session was 5-class multiclass** (normal / dos / probe / r2l / u2r)

The plan as written contradicts the decision. Three consequences follow if 5-class is kept:

1. **ROC-AUC becomes ambiguous.** The success criteria list ROC-AUC as a headline metric; for 5-class it must be specified as macro one-vs-rest, and that must be stated in every table.
2. **The sanity band is wrong.** Build Plan §2.3's "expect F1 0.75–0.85" is a *binary* NSL-KDD band. Macro-F1 over 5 classes on NSL-KDD is typically far lower, because R2L and U2R are severely under-represented (U2R is roughly 0.04% of the training set). A reviewer comparing against §2.3 will think the pipeline is broken when it is not.
3. **UNSW-NB15 has 10 classes, exactly at `MAX_NUMBER_OF_CLASSES`.** Any mapping that adds an "unknown" bucket exceeds the limit.

**Recommendation:** run **binary as the primary headline metric** and **5-class as a secondary table**. This satisfies the plan, keeps the Ferrag 2020 comparison anchor valid, keeps ROC-AUC unambiguous, and still delivers the multiclass depth. Cost is roughly 1.4× runtime, since both share the same fitted context. If only one can be run, the binary result is the one the review needs.

### 3.3 — UNSW-NB15 row count is wrong by 14× *(high)*

Guide §4, PPT slide 6, and Build Plan §6.3 all describe UNSW-NB15 as ~2.5M rows and plan "250 chunks on 2.5M rows" as the headline scaling result.

The Kaggle source specified (`dhoogla/unswnb15`) is the **cleaned partition**, which is the standard 175,341-row training set and 82,332-row test set — about 257,000 rows total, not 2.5M. The full 2.54M-record UNSW-NB15 is a different distribution, published as four separate CSVs.

At 175K rows the chunked ensemble yields ~18 chunks, not 250. The enhancement still works and still demonstrates the point, but **the headline number in the deck is wrong** and a reviewer who knows the dataset will notice.

**Recommendation:** correct the deck to the true row count, and describe Enhancement 1 as "scales TabPFN from a 10K context to the full 175K training partition (~18 chunks)". If the 2.5M figure is wanted, the full four-CSV release must be downloaded instead, which changes preprocessing and adds several GB.

### 3.4 — Wilcoxon test cannot reach significance as planned *(high)*

Build Plan §5.2 specifies a `wilcoxon_test` comparing vanilla vs enhanced. Guide §5 Week 5 specifies **3 runs**.

The Wilcoxon signed-rank test with n=3 paired samples has a minimum achievable two-sided p-value of 0.25. **No result can ever be significant at α=0.05.** Reporting "p = 0.25, not significant" when the design made significance impossible is a methodological error a reviewer may well catch.

**Recommendation:** either run **≥ 10 seeds** (the smallest n where two-sided p < 0.05 is reachable is n=6, but 10 gives margin), or drop the hypothesis test and report **mean ± std across 5 seeds** honestly. The second option is cheaper and perfectly acceptable at Phase-I level. Do not report a test that cannot pass.

### 3.5 — Project title contradicts itself *(medium, cosmetic but visible)*

- PPT slide 1: "Zero-Shot Network Intrusion Detection"
- Build Plan header: "Zero-Shot Network Intrusion Detection"
- `README.md`: "Zero-shot network intrusion detection"
- **Guide §10 explicitly rejects this title**: "'Zero-shot' is technically contestable given we do provide labeled training rows as in-context examples. The stronger title is *Scaling Tabular Foundation Models for Network Intrusion Detection*."

The Guide is right. TabPFN receives labelled examples in its context window; that is in-context learning, not zero-shot. This is the easiest possible question for a reviewer to ask, and the current title invites it.

**Recommendation:** adopt "Scaling Tabular Foundation Models for Network Intrusion Detection" everywhere — deck, README, repo description, report. Keep "zero-shot transfer" only for Enhancement 3, where no target-domain labels are used and the term is accurate.

### 3.6 — Chunk overlap is specified two different ways *(medium)*

- Guide §6 Enhancement 1: "split training data into **overlapping** stratified chunks"
- Build Plan §3.1: "returns a **list** of DataFrames" — implies a disjoint partition

Overlapping and disjoint chunks are different algorithms with different variance and different runtime.

**Recommendation:** implement **disjoint stratified partitioning as the default**, with an `overlap_ratio: float = 0.0` parameter that enables overlap. Disjoint is simpler to explain, cheaper to run, and each chunk stays an independent sample. Overlap becomes an ablation if time permits. Whichever is used must be stated in the deck.

### 3.7 — Enhancement 1's novelty needs a defensive citation *(medium)*

TabPFN 8.5.0 exposes `n_estimators` with `auto_scale_n_estimators=True` and ships its own ensembling machinery. A reviewer may ask whether Enhancement 1 duplicates a library feature.

It does not — `n_estimators` ensembles *preprocessing variations over the same context*, not *disjoint subsamples of a larger-than-context dataset*. But this distinction must be stated explicitly in `docs/CONTRIBUTIONS.md`, with the baseline configuration (`n_estimators`) held constant between vanilla and enhanced runs so the comparison isolates chunking. **If `n_estimators` differs between the two arms, the comparison is invalid.**

### 3.8 — Timelines are mutually inconsistent *(context)*

Guide: 16 weeks. Build Plan: 24 days. Guide's own "last updated" is 5 August 2026. Today is 28 August 2026; the review is 29 August 2026. The 24-day allocation in Build Plan §"Rough time allocation" has already elapsed.

This document does not attempt to resolve the schedule; it notes that **the build must be prioritised as if one day remains**, and that Build Plan §"Which features to cut if running late" (cut Phase 7 cross-dataset, cut LightGBM, cut plots) is the operative guidance.

### 3.9 — Team and contact metadata is stale *(low, but visible to reviewers)*

- PPT slide 1 lists two students: Sujeeth G (2320090080) and P. Mahitha (2320090056), supervised by P. Krishnanjaneyulu
- `README.md` still says "Teammate 1", "Teammate 2", "[Supervisor Name]"
- Guide §8 describes **three** roles; Guide §14 has "[Fill in supervisor name]"

**Recommendation:** reconcile to the two names on the deck, and fill in the supervisor everywhere before the repo is shown.

### 3.10 — Reference page numbers appear to be placeholders *(low)*

Guide §13 and PPT slide 10 cite Yousfi et al. 2024 as *The Journal of Supercomputing*, **80**, 12345–12367. The page range `12345–12367` is a sequential placeholder pattern. **TODO: verify the real page numbers before the report is submitted.** Per RULE 1 this has not been guessed at here.

---

## 4. Repository structure

### 4.1 The problem with `src/`

Build Plan §0.1 says to create `src/` with `__init__.py` in each subfolder. That works, but only while the working directory is the project root. It fails from `notebooks/`, which is where Phase 9 lives, and it fails under `pytest` from `tests/`. The usual workaround — `sys.path.append("..")` at the top of every notebook — is exactly the kind of thing that breaks the fresh-clone test in §13.1.

The fix is to make the code an **installed, named package** rather than a directory that happens to be called `src`. One `pip install -e .` and the same import works from a script, a notebook, a test, or a REPL, from any directory.

### 4.2 Proposed tree

```
TabPFN-NIDS/
├── pyproject.toml              # packaging + pytest/ruff config
├── requirements.txt            # exact pinned freeze (Build Plan §0.2)
├── Makefile                    # make setup / test / baseline / enhanced
├── README.md   LICENSE   .gitignore
├── smoke_test.py               # must be updated to pin the v2 checkpoint (§2.3)
│
├── tabpfn_nids/                # the installed package (replaces src/)
│   ├── __init__.py             # __version__ + curated re-exports only
│   ├── config.py               # paths, SEED, logging, environment capture  [EXISTS]
│   │
│   ├── datasets/               # one module per dataset, one shared interface
│   │   ├── __init__.py         # REGISTRY: name -> loader, for CLI dispatch
│   │   ├── base.py             # NIDSDataset protocol: load() -> (X, y, meta)
│   │   ├── nsl_kdd.py          # 41 column names live here, not in a notebook
│   │   ├── unsw_nb15.py
│   │   ├── cic_ids_2018.py
│   │   ├── schema.py           # dtypes, categorical columns, label columns
│   │   └── taxonomy.py         # per-dataset label -> 5-class mapping (§5.2)
│   │
│   ├── preprocessing/
│   │   ├── cleaning.py         # NaN/Inf handling with row-level logging (RULE 6)
│   │   ├── encoders.py         # categorical encoding + scaling, sklearn-compatible
│   │   ├── splitter.py         # StratifiedShuffleSplit wrappers (RULE 3)
│   │   └── alignment.py        # cross-dataset feature mapping (Build Plan §7.2)
│   │
│   ├── features/
│   │   └── engineered.py       # Enhancement 2, behind a single on/off flag
│   │
│   ├── models/
│   │   ├── tabpfn_wrapper.py   # checkpoint pinning, device selection, limit checks
│   │   ├── chunker.py          # stratified chunking, optional overlap
│   │   ├── chunked_ensemble.py # Enhancement 1 orchestration
│   │   ├── aggregation.py      # majority / confidence-weighted / learned
│   │   └── baselines.py        # XGBoost, LightGBM (Build Plan §8)
│   │
│   ├── evaluation/
│   │   ├── metrics.py          # F1, precision, recall, ROC-AUC, confusion matrix
│   │   ├── significance.py     # paired tests — see §3.4 before using
│   │   ├── reporter.py         # timestamped CSV + provenance columns (RULE 5)
│   │   └── plots.py            # confusion heatmap, ROC, PR curves
│   │
│   └── utils/
│       ├── logging.py          # one log format for the whole pipeline
│       └── checkpoint.py       # resume-after-crash support (RULE 4)
│
├── configs/                    # YAML experiment definitions
│   ├── nsl_kdd_baseline.yaml
│   ├── nsl_kdd_enhanced.yaml
│   ├── unsw_enhanced.yaml
│   └── cross_dataset.yaml
│
├── scripts/                    # thin CLI wrappers — argument parsing only, no logic
│   ├── verify_data.py          # Build Plan §1.1 row-count check
│   ├── run_baseline.py
│   ├── run_enhanced.py
│   ├── run_feature_ablation.py
│   ├── run_cross_dataset.py
│   ├── run_classical_baselines.py
│   └── build_comparison_table.py
│
├── tests/                      # mirrors the package layout one-to-one
│   ├── conftest.py             # shared synthetic-dataset fixtures
│   ├── fixtures/               # tiny CSVs, committed to git
│   ├── test_datasets.py
│   ├── test_preprocessing.py
│   ├── test_features.py
│   ├── test_ensemble.py
│   └── test_evaluation.py
│
├── data/                       # gitignored; .gitkeep committed
│   ├── raw/{nsl-kdd,unsw-nb15,cic-ids-2018}/
│   ├── interim/                # cleaned, before feature engineering
│   └── processed/              # model-ready matrices
│
├── reports/                    # gitignored except README.md and .gitkeep
│   ├── figures/   checkpoints/   tables/
│
├── notebooks/
│   ├── 00_eda_unsw.ipynb       # the existing EDA notebook, relocated
│   ├── 01_reproduction.ipynb
│   ├── 02_enhancement.ipynb
│   ├── 03_cross_dataset.ipynb
│   └── 04_results_summary.ipynb
│
└── docs/
    ├── REPRODUCIBILITY.md
    ├── CONTRIBUTIONS.md
    ├── FUTURE_WORK.md
    └── feature_design/
        ├── TabPFN-NIDS_Build_Plan.md
        └── design-setup.md     # this file
```

### 4.3 Why each deviation from Build Plan §0.1 earns its place

| Change | Reason |
|---|---|
| `src/` → `tabpfn_nids/` + editable install | Removes every `sys.path` hack; notebooks, tests and scripts import identically |
| `datasets/` subpackage with a registry | Build Plan §6.1 and §7.1 add two more datasets; a registry means adding a file, not editing an `if`/`else` |
| `preprocessing/` split four ways | §7.2 alignment is called out in the plan itself as "a non-trivial data engineering task"; it needs its own module and its own tests |
| `taxonomy.py` | The 5-class decision (§3.2) requires an explicit, reviewable, testable mapping per dataset. This is a research artefact, not an implementation detail |
| `configs/*.yaml` | Guide §7 installs `pyyaml` and never uses it. Configs turn "try chunk_size=5000" into a file edit, and give reviewers something concrete to read |
| `data/raw|interim|processed` | By Phase 7 there are ~9 CSVs in play; flat naming makes "which one is cleaned?" a live bug source |
| `utils/checkpoint.py` | RULE 4 wants resumability; that needs one shared helper, not per-script logic |
| `tests/fixtures/` + `conftest.py` | Every phase's tests need a tiny synthetic dataset — build the fixture once |
| `.gitkeep` in gitignored dirs | Without it a fresh clone has no `data/` or `reports/`, and every script fails on first write — exactly what §13.1 is meant to catch |
| `Makefile` | Makes the README's "3 commands" promise (§10.1) literally true |

### 4.4 Module contracts

Each module owns one thing. Keeping these boundaries is what makes the structure scalable.

| Module | Owns | Must not |
|---|---|---|
| `datasets/*` | Reading raw files, assigning column names, returning a tidy DataFrame | Scale, encode, split, or drop rows |
| `preprocessing/cleaning` | NaN/Inf policy, with a logged count of every row and cell affected | Silently drop anything (RULE 6) |
| `preprocessing/encoders` | Categorical encoding, numeric scaling, fitted **on train only** | See test data |
| `preprocessing/splitter` | Stratified splits with a fixed seed | Know which model consumes them |
| `features/engineered` | Adding derived columns, toggled by one flag | Modify existing columns in place |
| `models/tabpfn_wrapper` | Checkpoint pinning, device choice, limit validation, clear errors | Know about chunking |
| `models/chunker` | Producing stratified chunks | Call TabPFN |
| `models/chunked_ensemble` | Orchestration and checkpointing | Define the aggregation maths |
| `models/aggregation` | Turning per-chunk probabilities into one prediction | Know where chunks came from |
| `evaluation/*` | Metrics, provenance, figures | Change predictions |

The rule that matters most: **preprocessing is fitted on training data only.** The existing EDA notebook computes its correlations and its `KBinsDiscretizer` on `train + test` concatenated. That is fine for exploration and wrong for the pipeline; `encoders.py` must not repeat it.

---

## 5. Cross-cutting design decisions

### 5.1 Configuration and reproducibility

Every experiment is defined by a YAML file in `configs/` and produces one row in a timestamped CSV in `reports/`. Per RULE 5, each row carries: seed, hostname, platform, CPU count, torch version and backend, **tabpfn version and checkpoint filename**, dataset name and row counts, git commit, wall-clock runtime, and every metric. `tabpfn_nids/config.py` already implements this capture and is the only module that touches `Path`.

### 5.2 The 5-class taxonomy across three datasets

NSL-KDD defines the five classes natively. UNSW-NB15 and CIC-IDS-2018 do not, so `taxonomy.py` must hold an explicit mapping. That mapping is **lossy and arguable** — UNSW's "Generic" (attacks against block ciphers) has no clean NSL-KDD counterpart, and CIC's "Infiltration" spans several.

**Recommendation:** for the cross-dataset experiment in Enhancement 3, report **binary transfer as the primary result** and the mapped 5-class as secondary with the mapping printed in full. Claiming a clean 5-class transfer across datasets whose taxonomies were never designed to align would not survive questioning.

### 5.3 Device policy

`MAX_CPU_SAMPLES = 1000` (§2.2) means CPU is not a viable default. The wrapper should select MPS when available, fall back to CPU with an explicit warning, and record the device actually used in the results CSV. **TODO: confirm from the benchmark in §2.4 whether MPS is actually faster than CPU for this workload on an M1 Air** — for small transformer inference it is not always so, and the honest choice is whichever the measurement supports.

### 5.4 Notebooks stay thin

Notebooks import the package, call it, and plot. No pipeline logic lives in a notebook. This is what makes §13.3's "pre-run notebooks as a demo backup" safe: the notebook is a view over tested code, not a second implementation.

---

## 6. Migration from the current state

Current contents: `src/{data_pipeline,features,models,evaluation}/__init__.py`, `src/config.py`, `tests/__init__.py`, empty `scripts/` and `notebooks/`, `reports/{figures,checkpoints}/`, and a working `venv/`.

| # | Step | Notes |
|---|---|---|
| 1 | `git mv src tabpfn_nids`, add the subpackages from §4.2 | `config.py` moves unchanged |
| 2 | Write `pyproject.toml`, run `pip install -e .` | Verify `python -c "import tabpfn_nids"` from `notebooks/` |
| 3 | Add `.gitkeep` to `data/{raw,interim,processed}` and `reports/{figures,checkpoints,tables}` | Required for the fresh-clone test |
| 4 | Move root clutter into place | `unsw-nb15-eda.ipynb` → `notebooks/00_eda_unsw.ipynb`; `Project-Phase-I-template.pptx.pdf`, `TabPFN-NIDS_Project_Guide.md`, `architecture_diagram.svg` → `docs/` |
| 5 | Update `smoke_test.py` to pin `model_path` | Without this it fails on a clean machine (§2.3) |
| 6 | Freeze `requirements.txt` with exact pins | Build Plan §0.2, still outstanding |
| 7 | Add `.gitignore` entries for `venv/`, `.DS_Store`, `data/`, `reports/*.csv` | `.DS_Store` files are present in the tree today |
| 8 | Commit as `Phase 0.1: package structure and packaging` | |

---

## 7. Decisions needed before building

Seven items. The first three change code that gets written on day one.

| # | Decision | Recommendation |
|---|---|---|
| 1 | Binary, 5-class, or both? (§3.2) | **Both — binary primary, 5-class secondary** |
| 2 | Pin `tabpfn==8.5.0` + v2 checkpoint, or downgrade to 2.2.1? (§3.1) | **Keep 8.5.0, pin the checkpoint** |
| 3 | UNSW cleaned partition (175K) or full release (2.5M)? (§3.3) | **Cleaned partition; correct the deck's row count** |
| 4 | Disjoint or overlapping chunks? (§3.6) | **Disjoint by default, overlap as a parameter** |
| 5 | Wilcoxon over ≥10 seeds, or mean ± std over 5? (§3.4) | **Mean ± std over 5 seeds** given the time available |
| 6 | Project title (§3.5) | **"Scaling Tabular Foundation Models for NIDS"** |
| 7 | Package name | **`tabpfn_nids`** |

Aggregation strategy, chunk size and the engineered feature list are deferred: chunk size follows from the benchmark in §2.4, and the other two are configurable and can be chosen at Phase 3 and Phase 4 respectively.

---

## 8. What this document does not cover

- Slide-deck rewrites (Build Plan Phase 12)
- The content of `REPRODUCIBILITY.md` (Build Plan §10.2)
- Kaggle download mechanics — `kaggle` CLI is installed and credentials are present at `~/.kaggle/kaggle.json`, but **no dataset has been downloaded yet**; `data/` contains only `README.md`
- Exact hyperparameters for the classical baselines (Build Plan §8.1)
