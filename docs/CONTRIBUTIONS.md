# Contribution Statement

What in this repository is ours, and what comes from prior work. Written to be
checkable: every claim below can be verified against the code or the results
CSVs.

---

## 1. What is not ours

**TabPFN v2 itself.** Hollmann, N., Müller, S., Purucker, L., Krishnakumar, A.,
Körfer, M., Hoo, S. B., Schirrmeister, R. T., & Hutter, F. (2025). *Accurate
predictions on small data with a tabular foundation model.* Nature, 637(8045),
319–326. The model architecture, its synthetic pretraining prior and the
`tabpfn` package are entirely theirs. We use the published `tabpfn-v2-classifier`
checkpoint unmodified — no fine-tuning, no architectural change.

**The datasets.** NSL-KDD (Tavallaee et al., 2009), UNSW-NB15 (Moustafa & Slay,
2015), CIC-IDS-2018 (Sharafaldin et al., 2018).

**The idea of applying TabPFN to intrusion detection.** Yousfi, S., Kessentini,
Y., & Chibani, A. (2024). *A TabPFN-based intrusion detection system for the
industrial internet of things.* The Journal of Supercomputing, 80. They applied
TabPFN **v1** to Industrial IoT datasets (Edge-IIoTset, WUSTL-IIoT) with small
training sets. We differ in using v2, in targeting general NIDS traffic, and in
addressing the scale limit rather than working within it.

**Standard library components**: scikit-learn's `StratifiedKFold`,
`StratifiedShuffleSplit`, `OneHotEncoder`, `StandardScaler` and metrics.

---

## 2. Enhancement 1 — Stratified chunked ensemble

**Status: implemented and tested; full-scale run not yet completed.**

### The problem

TabPFN v2 accepts at most 10,000 in-context training samples
(`MAX_NUMBER_OF_SAMPLES`, verified by reading `tabpfn/inference_config.py`).
NSL-KDD's training split is 125,973 rows, so a vanilla run must discard 92% of
the available data.

### What we built

`tabpfn_nids/models/chunker.py` and `chunked_ensemble.py`. The training set is
partitioned into stratified chunks that each fit one context; TabPFN runs
independently on each against the same test set; the per-chunk probability
matrices are aggregated.

Two design decisions we made and can defend:

**`StratifiedKFold` rather than repeated `StratifiedShuffleSplit`.** The build
plan allowed either. KFold's folds are mutually exclusive and jointly
exhaustive, so every training row enters exactly one context and none is
duplicated across contexts; repeated ShuffleSplit draws would overlap, using
some rows several times and others never. Verified: 13 chunks × 9,690–9,691
rows = exactly 125,973, with class-balance drift of 0.00005 against a ±2%
tolerance.

**Confidence-weighted aggregation.** Each chunk is weighted by
`mean(max(P(class), axis=1))` across the test set — how decisive that chunk's
context makes the model. A plain average (`majority`) is provided for
comparison.

### Distinguishing this from TabPFN's own ensembling

`tabpfn` exposes `n_estimators`, which ensembles **preprocessing variations over
the same context**. Our chunking ensembles **disjoint subsamples of a dataset
larger than one context**. These are different axes, and the distinction matters
because a reviewer will ask.

To keep the comparison honest, `n_estimators` is held **identical** between the
baseline and enhanced arms. If it differed, the measured delta would reflect
ensemble size rather than chunking.

### Engineering contribution beyond the algorithm

Two MPS failures we diagnosed and fixed, neither documented upstream:

1. **Cross-chunk memory accumulation.** The ensemble died at chunk 4 of 13 with
   `TabPFNMPSOutOfMemoryError`. MPS does not release memory when a Python
   reference is dropped. `TabPFNWrapper.free()` empties the cache between
   chunks; verified over 6 sequential chunks.
2. **Single-call prediction OOM.** Prediction is batched (default 1,000 test
   rows) independently of the above.

---

## 3. Enhancement 2 — Domain-aware feature engineering

**Status: implemented and tested; ablation measured at reduced scope.**

Six derived features in `tabpfn_nids/features/engineered.py`, behind a single
`enabled` flag so the ablation is a controlled comparison on identical splits:

| feature | definition | rationale |
|---|---|---|
| `bytes_ratio` | `src_bytes / (dst_bytes + 1)` | upload/download asymmetry; exfiltration skews high |
| `total_bytes` | `src_bytes + dst_bytes` | flow volume |
| `bytes_per_second` | `total_bytes / (duration + 1)` | throughput; volumetric floods are extreme |
| `is_short_session` | `duration < 5` | scans and probes are short |
| `error_rate_composite` | `serror_rate × srv_serror_rate` | high only when host- and service-level SYN errors coincide — a SYN flood, not one broken service |
| `common_service_flag` | `protocol_type + service + flag` | interactions a per-column encoding cannot express, e.g. `tcp_private_S0` (half-open scan) vs `tcp_http_SF` |

Every denominator is `+1`, so division by zero is impossible, and the output is
**asserted** finite rather than assumed — TabPFN rejects infinities at input
validation, so a silent `Inf` would surface much later and far less clearly.

### An honest constraint

`common_service_flag` has **336** distinct values in NSL-KDD's training split,
giving 463 total features. A 9,691 × 463 context **exhausts MPS memory on a
16 GB M1**, and it OOMs at every test batch size tried down to 100 — the memory
goes to the context, not the test rows, so batching cannot fix it.

The ablation therefore runs with `max_service_flag_categories=40`, giving 168
features. The top 40 combinations cover ~88% of training rows. **This is a
hardware constraint, not a design choice, and the reported Enhancement 2 result
is for the capped variant.** Presenting it as the full interaction feature would
be inaccurate.

---

## 4. Enhancement 3 — Cross-dataset transfer

**Status: not implemented.**

The datasets are downloaded and their schema problems documented (see
REPRODUCIBILITY.md §3.3), but no loader, feature alignment or transfer
experiment exists. It should be reported as future work, not as a result.

---

## 5. Methodological contributions

These are not enhancements to TabPFN, but they are original to this project and
affect how the results should be read.

**Refusing an unreachable significance test.** The build plan specified a
Wilcoxon signed-rank test over 3 seeds. Wilcoxon's smallest attainable
two-sided p-value at n=3 is **0.25** — significance was impossible before any
data existed. `evaluation/significance.py` computes that floor and returns
`underpowered=True` rather than a p-value it could never have rejected,
reporting mean ± std instead.

**Establishing the noise floor before claiming effects.** Three baseline seeds
give F1 std = **0.0229**. Any single-seed delta below ~2.3pp is not evidence.
This is stated wherever a delta is reported.

**Correcting three factual errors in the project's own planning documents**,
each verified against the data:

| claim | reality |
|---|---|
| UNSW-NB15 has ~2.5M rows, 49 features | 175,341 + 82,332 rows, 36 columns (cleaned partition) |
| NSL-KDD's test set has unseen *categorical values* | It does not — they are a strict subset. The real property is 17 unseen *attack types*. |
| "Zero-shot" intrusion detection | TabPFN takes labelled in-context examples; this is in-context learning. "Zero-shot" is accurate only for Enhancement 3. |

**Documenting the checkpoint trap.** `tabpfn` 8.5.0 defaults to a gated
repository holding `tabpfn_3` — a *different, newer* model than the Nature 2025
publication. Any project claiming to reproduce TabPFN v2 while loading the
default checkpoint is not reproducing what it says it is. We pin the v2
checkpoint and assert it in a test.

**Empirical runtime characterisation.** Measured, not assumed: TabPFN's cost is
almost entirely in `predict`, not `fit`; prediction cost grows super-linearly in
context size (10× context → 12.3× time); MPS is 3.8–5.3× faster than CPU; and
accuracy *saturates* around 5,000 context rows, so defaulting `chunk_size` to
the 10,000 ceiling costs 2.9× for no measured gain.

---

## 6. Summary

| Component | Origin |
|---|---|
| TabPFN v2 model and weights | Hollmann et al., Nature 2025 |
| Applying TabPFN to intrusion detection | Yousfi et al. 2024 (v1, IIoT) |
| Datasets | Tavallaee 2009, Moustafa 2015, Sharafaldin 2018 |
| Stratified chunked ensemble | **ours** |
| Confidence-weighted chunk aggregation | **ours** |
| MPS memory fixes for sequential inference | **ours** |
| Six domain-aware NIDS features | **ours** |
| Ablation harness and controlled toggle | **ours** |
| Underpowered-test detection | **ours** |
| Runtime characterisation on Apple Silicon | **ours** |
| Cross-dataset transfer | **not done** |
