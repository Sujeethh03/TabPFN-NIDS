# Future Work

Ordered by how much they would strengthen the project, most valuable first.
The first three are unfinished work from this phase rather than new directions.

---

## Immediate — completing what is started

### 1. Threshold tuning (highest value for the least effort)

The baseline shows **ROC-AUC 0.9555 against F1 0.7539**. The model ranks
attacks well; the default 0.5 decision threshold simply places the operating
point badly for NSL-KDD's distribution shift. Precision 0.93 against recall
0.63 is the visible symptom.

Selecting a threshold on a validation split — by maximising F1, or by fixing an
operational false-positive budget, which is how a real NIDS is tuned — would
likely lift F1 by several points with **no model change at all**. Nothing else
on this list has a comparable effort-to-payoff ratio.

### 2. Enhancement 1 at full scale

The 13-chunk run crashed on the MPS memory bug before completing. The fix is in
and verified on a smaller configuration, but the headline Enhancement 1 number
against the baseline has not been produced. This is the project's stated main
contribution and should be finished first.

### 3. Enhancement 3 — cross-dataset transfer

Not implemented. Requires a CIC-IDS-2018 loader handling four known defects
(84-column outlier file, embedded header rows, CRLF labels, `Inf` values) and a
hand-built feature alignment across three genuinely different schemas — 122,
36 and ~80 columns. The alignment is the hard part and deserves its own tests.

### 4. Classical baselines

XGBoost and LightGBM on identical splits. Reviewers expect an anchor, and
"TabPFN F1 vs XGBoost F1 on the same 5,000 test rows" is a stronger claim than
a TabPFN number alone. `lightgbm` is already installed.

---

## Near-term extensions

### 5. Multi-class classification

Currently binary. The five-class NSL-KDD taxonomy (normal/dos/probe/r2l/u2r)
would be richer, but note the obstacle before committing: **U2R is about 0.04%
of the training split — roughly 52 rows of 125,973** (`buffer_overflow` 30,
`rootkit` 10, `loadmodule` 9, `perl` 3). Macro-F1 would be dominated by a class
with almost no support, and that needs explaining rather than hiding. UNSW-NB15
has exactly 10 classes, sitting precisely on TabPFN's `MAX_NUMBER_OF_CLASSES`
with no headroom for an "unknown" bucket.

### 6. Chunk-count and chunk-size ablation

Measurement showed accuracy **saturating** around 5,000 context rows (0.969 at
5,000 vs 0.961 at 10,000) while cost grows super-linearly. The obvious study —
does the ensemble's benefit saturate after *k* chunks, and is `chunk_size=5000`
strictly better than 10,000? — is cheap and directly informs the design.

### 7. Overlapping chunks

Currently disjoint. `stratified_chunk` has an `overlap_ratio` hook that is not
yet exercised. Overlap trades compute for variance reduction; whether it pays
is an empirical question.

### 8. Rare-class-aware chunking

Stratification currently preserves the *binary* balance. For multi-class work,
chunks should preserve the rare attack families specifically — a chunk with no
`u2r` examples cannot recognise `u2r` at all.

---

## Research directions (Phase II)

### 9. Explainability

SHAP over TabPFN predictions. Interpretability is a compliance requirement in
enterprise NIDS, and per-chunk confidence weights already provide a natural
hook: which chunks were confident about which test rows, and why.

### 10. Adversarial robustness

Test against evasion — traffic perturbed to cross the decision boundary while
preserving attack semantics. In-context learning may behave quite differently
from a trained classifier here, which is itself the interesting question.

### 11. Real-time streaming

Wrap the ensemble in a streaming pipeline for online inference. The measured
throughput (~0.2–0.9 s per test row on an M1, depending on configuration) is far
from line-rate, so this needs either batching strategy work or different
hardware, and that gap should be stated honestly rather than glossed.

### 12. Federated variant

Distribute chunks across nodes so organisations contribute context without
sharing raw traffic. The chunked design maps onto this unusually cleanly: chunks
are already independent and only probabilities need to be combined.

### 13. Domain adaptation

If Enhancement 3 shows a large transfer gap, a small adaptation layer between
source and target feature spaces is the natural follow-up.

---

## Infrastructure

### 14. Checkpointing and resumability

`utils/checkpoint.py` is a placeholder. A 13-chunk run that dies at chunk 12
currently loses everything — as one did. Per-chunk probability matrices should
be persisted so a run can resume.

### 15. Hardware beyond one M1 Air

Every timing here is from a single 16 GB M1 Air, and MPS memory is the binding
constraint on Enhancement 2. A CUDA machine would remove the 463-feature ceiling
and let the uncapped `common_service_flag` be evaluated as designed.
