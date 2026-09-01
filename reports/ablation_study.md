# Ablation study

Generated 2026-08-31 15:31. Seed 42, NSL-KDD, binary labels.

Answers the question a reviewer will ask: **how do you know these design choices matter?**

## Shared configuration

| Setting | Value |
|---|---|
| seed | 42 |
| chunks per run | 3 (held fixed) |
| test rows | 1,000 (stratified sample) |
| features | 168 |
| engineered features | on |
| n_estimators | 2 |
| device | mps |
| tabpfn | 8.5.0, checkpoint tabpfn-v2-classifier.ckpt |

## A. Context size

Chunk count is held fixed so that context size is the variable under test. Note that this also varies total training data seen; see the caveat below.

| chunk_size | accuracy | precision | recall | f1_score | roc_auc | context rows | runtime (s) |
|---|---|---|---|---|---|---|---|
| 1,000 | 0.7560 | 0.9243 | 0.6221 | 0.7437 | 0.9562 | 3,000 | 39 |
| 2,500 | 0.7780 | 0.9284 | 0.6608 | 0.7721 | 0.9569 | 7,413 | 111 |
| 5,000 | 0.7880 | 0.9301 | 0.6784 | 0.7846 | 0.9602 | 14,538 | 290 |
| 10,000 | 0.7610 | 0.9253 | 0.6309 | 0.7503 | 0.9617 | 29,073 | 2540 |

### Interpretation

Best F1 was 0.7846 at chunk_size=5,000.
Across a 10x range of context size, F1 varied by 0.0409 (4.09 pp).
Runtime rose from 39s to 2540s, a factor of 64.7x.

F1 peaks at chunk_size=5,000 and is *lower* at the 10,000 limit. Defaulting to the maximum context would cost 3.43 pp of F1 and 8.8x the runtime -- the ceiling is not the optimum.

Caveat on the design: chunk count is held fixed, so a larger chunk size also means more total training data. This measures the joint effect of context size and data volume, which is the practical question when choosing chunk_size, but it is not a pure isolation of context length.

## B. Stratified vs random chunking

Both at chunk_size=10,000, 3 chunks, identical in every other respect.

| chunking | accuracy | precision | recall | f1_score | roc_auc | max balance drift |
|---|---|---|---|---|---|---|
| stratified | 0.7610 | 0.9253 | 0.6309 | 0.7503 | 0.9617 | 0.00004 |
| random | 0.7800 | 0.9287 | 0.6643 | 0.7746 | 0.9570 | 0.01190 |

### Per-chunk class balance

| chunking | population rate | per-chunk rates |
|---|---|---|
| stratified | 0.4654 | `[0.46538, 0.46538, 0.46538]` |
| random | 0.4654 | `[0.46879, 0.46321, 0.45351]` |

### Interpretation

Stratified F1 0.7503 vs random 0.7746, a difference of -0.0243 (-2.43 pp).

Class-balance drift is the mechanism, and it differs clearly: 0.00004 stratified against 0.01190 random, a factor of 322x.

Random chunking scored higher than stratified by more than the noise floor. That is unexpected and should be investigated before the result is reported; a single seed is thin evidence either way.

## Limitations

- Single seed (42). The baseline's seed-to-seed F1 spread is 0.0229 (2.29 pp); differences smaller than that are noise.
- 3 chunks per run, not the full 13-chunk partition, and a 1,000-row test sample rather than all 22,544. Both are runtime caps.
- Binary labels only. The stratification ablation is far more informative under multi-class labels, where rare families can be missed entirely by a random chunk.
- `common_service_flag` is capped at 40 categories; uncapped it produces 463 features and exhausts MPS memory.
