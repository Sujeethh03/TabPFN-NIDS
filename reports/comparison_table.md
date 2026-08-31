# Comparison table

TabPFN v2 on NSL-KDD, binary classification (0 = normal, 1 = attack).
Values are mean ± standard deviation across seeds; std is 0.0000 where
only one seed was run.

| Metric | Vanilla TabPFN (10K subsample) | Enhanced (Chunked Ensemble) | Enhanced + Feature Engineering |
|---|---|---|---|
| **accuracy** | 0.7653 ± 0.0212 | not run | not run |
| **precision** | 0.9347 ± 0.0269 | not run | not run |
| **recall** | 0.6318 ± 0.0213 | not run | not run |
| **f1_score** | 0.7539 ± 0.0229 | not run | not run |
| **roc_auc** | 0.9555 ± 0.0040 | not run | not run |

## Run settings

| Setting | Vanilla TabPFN (10K subsample) | Enhanced (Chunked Ensemble) | Enhanced + Feature Engineering |
|---|---|---|---|
| seeds | 3 | — | — |
| context rows | 10000 | — | — |
| test rows | 5000 | — | — |
| features | 122 | — | — |
| chunks | - | — | — |
| n_estimators | 2 | — | — |
| runtime (s) | 295.8 | — | — |

## Not yet run

These arms have no results in `reports/` and are shown as *not run* rather than omitted:

- Enhanced (Chunked Ensemble)
- Enhanced + Feature Engineering

## Reading the deltas

The baseline's F1 standard deviation across seeds is **0.0229** (2.29 pp). A difference between arms smaller than that is within seed-to-seed noise and should not be reported as an improvement.
