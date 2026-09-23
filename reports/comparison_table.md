# Comparison table

TabPFN v2 on NSL-KDD, binary classification (0 = normal, 1 = attack).
Values are mean ± standard deviation across seeds; std is 0.0000 where
only one seed was run.

| Metric | Vanilla TabPFN (10K subsample) | Enhanced (Chunked Ensemble) | Enhanced + Feature Engineering |
|---|---|---|---|
| **accuracy** | 0.7618 ± 0.0214 | 0.7880 ± 0.0000 | 0.7610 ± 0.0000 |
| **precision** | 0.9395 ± 0.0286 | 0.9838 ± 0.0000 | 0.9253 ± 0.0000 |
| **recall** | 0.6219 ± 0.0233 | 0.6386 ± 0.0000 | 0.6309 ± 0.0000 |
| **f1_score** | 0.7482 ± 0.0236 | 0.7745 ± 0.0000 | 0.7503 ± 0.0000 |
| **roc_auc** | 0.9572 ± 0.0060 | 0.9693 ± 0.0000 | 0.9617 ± 0.0000 |

## Run settings

| Setting | Vanilla TabPFN (10K subsample) | Enhanced (Chunked Ensemble) | Enhanced + Feature Engineering |
|---|---|---|---|
| seeds | 6 | 1 | 1 |
| context rows | mixed | 7413 | 29073 |
| test rows | mixed | 500 | 1000 |
| features | 122 | 122 | 168 |
| chunks | - | 3 | 3 |
| n_estimators | 2 | 2 | 2 |
| runtime (s) | 410.7 | 279.6 | 952.8 |

## Reading the deltas

The baseline's F1 standard deviation across seeds is **0.0236** (2.36 pp). A difference between arms smaller than that is within seed-to-seed noise and should not be reported as an improvement.
