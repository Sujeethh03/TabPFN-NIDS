# Results comparison

Mean +/- standard deviation across seeds. Standard deviation is 0.000 where only one seed was run.

| Metric | ablation | baseline | comparison | enhanced | feature |
|---|---|---|---|---|---|
| accuracy | 0.7726 ± 0.0135 | 0.7618 ± 0.0214 | n/a | 0.7880 ± 0.0000 | 0.7705 ± 0.0134 |
| precision | 0.9274 ± 0.0025 | 0.9395 ± 0.0286 | n/a | 0.9838 ± 0.0000 | 0.9335 ± 0.0117 |
| recall | 0.6513 ± 0.0238 | 0.6219 ± 0.0233 | n/a | 0.6386 ± 0.0000 | 0.6424 ± 0.0162 |
| f1_score | 0.7650 ± 0.0173 | 0.7482 ± 0.0236 | n/a | 0.7745 ± 0.0000 | 0.7610 ± 0.0152 |
| roc_auc | 0.9584 ± 0.0024 | 0.9572 ± 0.0060 | n/a | 0.9693 ± 0.0000 | 0.9634 ± 0.0024 |

| Setting | ablation | baseline | comparison | enhanced | feature |
|---|---|---|---|---|---|
| seeds | 5 | 6 | 3 | 1 | 2 |
| context rows | mixed | mixed | mixed | 7413 | 29073 |
| test rows | 1000 | mixed | mixed | 500 | 1000 |
| features | 168 | 122 | mixed | 122 | mixed |
| chunks | 3 | - | mixed | 3 | 3 |
