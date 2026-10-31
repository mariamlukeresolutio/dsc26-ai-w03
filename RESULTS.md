# Results: Tawseel Failed-Delivery Baseline

## Baseline table

Evaluation uses `make_parcels(n=4000, seed=0)`. The dataset is split chronologically, with the last 20% of parcels by creation time reserved for testing. All metrics are calculated on the same held-out test set of 800 parcels.

| Model               | ROC AUC | PR-AUC |  Brier | Base rate |
| ------------------- | ------: | -----: | -----: | --------: |
| Dummy (prior)       |  0.5000 | 0.2200 | 0.1717 |    0.2200 |
| Rule-based baseline |  0.6532 | 0.3564 | 0.1625 |    0.2200 |
| Logistic regression |  0.6295 | 0.3462 | 0.1654 |    0.2200 |

**Training parcels:** 3,200
**Test parcels:** 800
**Logistic regression PR-AUC lift over dummy:** +0.1262
**Rule-based PR-AUC lift over dummy:** +0.1364

## Three sentences for Sherif

The model uses information available when a parcel is created, including customer delivery history, to estimate the probability of delivery failure. On the synthetic test set, logistic regression achieved a PR-AUC of 0.3462 compared with 0.2200 for the dummy baseline, while the rule-based baseline achieved 0.3564. Although the rule-based approach performed slightly better on this test split, all results are based on synthetic data and require validation on real delivery records before operational use.