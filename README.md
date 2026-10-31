# W3 task · Tawseel: an honest baseline for failed COD deliveries

DSC Summer Internship 2026 · AI & Deep Tech track · Week 3 · expected time 3–4 hours

## Scenario

Sherif Mansour wants a first failure-risk model for cash-on-delivery parcels, but he has been
burned before by a vendor who showed him "99% accuracy" that vanished in production. Your job
is the opposite: a modest model he can believe. You get a synthetic parcel table with the usual
traps in it. Build point-in-time customer history, an honest time split, a full preprocessing
pipeline, and a comparison with a dummy baseline.

## Set up

1. Open this starter (`dsc26-ai-w03-starter`) on GitHub, press **Use this template → Create a new
   repository**, owner your own account, name exactly **`dsc26-ai-w03`**.
2. Clone it, then `uv sync` and `uv run pytest`. The public tests fail until you write the code.

Versions: Python 3.12, numpy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1 (scipy 1.18.1), pytest 9.1.1.
**No other dependencies**: the grader has nothing else installed.

## Layout

```
dsc26-ai-w03/
├── pyproject.toml     # provided, do not change the dependencies
├── uv.lock            # provided
├── tawseel/
│   ├── __init__.py
│   ├── data.py        # provided by DSC, do not edit (the grader uses its own copy)
│   └── model.py       # you write
├── tests/test_public.py
├── RESULTS.md         # you write: your baseline table and three sentences for Sherif
└── AI_USAGE.md        # you write
```

## The data

`tawseel.data.make_parcels(n=4000, seed=0)` returns one row per parcel with columns
`parcel_id`, `customer_id`, `merchant_id`, `area`, `created_at` (datetime), `promised_slot`
(`morning`/`afternoon`/`evening`), `cod_amount` (EGP, some missing), `n_items`, `address_text`
(Arabic, English or Franco, sometimes empty), `courier_notes` (written after the attempt),
`delivered_at` (empty when the delivery failed), and the label `failed` (0/1). Read `data.py`
to see exactly how it is made. The hidden tests call it with other seeds and sizes.

## What to build in `tawseel/model.py`

```python
TARGET = "failed"
FEATURE_COLUMNS: list[str]            # raw columns the model may use at decision time

def add_history_features(df: pd.DataFrame) -> pd.DataFrame: ...
def time_split(df: pd.DataFrame, time_col: str = "created_at",
               test_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]: ...
def build_pipeline(seed: int = 0) -> sklearn.pipeline.Pipeline: ...
def evaluate(model, test_df: pd.DataFrame) -> dict: ...
def compare_to_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame, seed: int = 0) -> dict: ...
```

- `FEATURE_COLUMNS`: the raw columns known when the parcel is created. It must not contain
  anything known only after the delivery attempt, the label, or the row identifier. The starter
  lists every column; remove the ones that do not belong.
- `add_history_features(df)`: needs `customer_id`, `created_at` and `failed`. Returns a copy
  (never modifies the input) with the same index and row order, plus two columns.
  `customer_prior_parcels` (int) is the number of the same customer's parcels with `created_at`
  strictly earlier than this row. `customer_prior_fail_rate` (float) is the share of those that
  failed, or NaN when there are none. Parcels with the same timestamp do not count as prior to
  each other. The result must not depend on the input row order. You may assume the index is
  unique (it is not always 0..n-1).
- `time_split(df, time_col, test_fraction)`: `ValueError` unless `0 < test_fraction < 1`. Sort by
  time; let `n_test = ceil(len(df) * test_fraction)`; the boundary is the timestamp of the first
  row of the last `n_test` rows. Every row at or after the boundary is test, every row before it
  is train. So all rows sharing the boundary timestamp go to test, and no timestamp appears on
  both sides. Keep the original index on both parts (the order of rows inside each part is not
  checked).
- `build_pipeline(seed)`: returns an **unfitted** `Pipeline` whose input is a DataFrame containing
  at least `FEATURE_COLUMNS` plus the two history columns; any other column must be ignored. The
  final step must support `predict_proba`. It must not fail on unseen categories, missing numbers
  (in any numeric column), missing categories (`None`/NaN) or empty address strings, whether it
  predicts one row or many. Same seed and same data must give identical predictions.
- `evaluate(model, test_df)`: returns `{"roc_auc", "pr_auc", "brier", "base_rate", "n"}` using
  `predict_proba(test_df)[:, 1]` against `test_df["failed"]`. Use `roc_auc_score`,
  `average_precision_score` and `brier_score_loss`.
- `compare_to_baseline(train_df, test_df, seed)`: fits `build_pipeline(seed)` and
  `DummyClassifier(strategy="prior")` on the training data only, evaluates both on the test
  data, and returns `{"model": {...}, "dummy": {...}, "lift_pr_auc": model_pr_auc - dummy_pr_auc}`.

With this generator the failure rate is around 21–24%, and a straightforward logistic-regression
pipeline with history features reaches a PR-AUC roughly 0.10–0.18 above the dummy on the last
20%. That gap is the realistic size of the win. **If you see a ROC AUC near 0.99, you have a
leak**; the grader treats a ROC AUC above 0.9 on this data as a leak.

`RESULTS.md` holds your baseline table (dummy, any rule you tried, your model) and three plain
sentences you would send Sherif. It is read by your Track Mentor, not graded by tests.

## How grading works

Push, then press **Submit for grading** on DSC Hex. The grader clones that commit, replaces
`tawseel/data.py` with DSC's copy, and runs the hidden tests on Python 3.12 with the versions
above, network off, 120 seconds per test and 8 minutes for the whole run. Score 0–100 with
feedback per test name; each test is all or nothing. Up to five runs per task per day; the best
attempt counts.

For the pipeline and evaluation tests (8 to 14) the grader builds the history columns and the
time split with its own correct implementation, so a bug in `add_history_features` costs you
only the history tests, not everything.

| Test name | What it checks | Points |
|---|---|---|
| `history_is_point_in_time` | Shuffling input rows gives identical features; appending later parcels leaves earlier rows unchanged | 12 |
| `history_ties_are_not_prior` | Parcels of one customer at the same minute do not see each other | 6 |
| `history_first_parcel` | Prior count 0 and NaN rate for each customer's first parcel | 5 |
| `history_does_not_mutate_input` | Input unchanged; index and order preserved | 4 |
| `time_split_sizes_and_order` | `ceil` rule, no overlap, every row in exactly one part, bad fraction raises | 8 |
| `time_split_boundary_ties` | All rows sharing the boundary timestamp go to test | 6 |
| `feature_columns_exclude_post_event_data` | No `courier_notes`, `delivered_at`, `failed`, `parcel_id` | 10 |
| `pipeline_ignores_extra_and_works_without_leaks` | Fits and predicts with only `FEATURE_COLUMNS` plus history; extra columns ignored | 6 |
| `pipeline_survives_production_inputs` | Unseen area and merchant, NaN numbers, missing category, empty address | 8 |
| `preprocessing_is_inside_the_pipeline` | Unfitted on return; one row alone predicts the same as inside a batch | 4 |
| `evaluate_matches_sklearn` | Metrics equal scikit-learn's on a fixed fitted model | 8 |
| `beats_dummy_on_unseen_seeds` | PR-AUC lift ≥ 0.05 on 3 hidden seeds of 4,000 parcels, dummy fitted on train only, no leak | 12 |
| `brier_not_worse_than_dummy` | Model Brier score ≤ dummy Brier score on those seeds | 5 |
| `deterministic_with_seed` | Same seed, same data, identical probabilities | 6 |

Late policy: ×1.0 within 72 hours of the session starting, ×0.6 in the following 7 days, ×0.5
after that until Thu 3 Dec 23:59. During W5 (7–13 Nov) you can resubmit at ×1.0.

## `AI_USAGE.md`

Fill in the template: the tools you used and for what; the AI-suggested code you rejected or
fixed in `add_history_features` or `time_split`, with the test that showed the problem; one
sentence on how you checked that your model has no leak. Your Track Mentor reads it.

If you are stuck for more than an hour, ask in the AI track channel on Discord or ask Hexa
rather than spending grader runs.
