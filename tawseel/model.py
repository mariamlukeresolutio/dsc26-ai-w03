from __future__ import annotations

import math

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler,
)


TARGET = "failed"

# Only information available when a parcel is created.
FEATURE_COLUMNS: list[str] = [
    "customer_id",
    "merchant_id",
    "area",
    "created_at",
    "promised_slot",
    "cod_amount",
    "n_items",
    "address_text",
]

HISTORY_COLUMNS = [
    "customer_prior_parcels",
    "customer_prior_fail_rate",
]


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add point-in-time customer history without modifying the input."""

    result = df.copy()

    work = df[["customer_id", "created_at", TARGET]].copy()
    work["_position"] = np.arange(len(work))

    # Parcels with identical customer and timestamp are treated as a group.
    # This prevents tied timestamps from counting as prior to one another.
    history = (
        work.groupby(
            ["customer_id", "created_at"],
            dropna=False,
            sort=False,
        )
        .agg(
            group_count=(TARGET, "size"),
            group_failures=(TARGET, "sum"),
        )
        .reset_index()
    )

    history = history.sort_values(
        ["customer_id", "created_at"],
        kind="mergesort",
        na_position="first",
    )

    customer_groups = history.groupby(
        "customer_id",
        sort=False,
        dropna=False,
    )

    # Exclude the current timestamp group.
    history["customer_prior_parcels"] = (
        customer_groups["group_count"].cumsum()
        - history["group_count"]
    )

    history["customer_prior_failures"] = (
        customer_groups["group_failures"].cumsum()
        - history["group_failures"]
    )

    history["customer_prior_fail_rate"] = (
        history["customer_prior_failures"]
        / history["customer_prior_parcels"].replace(0, np.nan)
    )

    mapped = work.merge(
        history[
            [
                "customer_id",
                "created_at",
                "customer_prior_parcels",
                "customer_prior_fail_rate",
            ]
        ],
        on=["customer_id", "created_at"],
        how="left",
        sort=False,
        validate="many_to_one",
    )

    mapped = mapped.sort_values("_position", kind="mergesort")

    result["customer_prior_parcels"] = (
        mapped["customer_prior_parcels"].to_numpy(dtype=np.int64)
    )

    result["customer_prior_fail_rate"] = (
        mapped["customer_prior_fail_rate"].to_numpy(dtype=float)
    )

    return result


def time_split(
    df: pd.DataFrame,
    time_col: str = "created_at",
    test_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split chronologically while keeping identical timestamps together."""

    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")

    if len(df) == 0:
        return df.copy(), df.copy()

    n_test = math.ceil(len(df) * test_fraction)

    ordered = df.sort_values(
        time_col,
        kind="mergesort",
    )

    boundary = ordered[time_col].iloc[-n_test]

    train_df = df.loc[df[time_col] < boundary].copy()
    test_df = df.loc[df[time_col] >= boundary].copy()

    return train_df, test_df


def _date_features(values):
    """Extract numeric calendar features from a datetime column."""

    if isinstance(values, pd.DataFrame):
        values = values.iloc[:, 0]

    dates = pd.to_datetime(values, errors="coerce")

    if not isinstance(dates, pd.Series):
        dates = pd.Series(dates)

    return np.column_stack(
        [
            dates.dt.year.to_numpy(dtype=float),
            dates.dt.month.to_numpy(dtype=float),
            dates.dt.day.to_numpy(dtype=float),
            dates.dt.dayofweek.to_numpy(dtype=float),
            dates.dt.hour.to_numpy(dtype=float),
        ]
    )


def _clean_text(values):
    """Convert missing address values into strings."""

    if isinstance(values, pd.DataFrame):
        values = values.iloc[:, 0]

    if isinstance(values, pd.Series):
        return values.fillna("").astype(str).to_numpy()

    values = np.asarray(values, dtype=object).reshape(-1)

    return np.array(
        [
            "" if pd.isna(value) else str(value)
            for value in values
        ],
        dtype=object,
    )


def build_pipeline(seed: int = 0) -> Pipeline:
    """Return an unfitted, leakage-conscious classification pipeline."""

    numeric_columns = [
        "cod_amount",
        "n_items",
        "customer_prior_parcels",
        "customer_prior_fail_rate",
    ]

    categorical_columns = [
        "merchant_id",
        "area",
        "promised_slot",
    ]

    date_pipeline = Pipeline(
        steps=[
            (
                "extract_date",
                FunctionTransformer(
                    _date_features,
                    validate=False,
                ),
            ),
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "impute",
                SimpleImputer(
                    strategy="constant",
                    fill_value="__MISSING__",
                ),
            ),
            (
                "encode",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    address_pipeline = Pipeline(
        steps=[
            (
                "clean",
                FunctionTransformer(
                    _clean_text,
                    validate=False,
                ),
            ),
            (
                "vectorize",
                HashingVectorizer(
                    analyzer="char",
                    ngram_range=(2, 5),
                    n_features=2**12,
                    alternate_sign=False,
                    norm="l2",
                ),
            ),
        ]
    )

    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
            ("date", date_pipeline, ["created_at"]),
            ("address", address_pipeline, "address_text"),
        ],
        remainder="drop",
    )

    classifier = LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=seed,
    )

    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("classifier", classifier),
        ]
    )


def evaluate(model, test_df: pd.DataFrame) -> dict:
    """Evaluate predicted failure probabilities on a test DataFrame."""

    y_true = test_df[TARGET].to_numpy()
    probabilities = model.predict_proba(test_df)[:, 1]

    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(
            average_precision_score(y_true, probabilities)
        ),
        "brier": float(
            brier_score_loss(y_true, probabilities)
        ),
        "base_rate": float(np.mean(y_true)),
        "n": int(len(test_df)),
    }


def _rule_based_probabilities(
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
) -> np.ndarray:
    """Create probabilities using transparent business rules."""

    risky_areas = {
        "Shubra",
        "Imbaba",
        "Ain Shams",
        "Helwan",
        "Faisal",
        "Haram",
        "Matareya",
    }

    vague_terms = [
        "behind",
        "near",
        "gamb",
        "odam",
        "ورا",
        "قصاد",
        "جنب",
    ]

    def calculate_score(data):
        score = np.zeros(len(data), dtype=float)

        address = (
            data["address_text"]
            .fillna("")
            .astype(str)
            .str.lower()
        )

        vague_address = address.apply(
            lambda text: any(term in text for term in vague_terms)
        )

        empty_address = address.str.strip().eq("")

        score += 0.9 * vague_address.to_numpy()
        score += 0.6 * empty_address.to_numpy()

        evening = (
            data["promised_slot"]
            .eq("evening")
            .fillna(False)
            .to_numpy()
        )

        score += 0.35 * evening

        risky_area = (
            data["area"]
            .isin(risky_areas)
            .fillna(False)
            .to_numpy()
        )

        score += 0.5 * risky_area

        cod = pd.to_numeric(
            data["cod_amount"],
            errors="coerce",
        ).fillna(500)

        score += 0.55 * np.log1p(cod.to_numpy() / 500)

        history = pd.to_numeric(
            data["customer_prior_fail_rate"],
            errors="coerce",
        ).fillna(0)

        score += history.to_numpy()

        return score

    train_scores = calculate_score(train_df)
    test_scores = calculate_score(test_df)

    # Calibrate the intercept using training data only.
    target_rate = float(train_df[TARGET].mean())

    target_rate = np.clip(
        target_rate,
        1e-6,
        1 - 1e-6,
    )

    low = -50.0
    high = 50.0

    for _ in range(100):
        middle = (low + high) / 2

        probabilities = 1 / (
            1 + np.exp(-(middle + train_scores))
        )

        if probabilities.mean() < target_rate:
            low = middle
        else:
            high = middle

    intercept = (low + high) / 2

    return 1 / (
        1 + np.exp(-(intercept + test_scores))
    )


def compare_to_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int = 0,
) -> dict:
    """Compare logistic regression, dummy, and rule-based baselines."""

    y_true = test_df[TARGET].to_numpy()

    # Logistic regression
    model = build_pipeline(seed)
    model.fit(train_df, train_df[TARGET])

    model_results = evaluate(model, test_df)

    # Dummy baseline
    dummy = DummyClassifier(strategy="prior")

    dummy.fit(
        np.zeros((len(train_df), 1)),
        train_df[TARGET],
    )

    dummy_probabilities = dummy.predict_proba(
        np.zeros((len(test_df), 1))
    )[:, 1]

    # Rule-based baseline
    rule_probabilities = _rule_based_probabilities(
        test_df,
        train_df,
    )

    def calculate_metrics(probabilities):
        return {
            "roc_auc": float(
                roc_auc_score(y_true, probabilities)
            ),
            "pr_auc": float(
                average_precision_score(y_true, probabilities)
            ),
            "brier": float(
                brier_score_loss(y_true, probabilities)
            ),
            "base_rate": float(np.mean(y_true)),
            "n": int(len(test_df)),
        }

    dummy_results = calculate_metrics(dummy_probabilities)
    rule_results = calculate_metrics(rule_probabilities)

    return {
        "model": model_results,
        "dummy": dummy_results,
        "rule_based": rule_results,
        "lift_pr_auc": float(
            model_results["pr_auc"] - dummy_results["pr_auc"]
        ),
    }