import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from tawseel.data import make_parcels
from tawseel.model import (FEATURE_COLUMNS, add_history_features, build_pipeline,
                           compare_to_baseline, time_split)


def tiny():
    return pd.DataFrame({
        "customer_id": ["C1", "C1", "C1", "C2"],
        "created_at": pd.to_datetime(["2026-06-01 09:00", "2026-06-02 09:00",
                                      "2026-06-03 09:00", "2026-06-01 10:00"]),
        "failed": [1, 0, 1, 0],
    })


def test_history_features_only_look_at_earlier_parcels():
    out = add_history_features(tiny())
    assert out["customer_prior_parcels"].tolist() == [0, 1, 2, 0]
    assert np.isnan(out["customer_prior_fail_rate"].iloc[0])
    assert out["customer_prior_fail_rate"].iloc[1] == pytest.approx(1.0)
    assert out["customer_prior_fail_rate"].iloc[2] == pytest.approx(0.5)


def test_time_split_puts_the_latest_parcels_in_test():
    df = make_parcels(500, seed=7)
    train, test = time_split(df, "created_at", test_fraction=0.2)
    assert len(train) + len(test) == len(df)
    assert train["created_at"].max() < test["created_at"].min()


def test_leaky_columns_are_not_features():
    for col in ["courier_notes", "delivered_at", "failed", "parcel_id"]:
        assert col not in FEATURE_COLUMNS


def test_build_pipeline_returns_an_unfitted_pipeline():
    assert isinstance(build_pipeline(seed=0), Pipeline)


def test_model_beats_the_dummy_baseline():
    df = add_history_features(make_parcels(3000, seed=11))
    train, test = time_split(df)
    result = compare_to_baseline(train, test)
    assert result["model"]["pr_auc"] > result["dummy"]["pr_auc"] + 0.05
