"""Shared fixtures: small synthetic dataset with the Cleveland schema (no network needed)."""
import numpy as np
import pandas as pd
import pytest

from src import config


def make_synthetic(n=200, seed=0, with_missing=False):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(29, 78, n),
        "sex": rng.integers(0, 2, n),
        "cp": rng.integers(1, 5, n),
        "trestbps": rng.integers(94, 200, n),
        "chol": rng.integers(126, 400, n),
        "fbs": rng.integers(0, 2, n),
        "restecg": rng.integers(0, 3, n),
        "thalach": rng.integers(71, 202, n),
        "exang": rng.integers(0, 2, n),
        "oldpeak": np.round(rng.uniform(0, 6, n), 1),
        "slope": rng.integers(1, 4, n),
        "ca": rng.integers(0, 4, n),
        "thal": rng.choice([3, 6, 7], n),
    })
    risk = (0.03 * df.age + 0.5 * df.exang + 0.4 * df.oldpeak + 0.5 * df.ca
            - 0.02 * df.thalach + rng.normal(0, 1, n))
    df["num"] = (risk > np.median(risk)).astype(int) * rng.integers(1, 5, n)
    if with_missing:
        df = df.astype({"ca": float, "thal": float})
        df.loc[[3, 10], "ca"] = np.nan
        df.loc[[5], "thal"] = np.nan
    return df


@pytest.fixture
def raw_df():
    return make_synthetic(with_missing=True)


@pytest.fixture
def clean_df():
    from src.data import clean_data
    return clean_data(make_synthetic())


@pytest.fixture
def sample_patient():
    return {"age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
            "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
            "ca": 0, "thal": 6}


@pytest.fixture
def model_path(tmp_path, clean_df):
    import joblib
    from sklearn.linear_model import LogisticRegression

    from src.features import build_pipeline
    X, y = clean_df[config.FEATURES], clean_df[config.TARGET]
    pipe = build_pipeline(LogisticRegression(max_iter=500, random_state=0)).fit(X, y)
    p = tmp_path / "model.joblib"
    joblib.dump(pipe, p)
    return p
