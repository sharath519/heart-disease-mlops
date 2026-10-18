import numpy as np

from src import config
from src.features import build_pipeline, build_preprocessor
from sklearn.linear_model import LogisticRegression


def test_preprocessor_output_is_numeric_and_finite(clean_df):
    X = clean_df[config.FEATURES]
    out = build_preprocessor().fit_transform(X)
    assert out.shape[0] == len(X)
    assert np.isfinite(np.asarray(out)).all()


def test_preprocessor_handles_missing_values(clean_df):
    X = clean_df[config.FEATURES].astype(float)
    X.iloc[0, X.columns.get_loc("chol")] = np.nan
    X.iloc[1, X.columns.get_loc("ca")] = np.nan
    out = build_preprocessor().fit_transform(X)
    assert np.isfinite(np.asarray(out)).all()


def test_numeric_features_are_scaled(clean_df):
    pre = build_preprocessor().fit(clean_df[config.FEATURES])
    out = pre.transform(clean_df[config.FEATURES])
    n = len(config.NUMERIC_FEATURES)
    assert np.allclose(np.asarray(out)[:, :n].mean(axis=0), 0, atol=1e-6)


def test_unseen_category_does_not_crash(clean_df):
    pipe = build_pipeline(LogisticRegression(max_iter=500))
    pipe.fit(clean_df[config.FEATURES], clean_df[config.TARGET])
    row = clean_df[config.FEATURES].iloc[[0]].copy()
    row["thal"] = 99
    assert pipe.predict_proba(row).shape == (1, 2)
