import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src import config
from src.features import build_pipeline
from src.predict import load_model, predict_records
from src.train import evaluate


def test_model_trains_and_outputs_valid_probabilities(clean_df):
    X, y = clean_df[config.FEATURES], clean_df[config.TARGET]
    pipe = build_pipeline(RandomForestClassifier(n_estimators=20, random_state=0)).fit(X, y)
    proba = pipe.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert ((proba >= 0) & (proba <= 1)).all()
    assert np.allclose(proba.sum(axis=1), 1)


def test_training_is_reproducible(clean_df):
    X, y = clean_df[config.FEATURES], clean_df[config.TARGET]
    p1 = build_pipeline(RandomForestClassifier(n_estimators=20, random_state=1)).fit(X, y)
    p2 = build_pipeline(RandomForestClassifier(n_estimators=20, random_state=1)).fit(X, y)
    assert np.allclose(p1.predict_proba(X), p2.predict_proba(X))


def test_cv_evaluation_returns_all_metrics(clean_df):
    X, y = clean_df[config.FEATURES], clean_df[config.TARGET]
    pipe = build_pipeline(RandomForestClassifier(n_estimators=10, random_state=0))
    scores = evaluate(pipe, X, y)
    assert set(scores) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert all(0 <= mean <= 1 for mean, _ in scores.values())


def test_saved_pipeline_predicts_from_raw_features(model_path, sample_patient):
    result = predict_records(load_model(model_path), [sample_patient])[0]
    assert result["prediction"] in (0, 1)
    assert 0.5 <= result["confidence"] <= 1
