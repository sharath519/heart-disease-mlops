"""Inference helpers: load the saved pipeline and predict."""
import joblib
import pandas as pd

from src import config


def load_model(path=None):
    return joblib.load(path or config.MODEL_PATH)


def predict_records(model, records):
    """records: list of dicts with the 13 feature names -> list of result dicts."""
    X = pd.DataFrame(records)[config.FEATURES]
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return [{"prediction": int(p), "confidence": float(round(pr if p == 1 else 1 - pr, 4)),
             "probability_disease": float(round(pr, 4))} for p, pr in zip(pred, proba)]


if __name__ == "__main__":
    sample = {"age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
              "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
              "ca": 0, "thal": 6}
    print(predict_records(load_model(), [sample]))
