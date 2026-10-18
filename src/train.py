"""Train, tune, evaluate and track models; save the best full pipeline.

Usage:  python -m src.train
View runs:  mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
import json
import logging

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay  # noqa: E402
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,  # noqa: E402
                                     cross_val_predict, cross_validate)

from src import config  # noqa: E402
from src.data import load_clean, split_xy  # noqa: E402
from src.features import build_pipeline  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train")

SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]
RS = config.RANDOM_STATE

# name -> (estimator, hyper-parameter grid for the tuned run)
MODEL_SPECS = {
    "logistic_regression": (
        LogisticRegression(max_iter=2000, random_state=RS),
        {"model__C": [0.01, 0.1, 1, 10], "model__penalty": ["l2"]},
    ),
    "random_forest": (
        RandomForestClassifier(random_state=RS),
        {"model__n_estimators": [100, 300],
         "model__max_depth": [3, 5, None],
         "model__min_samples_leaf": [1, 3]},
    ),
    "gradient_boosting": (
        GradientBoostingClassifier(random_state=RS),
        {"model__n_estimators": [50, 100, 200],
         "model__learning_rate": [0.03, 0.1],
         "model__max_depth": [2, 3]},
    ),
}


def cv_splitter(seed: int) -> StratifiedKFold:
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)


def evaluate(pipe, X, y, seed=RS):
    """5-fold stratified CV; returns {metric: (mean, std)}."""
    res = cross_validate(pipe, X, y, cv=cv_splitter(seed), scoring=SCORING, n_jobs=-1)
    return {m: (float(res[f"test_{m}"].mean()), float(res[f"test_{m}"].std()))
            for m in SCORING}


def feature_names(fitted_pipe):
    return list(fitted_pipe.named_steps["preprocess"].get_feature_names_out())


def make_plots(pipe, X, y, name, out_dir):
    """Save ROC curve, confusion matrix and feature importance; return their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    proba = cross_val_predict(clone(pipe), X, y, cv=cv_splitter(RS),
                              method="predict_proba")[:, 1]

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y, proba, ax=ax, name=name)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_title(f"ROC curve (5-fold CV) - {name}")
    p = out_dir / f"{name}_roc.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(y, (proba >= 0.5).astype(int), ax=ax,
                                            cmap="Blues", display_labels=["No disease", "Disease"])
    ax.set_title(f"Confusion matrix (CV) - {name}")
    p = out_dir / f"{name}_confusion.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    fitted = clone(pipe).fit(X, y)
    est = fitted.named_steps["model"]
    if hasattr(est, "feature_importances_"):
        imp = est.feature_importances_
    elif hasattr(est, "coef_"):
        imp = np.abs(est.coef_[0])
    else:
        imp = None
    if imp is not None:
        s = pd.Series(imp, index=feature_names(fitted)).sort_values().tail(15)
        fig, ax = plt.subplots(figsize=(6, 5))
        s.plot.barh(ax=ax, color="#2a6f97")
        ax.set_title(f"Top feature importances - {name}")
        ax.set_xlabel("importance")
        p = out_dir / f"{name}_importance.png"
        fig.savefig(p, dpi=130, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)
    return paths


def run_one(run_name, model_key, X, y, tune):
    """One MLflow run: optionally tune, evaluate, log everything."""
    estimator, grid = MODEL_SPECS[model_key]
    pipe = build_pipeline(clone(estimator))

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({"model_type": model_key, "tuned": str(tune)})
        mlflow.log_param("model_type", model_key)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("random_state", RS)
        mlflow.log_param("n_samples", len(X))

        if tune:
            search = GridSearchCV(pipe, grid, cv=cv_splitter(RS), scoring="roc_auc",
                                  n_jobs=-1, refit=True)
            search.fit(X, y)
            pipe = search.best_estimator_
            mlflow.log_params({k.replace("model__", ""): v
                               for k, v in search.best_params_.items()})
            mlflow.log_metric("tuning_best_cv_roc_auc", float(search.best_score_))
            pd.DataFrame(search.cv_results_).to_csv(config.PLOTS_DIR / f"{run_name}_grid.csv",
                                                    index=False)
            mlflow.log_artifact(str(config.PLOTS_DIR / f"{run_name}_grid.csv"))
        else:
            mlflow.log_params({k: v for k, v in estimator.get_params().items()
                               if isinstance(v, (int, float, str, bool, type(None)))})

        # Evaluate with a different CV seed than the one used for tuning
        # to reduce optimistic bias from hyper-parameter selection.
        scores = evaluate(pipe, X, y, seed=RS + 1)
        for m, (mean, std) in scores.items():
            mlflow.log_metric(f"cv_{m}_mean", mean)
            mlflow.log_metric(f"cv_{m}_std", std)

        for p in make_plots(pipe, X, y, run_name, config.PLOTS_DIR / "plots"):
            mlflow.log_artifact(str(p), artifact_path="plots")

        final = clone(pipe).fit(X, y)
        mlflow.sklearn.log_model(final, name="model", input_example=X.head(3),
                                 serialization_format="cloudpickle")
        log.info("%-28s ROC-AUC=%.3f  recall=%.3f  acc=%.3f", run_name,
                 scores["roc_auc"][0], scores["recall"][0], scores["accuracy"][0])
        return {"run": run_name, "model": model_key, "tuned": tune,
                "pipeline": final,
                **{f"{m}_mean": v[0] for m, v in scores.items()},
                **{f"{m}_std": v[1] for m, v in scores.items()}}


def main():
    config.PLOTS_DIR.mkdir(exist_ok=True)
    config.MODEL_DIR.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)

    df = load_clean()
    X, y = split_xy(df)
    log.info("Training data: %s, positive rate %.2f", X.shape, y.mean())

    results = []
    for key in MODEL_SPECS:
        results.append(run_one(f"{key}_baseline", key, X, y, tune=False))
        results.append(run_one(f"{key}_tuned", key, X, y, tune=True))

    table = pd.DataFrame([{k: v for k, v in r.items() if k != "pipeline"} for r in results])
    table = table.sort_values("roc_auc_mean", ascending=False)
    table.to_csv(config.PLOTS_DIR / "model_comparison.csv", index=False)
    log.info("\n%s", table[["run", "accuracy_mean", "precision_mean", "recall_mean",
                            "f1_mean", "roc_auc_mean"]].round(3).to_string(index=False))

    best = max(results, key=lambda r: (round(r["roc_auc_mean"], 6), r["tuned"]))
    joblib.dump(best["pipeline"], config.MODEL_PATH)
    summary = {k: v for k, v in best.items() if k != "pipeline"}
    summary["features"] = config.FEATURES
    config.METRICS_PATH.write_text(json.dumps(summary, indent=2, default=str))
    log.info("Best model: %s -> saved to %s", best["run"], config.MODEL_PATH)


if __name__ == "__main__":
    main()
