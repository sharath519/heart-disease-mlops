# MLOps Assignment 1 - Heart Disease Prediction
Name / ID: ...   Course: AIMLCZG523   Repo: https://github.com/<you>/heart-disease-mlops

## 1. Introduction and problem statement (0.5 page)
Goal, dataset (303 patients, 13 features, binary target), why recall matters for screening.

## 2. Setup and installation (0.5 page)
Prerequisites, venv, `pip install -r requirements.txt`, command list from README.

## 3. EDA (2 pages)  - screenshots/eda/
Class balance, histograms, correlation heatmap, boxplots. 2-3 insights per plot. Cleaning decisions
(`?` in ca/thal -> mode, target binarised, duplicates).

## 4. Feature engineering and modelling (1.5 pages)
Pipeline (impute -> scale numeric, one-hot categorical). Models, grids, CV protocol, metrics table
(from reports/model_comparison.csv), ROC curve, confusion matrix, why the final model was chosen.

## 5. Experiment tracking (1 page)
MLflow screenshots: runs table, comparison, artifacts. Short results table, what you learnt.

## 6. Architecture diagram (0.5 page)
Data -> training -> MLflow -> model.joblib -> Docker -> Kubernetes -> Prometheus/Grafana (draw your own).

## 7. CI/CD (1 page)
ci.yml steps; screenshot of failed run, then passing run, with artifacts.

## 8. Containerisation and deployment (1.5 pages)
docker build/run/curl screenshots; kubectl get pods,svc; /predict via the service; manifest decisions
(replicas, probes, resource limits).

## 9. Monitoring and logging (1 page)
Log lines, Prometheus targets, Grafana dashboard screenshot, what each panel shows.

## 10. Conclusion, limitations, links (0.5 page)
Repo link, API access instructions, video link, future work (drift monitoring, model registry, HPA).
