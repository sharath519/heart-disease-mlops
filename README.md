# Heart Disease Risk Prediction - End-to-End MLOps

Predicts heart-disease risk from patient health data (UCI Heart Disease, Cleveland) and serves it as a
containerised, monitored API on Kubernetes.

> **Before you submit:** this is a working scaffold. Personalise it (extra features, models, dashboard
> panels, wording) and make sure you can explain every file. The assignment's integrity note penalises
> near-identical submissions. Fill in your own name/repo link in the report.

## Architecture

```
UCI data -> download_data.py -> data.py (clean) -> train.py (sklearn Pipeline + GridSearchCV)
                                                      |-> MLflow (params, metrics, plots, models)
                                                      '-> models/model.joblib
model.joblib -> Docker image (FastAPI) -> Kubernetes (2 replicas, LoadBalancer)
FastAPI /metrics -> Prometheus -> Grafana        GitHub Actions: lint -> test -> train -> docker build
```

## 0. Prerequisites

Python 3.11+ (3.12 recommended), Git, Docker Desktop (with Kubernetes enabled) or Minikube + kubectl.

## 1. Setup (clean environment)

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Data + EDA  (Task 1)

```bash
python -m src.download_data       # saves data/raw/heart_raw.csv
python -m src.data                # cleans -> data/heart_clean.csv
jupyter notebook notebooks/01_eda.ipynb   # run all; plots saved to screenshots/eda/
```
If the download fails (network), download `processed.cleveland.data` manually from
https://archive.ics.uci.edu/dataset/45/heart+disease and save as `data/raw/heart_raw.csv`
with the header row `age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,num`.

In the notebook, replace each "Insight (edit)" cell with your own 2-3 sentences.

## 3. Train + track experiments  (Tasks 2, 3, 4)

```bash
python -m src.train                                   # 6 MLflow runs; saves models/model.joblib
mlflow ui --backend-store-uri sqlite:///mlflow.db     # open http://127.0.0.1:5000
```
Screenshots to take: runs table, compare view (select runs -> Compare), one run's artifacts (plots, model).
Outputs: `models/model.joblib` (full preprocessing + model), `models/metrics.json`,
`reports/model_comparison.csv`, `reports/plots/*.png`.

Quick inference check: `python -m src.predict`

## 4. Tests + lint  (Task 5)

```bash
flake8 .
pytest
```

## 5. CI/CD  (Task 5)

1. Create a GitHub repo, then:
   ```bash
   git init && git add . && git commit -m "Initial MLOps pipeline"
   git branch -M main
   git remote add origin https://github.com/<you>/heart-disease-mlops.git
   git push -u origin main
   ```
2. Open the **Actions** tab, screenshot the green run and its uploaded artifacts.
3. Prove it fails on errors: push a commit that breaks a test (e.g. change an assert in
   `tests/test_data.py`), screenshot the red run, then fix it and screenshot green again.

`models/` is tracked so the Dockerfile works; if you prefer, add `models/` to `.gitignore` and let CI train it.

## 6. Docker  (Task 6)

Train first (the image copies `models/model.joblib`).

```bash
docker build -t heart-api:1.0 .
docker run -d --name heart-api -p 8000:8000 heart-api:1.0
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,"restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":6}'
```
Expected: `{"prediction":..., "confidence":..., "probability_disease":..., "label":...}`.
Interactive docs: http://localhost:8000/docs. Screenshot build, run and the curl response (Windows
PowerShell users: use `curl.exe` or the /docs page).

Feature encodings: sex 0/1; cp 1-4; fbs 0/1; restecg 0-2; exang 0/1; slope 1-3; ca 0-3; thal 3/6/7.

## 7. Kubernetes  (Task 7)

```bash
# Docker Desktop: enable Kubernetes in settings; the local image is visible to the cluster.
# Minikube:       minikube start && minikube image load heart-api:1.0
kubectl apply -f k8s/
kubectl get pods,svc
kubectl rollout status deployment/heart-api
```
Access the service:
- Docker Desktop: `http://localhost/health`, `http://localhost/predict` (Service port 80)
- Minikube: run `minikube tunnel` (separate terminal) and use the EXTERNAL-IP, or `minikube service heart-api-service --url`
- Fallback: `kubectl port-forward svc/heart-api-service 8080:80` -> `http://localhost:8080`

Screenshot `kubectl get pods,svc`, `kubectl describe deployment heart-api`, and a `/predict` response through the service.
Optional ingress: `minikube addons enable ingress`, then `kubectl apply -f k8s/ingress.yaml`.

## 8. Monitoring + logging  (Task 8)

Request logs: `docker logs heart-api` or `kubectl logs -l app=heart-api`
(each request logs method, path, status, latency; each prediction logs the input, class and confidence).

Prometheus + Grafana (stop the standalone container first so port 8000 is free):
```bash
docker rm -f heart-api
docker compose up --build
```
- Prometheus: http://localhost:9090 (Status -> Targets should show `heart-api` UP)
- Grafana: http://localhost:3000 (admin / admin) -> dashboard **Heart Disease API** (pre-provisioned)
- Generate traffic so panels fill in:
  ```bash
  for i in $(seq 1 50); do curl -s -X POST localhost:8000/predict -H "Content-Type: application/json" \
    -d '{"age":63,"sex":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,"restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":6}' >/dev/null; done
  curl -s -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{"age":1}'   # a 422 for the error panel
  ```

## 9. Report + video  (Task 9, deliverables)

Use `docs/report_template.md` (about 10 pages: export to PDF/DOCX). Record a 5-8 minute video:
repo -> CI run -> MLflow UI -> docker run + curl -> kubectl get pods -> /predict -> Grafana.

## Final clean-setup check

Clone the repo into a new folder, create a fresh venv, `pip install -r requirements.txt`,
`python -m src.train`, `pytest`, `docker build`. Fix anything that breaks. The assignment requires this.

## Project layout

```
src/ (config, download_data, data, features, train, predict)   api/ (app, schemas)
tests/        notebooks/ (EDA, training, inference)            k8s/ (deployment, service, ingress)
monitoring/   docker-compose.yml   Dockerfile   .github/workflows/ci.yml
requirements.txt (full)   requirements-api.txt (runtime only, used in Docker)
```

## Troubleshooting

- `MlflowException ... untrusted types`: you're on a different MLflow version; keep versions pinned.
- `ImagePullBackOff` in Kubernetes: image not visible to the cluster (`minikube image load heart-api:1.0`) or tag mismatch with `k8s/deployment.yaml`.
- Pickle/version errors in Docker: `requirements.txt` and `requirements-api.txt` must pin the same scikit-learn version.
- Pinned versions need Python 3.11+. On older Python, relax the pins and re-pin what you tested.
