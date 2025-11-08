---

# Perceptra

A production-ready AI/ML MLOps boilerplate integrating **Azure ML**, **AKS**, **FastAPI**, **Docker**, **Azure DevOps**, **Key Vault**, **App Insights**, **MLflow**, and **Locust** for end-to-end experimentation, deployment, and monitoring.

![Architecture Overview](./images/intro.png)

---

## Overview

**Perceptra** is an enterprise-grade template for building, deploying, and scaling machine learning applications on Azure’s cloud ecosystem. It standardizes the workflow from data ingestion to inference, including CI/CD, secrets management, observability, and model lifecycle tracking.

### Tech Stack Summary

| Layer        | Tool                               | Purpose                                                   |
| :----------- | :--------------------------------- | :-------------------------------------------------------- |
| Cloud        | **Azure ML**                       | Machine learning compute, pipelines, and model registry   |
| Compute      | **AKS (Azure Kubernetes Service)** | Container orchestration and scalable deployment           |
| Deployment   | **FastAPI + Docker**               | API serving and containerization                          |
| Automation   | **Azure DevOps**                   | CI/CD pipelines for training & deployment                 |
| Secrets      | **Azure Key Vault**                | Secret, credential, and key management                    |
| Monitoring   | **Azure App Insights**             | Application telemetry & performance monitoring            |
| Tracking     | **MLflow**                         | Experiment tracking and model versioning                  |
| Load Testing | **Locust**                         | Load testing for API latency, throughput, and scalability |

---

## Repository Structure

```
azure_components/   # Azure resource helpers (Key Vault, Identity, ML SDK clients)
deployment/         # Deployment scripts and manifests (Docker/K8s)
devops/             # Azure DevOps CI/CD pipeline YAML definitions
docker/             # Dockerfiles and container build context
environment/        # Environment configuration files (.env per environment)
integration/        # Service integration scripts and SDK adapters
jobs/               # Batch jobs, data prep, and cron schedulers
k8s/                # Kubernetes manifests (Deployments, Services, Ingress)
keyvault/           # Key Vault sync utilities for secrets
load_testing/       # Locust scripts, reports, and configuration files
monitoring/         # App Insights setup and dashboard JSONs
pipelines/          # MLflow & AzureML pipelines for training and scoring
requirements/       # Exported dependencies if not using Poetry
src/                # Application source code (FastAPI app, utils, models)
tests/              # Unit and integration tests
main.py             # Entry point for API or CLI
pyproject.toml      # Poetry project metadata and dependencies
```

---

## Setup Instructions

### Prerequisites

* Python 3.10+
* Poetry
* Docker Desktop (with Kubernetes enabled)
* Azure CLI & Azure ML CLI extension
* MLflow (for local experiment tracking)

### Local Environment Setup

```bash
git clone https://github.com/smaliaquib/Perceptra
cd Perceptra
poetry install
cp environment/sample.env .env
poetry run pytest -q
poetry run python main.py
```

---

## Secrets Management with Key Vault

```python
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential

vault_url = f"https://{os.getenv('AZURE_KEY_VAULT_NAME')}.vault.azure.net"
client = SecretClient(vault_url=vault_url, credential=DefaultAzureCredential())
secret_value = client.get_secret("DBConnectionString").value
```

Add to `.env`:

```
AZURE_KEY_VAULT_NAME=perceptra-vault
AZURE_CLIENT_ID=<client-id>
AZURE_TENANT_ID=<tenant-id>
AZURE_CLIENT_SECRET=<client-secret>
```

---

## MLflow Tracking & Model Management

Start local MLflow tracking server:

```bash
mlflow server \
  --host 127.0.0.1 --port 5000 \
  --backend-store-uri sqlite:///mlruns.db \
  --default-artifact-root ./mlruns
```

Log experiments:

```python
import mlflow

mlflow.set_experiment("perceptra-exp")
with mlflow.start_run(run_name="baseline"):
    mlflow.log_param("model", "xgboost")
    mlflow.log_metric("rmse", 0.231)
    mlflow.log_artifact("models/best_model.pkl")
```

Register model to Azure ML:

```bash
az ml model register -n perceptra-model -p ./mlruns/0/<run_id>/artifacts/model
```

---

## Deployment on AKS

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml
```

---

## Monitoring with App Insights

* Request logs
* Performance metrics
* Exception traces

Dashboards available under `monitoring/`.

---

## Load Testing with Locust

```bash
locust -f load_testing/locustfile.py --host http://localhost:8000
```

![Locust Dashboard](./images/locust.png)

---

## CI/CD Pipeline

Azure DevOps handles:

1. Build
2. Train
3. Evaluate
4. Register
5. Deploy

---

## Docker

```bash
docker build -t perceptra:local -f docker/Dockerfile .
docker run --rm -p 8000:8000 --env-file .env perceptra:local
```

---

## Contribution Guidelines

1. Fork & clone
2. Create feature branch
3. Commit changes
4. Add tests & docs
5. Submit PR

---

## Roadmap

* Distributed training (Ray / Azure Batch)
* Drift monitoring (Evidently AI)
* Prometheus + Grafana integration
* Canary deployment support
* Multi-environment promotion

---

## License

MIT License © 2025 Aquib Ali Khan

---
