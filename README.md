# Workflow CI - Personality Classification

Repository Kriteria 3 submission MSML Dicoding: CI pipeline untuk training otomatis + build Docker image.

## Struktur

- `MLProject/` — MLflow Project folder
  - `MLProject` — config MLflow Project
  - `conda.yaml` — Python env spec
  - `modelling.py` — script training
  - `Dockerfile` — Docker build instructions
  - `personality_preprocessing/` — preprocessed dataset
- `.github/workflows/main.yml` — CI pipeline

## Docker Hub

Image otomatis di-push ke: **https://hub.docker.com/r/michares/personality-classifier**

Cara pull & run:

```bash
docker pull michares/personality-classifier:latest
docker run -p 5005:8080 michares/personality-classifier:latest
```

## Trigger

CI ter-trigger setiap push ke `main` branch atau manual via GitHub Actions UI (tab Actions → Run workflow).

## Workflow Steps

1. Checkout repository
2. Set up Python 3.12.7
3. Install dependencies (mlflow, scikit-learn, pandas, numpy, etc.)
4. Run MLflow Project
5. Get latest MLflow run ID
6. Copy model artifact for Docker build
7. Upload artifacts to GitHub
8. Set up Docker Buildx
9. Log in to Docker Hub
10. Build Docker image
11. Tag Docker image
12. Push Docker image to Docker Hub
