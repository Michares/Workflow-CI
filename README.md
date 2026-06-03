\# Workflow CI - Personality Classification



Repository Kriteria 3 submission MSML Dicoding: CI pipeline untuk training otomatis + build Docker image.



\## Struktur

\- `MLProject/` — MLflow Project folder

&#x20; - `MLProject` — config MLflow Project

&#x20; - `conda.yaml` — Python env spec

&#x20; - `modelling.py` — script training

&#x20; - `personality\_preprocessing/` — preprocessed dataset

\- `.github/workflows/main.yml` — CI pipeline



\## Docker Hub

Image otomatis di-push ke: \*\*https://hub.docker.com/r/michares/personality-classifier\*\*



Cara pull \& run:

\\`\\`\\`bash

docker pull michares/personality-classifier:latest

docker run -p 5000:8080 michares/personality-classifier:latest

\\`\\`\\`



\## Trigger

CI ter-trigger setiap push ke `main` branch atau manual via GitHub Actions UI.

