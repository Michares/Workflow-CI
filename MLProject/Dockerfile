FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    mlflow==2.19.0 \
    scikit-learn==1.5.2 \
    pandas==2.3.3 \
    numpy==2.4.6 \
    joblib==1.5.3

COPY model_artifact/ /app/model/

EXPOSE 8080

CMD ["mlflow", "models", "serve", "-m", "/app/model", "-h", "0.0.0.0", "-p", "8080", "--no-conda"]