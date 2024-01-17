# Base image
FROM python:3.10-slim

EXPOSE 8080

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml

RUN pip install -r requirements.txt --no-cache-dir

COPY FishEye/ FishEye/
COPY webapp/ webapp/
COPY models/ models/
COPY data/processed/label_map.json data/processed/label_map.json
COPY /config/config.yaml /config/config.yaml
RUN pip install . --no-deps --no-cache-dir

CMD exec uvicorn webapp.api:app --port 8080 --host 0.0.0.0 --workers 1