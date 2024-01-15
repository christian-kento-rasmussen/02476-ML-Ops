# Base image
#FROM nvcr.io/nvidia/pytorch:22.07-py3
FROM python:3.10-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY FishEye/ FishEye/


WORKDIR /
RUN --mount=type=cache,target=/root/.cache pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir
RUN mkdir models
RUN mkdir data

ENTRYPOINT ["python", "-u", "FishEye/train_model.py"]