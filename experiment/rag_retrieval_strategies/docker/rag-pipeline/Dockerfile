FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY experiment/rag_retrieval_strategies/docker/rag-pipeline/requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock

COPY database /app/database
COPY experiment/rag_retrieval_strategies/src /app/experiment/rag_retrieval_strategies/src
COPY experiment/rag_retrieval_strategies/analysis /app/experiment/rag_retrieval_strategies/analysis
COPY experiment/rag_retrieval_strategies/data/benchmark /app/experiment/rag_retrieval_strategies/data/benchmark

ENV PYTHONPATH=/app/experiment/rag_retrieval_strategies/src:/app/database \
    PYTHONUNBUFFERED=1

CMD ["python"]
