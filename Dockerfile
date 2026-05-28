FROM python:3.12-slim

WORKDIR /app

# Use HF mirror for faster model download in China
ENV HF_ENDPOINT=https://hf-mirror.com

# onnxruntime needs libgomp1; curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY . .

RUN mkdir -p data knowledge_base models/text2vec-onnx

# Pre-built ONNX model is copied from build context (exported locally to avoid OOM on small servers)
# If models/text2vec-onnx/model.onnx doesn't exist, app falls back to PyTorch at runtime

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
