FROM python:3.12-slim

WORKDIR /app

# Use HF mirror for faster model download in China
ENV HF_ENDPOINT=https://hf-mirror.com

# onnxruntime needs libgomp1
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data knowledge_base models/text2vec-onnx

# Export ONNX model at build time (avoids runtime download, ~150MB vs 400MB PyTorch)
RUN python export_onnx_model.py

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
