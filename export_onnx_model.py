"""导出 ONNX 模型，用于低内存部署（~150MB vs ~400MB PyTorch）"""
import os
import sys
import tempfile
import warnings

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
warnings.filterwarnings("ignore")

MODEL_NAME = "shibing624/text2vec-base-chinese"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "text2vec-onnx")

# 将临时文件目录设到项目内，避免 Windows 系统临时目录清理竞态
_TMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp")
os.makedirs(_TMP, exist_ok=True)
os.environ["TMPDIR"] = _TMP
os.environ["TEMP"] = _TMP
os.environ["TMP"] = _TMP
tempfile.tempdir = _TMP


def export():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading {MODEL_NAME} ...")
    from sentence_transformers import SentenceTransformer
    st_model = SentenceTransformer(MODEL_NAME)
    st_model = st_model.to("cpu")

    # 先用 SentenceTransformer 的底层模型导出 ONNX
    print(f"Exporting ONNX model to {OUTPUT_DIR} ...")
    from optimum.onnxruntime import ORTModelForFeatureExtraction
    from transformers import AutoTokenizer

    ort_model = ORTModelForFeatureExtraction.from_pretrained(
        MODEL_NAME,
        export=True,
        trust_remote_code=True,
    )
    ort_model.save_pretrained(OUTPUT_DIR)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # 验证输出
    onnx_file = os.path.join(OUTPUT_DIR, "model.onnx")
    if os.path.isfile(onnx_file):
        size_mb = os.path.getsize(onnx_file) / (1024 * 1024)
        # 也检查外部数据文件
        data_file = onnx_file + ".data"
        if os.path.isfile(data_file):
            size_mb += os.path.getsize(data_file) / (1024 * 1024)
        print(f"ONNX model size: {size_mb:.1f} MB")
    else:
        files = os.listdir(OUTPUT_DIR) if os.path.isdir(OUTPUT_DIR) else []
        print(f"Export failed. Files in output dir: {files}")
        sys.exit(1)


if __name__ == "__main__":
    export()
