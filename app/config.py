import os
from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/tickets.db")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
API_TOKEN_SECRET = os.getenv("API_TOKEN_SECRET", "change-me")
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD_HASH = os.getenv("ADMIN_PASSWORD_HASH", "")
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))

os.environ.setdefault("HF_ENDPOINT", HF_ENDPOINT)
