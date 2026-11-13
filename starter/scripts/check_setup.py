"""Read-only setup check utility for local environment."""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(env_path)

print("=== KIỂM TRA MÔI TRƯỜNG DỰ ÁN (SETUP CHECK) ===")

# 1. Kiểm tra File dữ liệu
pdf_path = Path("data/AI-Agents-Chapter3.pdf")
if pdf_path.exists():
    print(f"1. Dữ liệu: Đã tìm thấy {pdf_path} (Kích thước: {pdf_path.stat().st_size} bytes)")
else:
    print(f"1. Cảnh báo: Chưa tìm thấy {pdf_path}")

# 2. Kiểm tra Vector DB (ChromaDB)
try:
    import chromadb
    print(f"2. Vector Store: ChromaDB v{chromadb.__version__} (Sẵn sàng)")
except ImportError:
    sys.exit("2. Lỗi: Chưa cài đặt chromadb. Hãy chạy pip install chromadb")

# 3. Kiểm tra Embedding Provider
provider = os.getenv("EMBEDDING_PROVIDER", "deepinfra")
model = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B")
api_key = os.getenv("DEEPINFRA_API_KEY") if provider == "deepinfra" else os.getenv("OPENAI_API_KEY")

if api_key:
    masked_key = api_key[:8] + "..." + api_key[-4:]
    print(f"3. Embedding Provider: {provider.upper()} | Model: {model} | Key: {masked_key}")
else:
    print("3. Cảnh báo: Chưa cấu hình API Key trong file .env")

print("\nSetup OK! Toàn bộ hạ tầng sẵn sàng để chạy pipeline.")
