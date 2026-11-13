"""Real embedding provider adapter supporting DeepInfra (Qwen, BGE) and OpenAI."""
import os
from typing import Protocol
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, question: str) -> list[float]:
        ...


class MyEmbedder:
    """Adapter gọi API Embedding chuẩn OpenAI/DeepInfra."""

    def __init__(self, model_name: str | None = None) -> None:
        self.provider = os.getenv("EMBEDDING_PROVIDER", "deepinfra").lower()

        if self.provider == "deepinfra":
            self.model_name = model_name or os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B")
            self.api_key = os.getenv("DEEPINFRA_API_KEY")
            self.base_url = os.getenv("DEEPINFRA_BASE_URL", "https://api.deepinfra.com/v1/openai")
            
            # Kích thước vector chuẩn của từng model trên DeepInfra
            if "bge-m3" in self.model_name.lower():
                self.dimensions = 1024
            elif "qwen" in self.model_name.lower():
                self.dimensions = 4096
            else:
                self.dimensions = 4096
        else:
            self.model_name = model_name or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
            self.api_key = os.getenv("OPENAI_API_KEY")
            self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            self.dimensions = 1536

        if not self.api_key:
            raise ValueError(f"Không tìm thấy API Key cho provider '{self.provider}' trong .env")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def embed_documents(self, texts: list[str], batch_size: int = 30) -> list[list[float]]:
        """Lấy embedding theo từng batch (với Qwen-8B, batch 30 để đảm bảo ổn định payload)."""
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            clean_batch = [t.replace("\n", " ").strip() for t in batch]
            response = self.client.embeddings.create(
                model=self.model_name,
                input=clean_batch
            )
            batch_vectors = [item.embedding for item in response.data]
            all_embeddings.extend(batch_vectors)
        return all_embeddings

    def embed_query(self, question: str) -> list[float]:
        """Tạo embedding cho câu hỏi truy vấn."""
        clean_q = question.replace("\n", " ").strip()
        response = self.client.embeddings.create(
            model=self.model_name,
            input=[clean_q]
        )
        return response.data[0].embedding


if __name__ == "__main__":
    embedder = MyEmbedder()
    vec = embedder.embed_query("Hệ thống bộ nhớ người dùng trong AI Agent là gì?")
    print(f"Provider: {embedder.provider}")
    print(f"Model: {embedder.model_name}")
    print(f"Dimensions: {len(vec)}")
