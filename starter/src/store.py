"""Vector Store implementation using persistent ChromaDB (No Docker required)."""
from pathlib import Path
from typing import Any
import chromadb
from .models import Chunk


class ChromaVectorStore:
    def __init__(self, persist_directory: str = "./chroma_db", dimensions: int = 4096) -> None:
        self.persist_directory = persist_directory
        self.dimensions = dimensions
        # Khởi tạo client lưu trữ bền vững trên đĩa
        self.client = chromadb.PersistentClient(path=self.persist_directory)

    def create_schema(self) -> None:
        """Đảm bảo thư mục lưu trữ sẵn sàng."""
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        print(f"ChromaDB đã sẵn sàng tại: {self.persist_directory}")

    def _get_collection(self, strategy: str):
        collection_name = f"strategy_{strategy.lower()}"
        return self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Lưu vector và metadata vào database; cơ chế upsert giúp chống trùng lặp khi chạy lại."""
        if len(chunks) != len(vectors):
            raise ValueError(f"Số lượng chunk ({len(chunks)}) không khớp với số vector ({len(vectors)})")

        collection = self._get_collection(strategy)

        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = []
        for c in chunks:
            meta = {
                "document_id": c.document_id,
                "source": c.source,
                "page": c.page,
                "section": c.metadata.get("section", "other"),
                "strategy": c.metadata.get("strategy", strategy)
            }
            metadatas.append(meta)

        # Batch upsert để tối ưu hiệu năng
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            collection.upsert(
                ids=ids[i : i + batch_size],
                embeddings=vectors[i : i + batch_size],
                documents=documents[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size]
            )
        print(f"-> Đã lưu {len(ids)} chunks vào collection '{collection.name}' thành công.")

    def search(
        self,
        strategy: str,
        query_vector: list[float],
        top_k: int = 3,
        filters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Truy xuất top-k vector tương đồng nhất theo cosine, hỗ trợ lọc metadata."""
        collection = self._get_collection(strategy)

        where_clause = None
        if filters:
            # Chỉ lấy các trường hợp lệ trong metadata để filter
            valid_filters = {k: v for k, v in filters.items() if v is not None}
            if len(valid_filters) == 1:
                where_clause = valid_filters
            elif len(valid_filters) > 1:
                where_clause = {"$and": [{k: v} for k, v in valid_filters.items()]}

        results = collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=where_clause
        )

        formatted_results = []
        if results and results["ids"] and len(results["ids"][0]) > 0:
            for idx in range(len(results["ids"][0])):
                # Cosine similarity = 1 - cosine distance
                distance = results["distances"][0][idx] if "distances" in results and results["distances"] else 0.0
                score = round(1.0 - distance, 4)
                item = {
                    "chunk_id": results["ids"][0][idx],
                    "text": results["documents"][0][idx],
                    "score": score,
                    "metadata": results["metadatas"][0][idx]
                }
                formatted_results.append(item)

        return formatted_results


# Giữ alias PgVectorStore để tương thích ngược nếu có file khác import
PgVectorStore = ChromaVectorStore
