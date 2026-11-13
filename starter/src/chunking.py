"""Implement fixed-size and structure-aware chunking."""
from .models import Chunk, DocumentPage


def fixed_size(page: DocumentPage, size: int = 500, overlap: int = 50) -> list[Chunk]:
    """Chiến lược A: Chia chunk kích thước cố định với sliding window overlap."""
    text = page.text
    step = size - overlap
    if step <= 0:
        raise ValueError("Kích thước size phải lớn hơn overlap")

    chunks: list[Chunk] = []
    chunk_idx = 0
    start = 0

    while start < len(text):
        end = min(start + size, len(text))
        chunk_text = text[start:end].strip()

        if chunk_text:
            chunk_id = f"{page.document_id}_p{page.page:02d}_fixed_{chunk_idx:03d}"
            c_meta = dict(page.metadata)
            c_meta.update({
                "strategy": "fixed_size",
                "chunk_index": chunk_idx,
                "char_length": len(chunk_text)
            })
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=page.document_id,
                    source=page.source,
                    page=page.page,
                    text=chunk_text,
                    metadata=c_meta
                )
            )
            chunk_idx += 1

        if end == len(text):
            break
        start += step

    return chunks


def by_paragraph(page: DocumentPage, max_size: int = 800) -> list[Chunk]:
    """Chiến lược B: Gom đoạn văn bản theo cấu trúc logic, không cắt giữa chừng câu."""
    raw_paragraphs = [p.strip() for p in page.text.split("\n\n") if p.strip()]
    if not raw_paragraphs:
        raw_paragraphs = [p.strip() for p in page.text.split("\n") if p.strip()]

    chunks: list[Chunk] = []
    current_text = ""
    chunk_idx = 0

    def add_chunk(text_content: str):
        nonlocal chunk_idx
        chunk_id = f"{page.document_id}_p{page.page:02d}_par_{chunk_idx:03d}"
        c_meta = dict(page.metadata)
        c_meta.update({
            "strategy": "by_paragraph",
            "chunk_index": chunk_idx,
            "char_length": len(text_content)
        })
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                document_id=page.document_id,
                source=page.source,
                page=page.page,
                text=text_content,
                metadata=c_meta
            )
        )
        chunk_idx += 1

    for para in raw_paragraphs:
        # Nếu đoạn đơn lẻ quá dài, cắt theo max_size
        if len(para) > max_size:
            if current_text:
                add_chunk(current_text)
                current_text = ""
            start = 0
            while start < len(para):
                end = min(start + max_size, len(para))
                add_chunk(para[start:end].strip())
                start += max_size
            continue

        # Gom các đoạn văn ngắn lại với nhau cho đủ ngữ cảnh
        if current_text and (len(current_text) + len(para) + 2 > max_size):
            add_chunk(current_text)
            current_text = para
        else:
            if current_text:
                current_text += "\n\n" + para
            else:
                current_text = para

    if current_text:
        add_chunk(current_text)

    return chunks


if __name__ == "__main__":
    from .ingestion import load_pdf
    from pathlib import Path
    
    pages = load_pdf(Path("data/AI-Agents-Chapter3.pdf"))
    all_chunks_a = []
    all_chunks_b = []
    for p in pages:
        all_chunks_a.extend(fixed_size(p, size=500, overlap=50))
        all_chunks_b.extend(by_paragraph(p, max_size=800))
        
    print(f"Tổng số chunk Chiến lược A (Fixed-size 500/50): {len(all_chunks_a)}")
    print(f"Tổng số chunk Chiến lược B (By-paragraph 800): {len(all_chunks_b)}")
