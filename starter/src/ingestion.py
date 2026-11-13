"""Read and normalize source data from Chapter 3 of AI Agents book."""
import unicodedata
from pathlib import Path
from pypdf import PdfReader
from .models import DocumentPage


def normalize_text(text: str) -> str:
    """Chuẩn hóa Unicode (NFKC) và khoảng trắng nhưng giữ nguyên cấu trúc từ ngữ tiếng Việt."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def get_section_label(page_num: int) -> str:
    """Xác định nhãn section theo 40 trang của Chương 3 (sách AI Agent)."""
    if 1 <= page_num <= 11:
        return "user_memory"
    elif 12 <= page_num <= 21:
        return "rag_fundamentals"
    elif 22 <= page_num <= 38:
        return "advanced_retrieval"
    elif 39 <= page_num <= 40:
        return "summary"
    return "other"


def load_pdf(path: Path | str = "data/AI-Agents-Chapter3.pdf", document_id: str = "ai-agents-chapter3", source: str = "data/AI-Agents-Chapter3.pdf") -> list[DocumentPage]:
    """Đọc toàn bộ 40 trang kỹ thuật từ PDF Chương 3."""
    pdf_path = Path(path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    pages: list[DocumentPage] = []

    for idx, page in enumerate(reader.pages):
        page_num = idx + 1
        raw_text = page.extract_text() or ""
        clean_text = normalize_text(raw_text)

        if len(clean_text) < 50:
            print(f"[Cảnh báo] Trang {page_num} có ít hơn 50 ký tự text.")
            continue

        section = get_section_label(page_num)

        doc_page = DocumentPage(
            document_id=document_id,
            source=source,
            page=page_num,
            text=clean_text,
            metadata={
                "section": section,
                "chapter": 3,
                "original_book_page": page_num + 105,
                "char_count": len(clean_text)
            }
        )
        pages.append(doc_page)

    print(f"-> Đã load thành công {len(pages)} trang từ Chương 3.")
    return pages


if __name__ == "__main__":
    pages = load_pdf()
    print(f"Ví dụ Trang 1 - Section: {pages[0].metadata['section']}")
    print(f"Text trích đoạn:\n{pages[0].text[:200]}...")
