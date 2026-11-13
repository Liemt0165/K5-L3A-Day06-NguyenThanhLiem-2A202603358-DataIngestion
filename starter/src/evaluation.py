"""Evaluate retrieved evidence and preserve per-query traces."""
from typing import Any
import re


def _normalize_evidence(text: str) -> str:
    """Loại bỏ ký tự đặc biệt và khoảng trắng thừa để so khớp evidence."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text.lower())).strip()


def is_hit(question: dict[str, Any], hit: dict[str, Any]) -> bool:
    """Một hit hợp lệ khi chunk đúng trang gold và chứa bằng chứng đã ghi nhận."""
    hit_page = hit.get("metadata", {}).get("page")
    gold_page = question.get("gold_page")

    # Kiểm tra đúng trang vật lý của tài liệu
    if hit_page != gold_page:
        return False

    chunk_text = _normalize_evidence(hit.get("text", ""))
    evidence = _normalize_evidence(question.get("evidence_quote", ""))

    if not evidence:
        return True

    # 1. Trực tiếp chứa nguyên cụm evidence
    if evidence in chunk_text:
        return True

    # 2. Hoặc chứa phần lớn các từ khoá chính của evidence
    ev_words = [w for w in evidence.split() if len(w) > 3]
    if not ev_words:
        return True
    matched_words = sum(1 for w in ev_words if w in chunk_text)
    return (matched_words / len(ev_words)) >= 0.75


def evaluate_query(question: dict[str, Any], hits: list[dict[str, Any]]) -> dict[str, Any]:
    """Đánh giá chi tiết top-3 hits cho một câu hỏi."""
    top3_details = []
    hit_at_1 = False
    hit_at_3 = False

    for rank, hit in enumerate(hits, start=1):
        hit_valid = is_hit(question, hit)
        if rank == 1 and hit_valid:
            hit_at_1 = True
        if hit_valid:
            hit_at_3 = True

        top3_details.append({
            "rank": rank,
            "chunk_id": hit.get("chunk_id"),
            "score": hit.get("score"),
            "page": hit.get("metadata", {}).get("page"),
            "section": hit.get("metadata", {}).get("section"),
            "is_hit": hit_valid,
            "text_snippet": hit.get("text", "")[:150] + "..."
        })

    return {
        "question_id": question.get("id"),
        "split": question.get("split"),
        "question": question.get("question"),
        "gold_page": question.get("gold_page"),
        "gold_evidence": question.get("evidence_quote"),
        "filter": question.get("filter"),
        "hit_at_1": hit_at_1,
        "hit_at_3": hit_at_3,
        "hits": top3_details
    }


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Tổng hợp chỉ số Hit@1 và Evidence Hit@3 theo split (dev / test)."""
    splits = ["dev", "test", "all"]
    summary = {}

    for s in splits:
        filtered = traces if s == "all" else [t for t in traces if t.get("split") == s]
        total = len(filtered)
        if total == 0:
            summary[s] = {"total": 0, "hit1": "0/0 (0.0%)", "hit3": "0/0 (0.0%)"}
            continue

        h1 = sum(1 for t in filtered if t.get("hit_at_1"))
        h3 = sum(1 for t in filtered if t.get("hit_at_3"))

        summary[s] = {
            "total": total,
            "hit1_count": h1,
            "hit1_ratio": f"{h1}/{total}",
            "hit1_pct": round(h1 / total * 100, 2),
            "hit3_count": h3,
            "hit3_ratio": f"{h3}/{total}",
            "hit3_pct": round(h3 / total * 100, 2)
        }

    return summary
