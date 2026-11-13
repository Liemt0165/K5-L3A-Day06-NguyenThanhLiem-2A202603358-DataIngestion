"""Orchestrate the full Day 06 RAG Ingestion & Evaluation Pipeline."""
import json
import time
from pathlib import Path
from typing import Any

from .ingestion import load_pdf
from .chunking import fixed_size, by_paragraph
from .embeddings import MyEmbedder
from .store import ChromaVectorStore
from .evaluation import evaluate_query, summarize


def main() -> None:
    print("=" * 60)
    print("BẮT ĐẦU PIPELINE DAY 06: DATA INGESTION & RETRIEVAL EVALUATION")
    print("=" * 60)

    # 1. Đọc và chuẩn hóa dữ liệu
    data_path = Path("data/AI-Agents-Chapter3.pdf")
    print(f"\n[1/5] Đang nạp dữ liệu từ {data_path}...")
    pages = load_pdf(data_path)
    print(f"-> Đã nạp thành công {len(pages)} trang nội dung hữu ích từ Chương 3.")

    # 2. Xây dựng các cấu hình Chunking
    print("\n[2/5] Đang chia chunk theo các chiến lược...")
    chunks_a = []
    chunks_b = []
    for p in pages:
        chunks_a.extend(fixed_size(p, size=500, overlap=50))
        chunks_b.extend(by_paragraph(p, max_size=800))

    # Cấu hình C dùng lại chính xác các chunk của B (để cô lập biến số lọc metadata)
    chunks_c = list(chunks_b)

    print(f"-> Cấu hình A (Fixed-size 500/50): {len(chunks_a)} chunks")
    print(f"-> Cấu hình B (By-paragraph 800):  {len(chunks_b)} chunks")
    print(f"-> Cấu hình C (Giống B + Filter): {len(chunks_c)} chunks")

    # 3. Khởi tạo Embedder và Vector Store
    print("\n[3/5] Khởi tạo Embedding Model & Vector Store...")
    embedder = MyEmbedder()
    print(f"-> Model: {embedder.model_name} ({embedder.dimensions} dimensions) qua provider: {embedder.provider.upper()}")
    
    store = ChromaVectorStore(persist_directory="./chroma_db", dimensions=embedder.dimensions)
    store.create_schema()

    # Nhúng vector và lưu trữ cho từng chiến lược
    strategies = [
        ("A", chunks_a),
        ("B", chunks_b),
        ("C", chunks_c)
    ]

    perf_stats = {}
    cached_vectors = {}

    for strat_name, strat_chunks in strategies:
        print(f"\n-> Đang nạp vào Collection Chiến lược {strat_name}...")
        texts = [c.text for c in strat_chunks]
        
        t0 = time.time()
        # Tối ưu: Chiến lược C dùng lại chính xác vector của B để tiết kiệm quota và tăng tốc
        if strat_name == "C" and "B" in cached_vectors:
            vectors = cached_vectors["B"]
            print("   (Tái sử dụng vector từ Chiến lược B - cô lập biến số lọc)")
        else:
            vectors = embedder.embed_documents(texts)
            if strat_name == "B":
                cached_vectors["B"] = vectors

        store.upsert(strategy=strat_name, chunks=strat_chunks, vectors=vectors)
        elapsed = time.time() - t0

        perf_stats[strat_name] = {
            "chunk_count": len(strat_chunks),
            "ingest_time_seconds": round(elapsed, 2),
            "total_chars": sum(len(t) for t in texts)
        }
        print(f"   Hoàn thành lưu trữ Chiến lược {strat_name} trong {elapsed:.2f} giây.")

    # 4. Nạp bộ câu hỏi Benchmark
    benchmark_path = Path("benchmark/questions.json")
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {benchmark_path}")

    with open(benchmark_path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    print(f"\n[4/5] Đã nạp {len(questions)} câu hỏi benchmark (15 dev / 5 test).")

    # Tạo vector câu hỏi 1 lần duy nhất cho cả 3 chiến lược (đảm bảo tính nhất quán tuyệt đối)
    print("-> Đang tạo embedding cho 20 câu hỏi...")
    question_vectors = {}
    for q in questions:
        question_vectors[q["id"]] = embedder.embed_query(q["question"])

    # 5. Chạy truy xuất Top-3 và Đánh giá
    print("\n[5/5] Đang chạy truy xuất Top-3 và chấm điểm cho cả 3 chiến lược...")
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = {}

    for strat_name, _ in strategies:
        print(f"\nĐang đánh giá Chiến lược {strat_name}...")
        traces = []

        for q in questions:
            q_vec = question_vectors[q["id"]]

            # Cấu hình C: Áp dụng filter nếu câu hỏi có yêu cầu lọc
            filters = None
            if strat_name == "C" and q.get("filter"):
                filters = q["filter"]

            # Truy xuất Top-3 từ ChromaDB
            hits = store.search(
                strategy=strat_name,
                query_vector=q_vec,
                top_k=3,
                filters=filters
            )

            # Chấm điểm Hit@1, Hit@3
            query_trace = evaluate_query(q, hits)
            traces.append(query_trace)

        # Lưu trace chi tiết
        trace_file = results_dir / f"traces_strategy_{strat_name.lower()}.json"
        with open(trace_file, "w", encoding="utf-8") as f:
            json.dump(traces, f, ensure_ascii=False, indent=2)

        # Tổng hợp kết quả
        summary = summarize(traces)
        all_summaries[strat_name] = summary
        print(f"-> Chiến lược {strat_name}:")
        print(f"   [DEV]  Hit@1: {summary['dev']['hit1_ratio']} ({summary['dev']['hit1_pct']}%) | Hit@3: {summary['dev']['hit3_ratio']} ({summary['dev']['hit3_pct']}%)")
        print(f"   [TEST] Hit@1: {summary['test']['hit1_ratio']} ({summary['test']['hit1_pct']}%) | Hit@3: {summary['test']['hit3_ratio']} ({summary['test']['hit3_pct']}%)")

    # Lưu bản tóm tắt tổng thể
    summary_path = results_dir / "summary_metrics.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": embedder.model_name,
            "dimensions": embedder.dimensions,
            "performance": perf_stats,
            "metrics": all_summaries
        }, f, ensure_ascii=False, indent=2)

    # Cập nhật artifacts/experiment_log.csv
    csv_path = Path("artifacts/experiment_log.csv")
    csv_lines = [
        "strategy,chunk_method,chunk_size,overlap,filter_enabled,chunk_count,dev_hit1,dev_hit3,test_hit1,test_hit3,ingest_time_s",
        f"A,fixed_size,500,50,False,{perf_stats['A']['chunk_count']},{all_summaries['A']['dev']['hit1_ratio']},{all_summaries['A']['dev']['hit3_ratio']},{all_summaries['A']['test']['hit1_ratio']},{all_summaries['A']['test']['hit3_ratio']},{perf_stats['A']['ingest_time_seconds']}",
        f"B,by_paragraph,800,0,False,{perf_stats['B']['chunk_count']},{all_summaries['B']['dev']['hit1_ratio']},{all_summaries['B']['dev']['hit3_ratio']},{all_summaries['B']['test']['hit1_ratio']},{all_summaries['B']['test']['hit3_ratio']},{perf_stats['B']['ingest_time_seconds']}",
        f"C,by_paragraph,800,0,True,{perf_stats['C']['chunk_count']},{all_summaries['C']['dev']['hit1_ratio']},{all_summaries['C']['dev']['hit3_ratio']},{all_summaries['C']['test']['hit1_ratio']},{all_summaries['C']['test']['hit3_ratio']},{perf_stats['C']['ingest_time_seconds']}"
    ]
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(csv_lines))

    print("\n" + "=" * 60)
    print("HOÀN THÀNH TOÀN BỘ PIPELINE THÀNH CÔNG!")
    print(f"1. Chi tiết traces đã lưu tại: {results_dir.resolve()}")
    print(f"2. Bảng tổng hợp đã cập nhật: {csv_path.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
