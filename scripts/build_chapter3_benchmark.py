import json
import re
from pypdf import PdfReader

reader = PdfReader("data/AI-Agents-Chapter3.pdf")

def get_page_text(p_num):
    txt = reader.pages[p_num - 1].extract_text() or ""
    return " ".join(txt.split())

def find_best_sentence(page_num, keyword):
    clean_txt = get_page_text(page_num)
    sents = re.split(r'(?<=[.!?])\s+', clean_txt)
    for s in sents:
        if keyword.lower() in s.lower():
            return s.strip()
    return clean_txt[:200]

questions_spec = [
    {
        "id": "q01",
        "split": "dev",
        "type": "truc_tiep",
        "page": 1,
        "question": "Chương 3 giải quyết vấn đề cốt lõi nào của Agent sau khi cuộc trò chuyện kết thúc?",
        "expected_answer": "Làm thế nào để Agent vẫn ghi nhớ người dùng và ghi nhớ kiến thức sau khi cuộc trò chuyện kết thúc.",
        "kw": "làm thế nào để Agent vẫn ghi nhớ",
        "filter": None
    },
    {
        "id": "q02",
        "split": "dev",
        "type": "metadata_filter",
        "page": 2,
        "question": "Theo phần bộ nhớ người dùng (section: user_memory), khung ba cấp độ đánh giá khả năng ghi nhớ bao gồm những gì?",
        "expected_answer": "Ghi nhớ chính xác (từng từ), ghi nhớ ngữ nghĩa (hiểu ý nghĩa), và suy luận tích hợp (tổng hợp qua nhiều cuộc trò chuyện).",
        "kw": "khung ba cấp độ",
        "filter": {"section": "user_memory"}
    },
    {
        "id": "q03",
        "split": "dev",
        "type": "truc_tiep",
        "page": 4,
        "question": "Bốn định dạng lưu trữ phổ biến cho bộ nhớ người dùng là những định dạng nào?",
        "expected_answer": "Văn bản tự nhiên, cặp khóa-giá trị, đồ thị tri thức (triples) và biểu diễn vector.",
        "kw": "Bốn định dạng lưu trữ",
        "filter": None
    },
    {
        "id": "q04",
        "split": "dev",
        "type": "metadata_filter",
        "page": 7,
        "question": "Trong phần bộ nhớ người dùng (section: user_memory), bộ nhớ làm việc trong tâm lý học nhận thức tương đương với thành phần nào của Agent?",
        "expected_answer": "Tương đương với cửa sổ ngữ cảnh (context window) của mô hình lớn.",
        "kw": "Bộ nhớ làm việc tương ứng",
        "filter": {"section": "user_memory"}
    },
    {
        "id": "q05",
        "split": "dev",
        "type": "truc_tiep",
        "page": 11,
        "question": "Mục đích của việc giải mẫn cảm nhật ký (log de-identification) trong bảo vệ quyền riêng tư bộ nhớ là gì?",
        "expected_answer": "Loại bỏ hoặc che giấu thông tin nhận dạng cá nhân (PII) trước khi đưa vào bộ nhớ lâu dài.",
        "kw": "Giải mẫn cảm nhật ký",
        "filter": None
    },
    {
        "id": "q06",
        "split": "dev",
        "type": "truc_tiep",
        "page": 12,
        "question": "Mục tiêu cơ bản của việc xây dựng quy trình RAG cho Agent là gì?",
        "expected_answer": "Mở rộng khả năng tiếp thu kiến thức ngoài dữ liệu huấn luyện thông qua truy xuất ngoài.",
        "kw": "tiếp thu kiến thức",
        "filter": None
    },
    {
        "id": "q07",
        "split": "dev",
        "type": "truc_tiep",
        "page": 13,
        "question": "Tại sao phân đoạn tài liệu (Chunking) lại là bước then chốt quyết định hiệu quả của RAG?",
        "expected_answer": "Vì kích thước chunk ảnh hưởng trực tiếp đến độ chính xác truy xuất và tính toàn vẹn của ngữ cảnh.",
        "kw": "Phân đoạn tài liệu",
        "filter": None
    },
    {
        "id": "q08",
        "split": "dev",
        "type": "metadata_filter",
        "page": 14,
        "question": "Theo phần kiến thức cơ bản RAG (section: rag_fundamentals), ưu điểm vượt trội của Dense Embedding so với từ vựng truyền thống là gì?",
        "expected_answer": "Hiểu được ngữ nghĩa tương đồng ngay cả khi không dùng chung từ khóa.",
        "kw": "Nhúng dày đặc sử dụng phương pháp học sâu",
        "filter": {"section": "rag_fundamentals"}
    },
    {
        "id": "q09",
        "split": "dev",
        "type": "truc_tiep",
        "page": 16,
        "question": "Bảng 3-2 so sánh hai thuật toán chỉ mục vector phổ biến nào theo các khía cạnh tốc độ và độ chính xác?",
        "expected_answer": "So sánh hai thuật toán chỉ mục ANNOY và HNSW.",
        "kw": "Bảng 3-2 so sánh",
        "filter": None
    },
    {
        "id": "q10",
        "split": "dev",
        "type": "truc_tiep",
        "page": 19,
        "question": "Tại sao tìm kiếm kết hợp (Hybrid Search) lại kết hợp cả nhúng dày đặc (dense) và nhúng thưa thớt (sparse)?",
        "expected_answer": "Để tận dụng đồng thời khả năng hiểu ngữ nghĩa của Dense Embedding và khả năng khớp từ khóa chính xác của BM25.",
        "kw": "Cả hai phương pháp đều có điểm mù",
        "filter": None
    },
    {
        "id": "q11",
        "split": "dev",
        "type": "dieu_kien",
        "page": 20,
        "question": "Khi áp dụng Reciprocal Rank Fusion (RRF) trong Hybrid Search, các danh sách xếp hạng được kết hợp dựa trên nguyên lý nào?",
        "expected_answer": "Dựa trên thứ hạng nghịch đảo của từng kết quả trong các danh sách khác nhau.",
        "kw": "RRF",
        "filter": None
    },
    {
        "id": "q12",
        "split": "dev",
        "type": "truc_tiep",
        "page": 23,
        "question": "Mục tiêu của lập chỉ mục có cấu trúc (Structured Indexing) là gì?",
        "expected_answer": "Chuyển từ truy xuất thông tin đơn thuần sang mô hình hóa kiến thức phân cấp hoặc đồ thị.",
        "kw": "Lập chỉ mục có cấu trúc",
        "filter": None
    },
    {
        "id": "q13",
        "split": "dev",
        "type": "dien_dat_lai",
        "page": 26,
        "question": "Mô hình hệ thống tập tin tổ chức kiến thức của Agent như thế nào?",
        "expected_answer": "Tổ chức kiến thức theo cấu trúc cây thư mục và tệp tin để dễ dàng duyệt và quản lý.",
        "kw": "tổ chức kiến thức với cấu trúc thư mục",
        "filter": None
    },
    {
        "id": "q14",
        "split": "dev",
        "type": "truc_tiep",
        "page": 27,
        "question": "Cập nhật gia tăng (Incremental Update) cho cơ sở tri thức giúp giải quyết vấn đề gì?",
        "expected_answer": "Cập nhật các mẩu thông tin mới mà không cần phải lập lại chỉ mục toàn bộ kho tài liệu.",
        "kw": "Cập nhật gia tăng",
        "filter": None
    },
    {
        "id": "q15",
        "split": "dev",
        "type": "truc_tiep",
        "page": 30,
        "question": "Sự chuyển đổi mô hình của RAG thông minh (Agentic RAG) là gì?",
        "expected_answer": "Biến việc truy xuất kiến thức thành một công cụ hành động có phản hồi và lập kế hoạch.",
        "kw": "RAG thông minh",
        "filter": None
    },
    {
        "id": "q16",
        "split": "test",
        "type": "truc_tiep",
        "page": 34,
        "question": "Kỹ thuật truy xuất theo ngữ cảnh (Contextual Retrieval) giải quyết hạn chế gì của các chunk độc lập?",
        "expected_answer": "Bổ sung ngữ cảnh toàn cục của tài liệu vào từng chunk riêng lẻ để chunk không bị thiếu bối cảnh.",
        "kw": "Truy xuất theo ngữ cảnh",
        "filter": None
    },
    {
        "id": "q17",
        "split": "test",
        "type": "metadata_filter",
        "page": 36,
        "question": "Theo phần truy xuất nâng cao (section: advanced_retrieval), mục tiêu của trích xuất tri thức sâu là gì?",
        "expected_answer": "Chuyển từ truy xuất thông tin bề mặt sang khám phá tri thức và mối quan hệ ẩn.",
        "kw": "Trích xuất tri thức sâu",
        "filter": {"section": "advanced_retrieval"}
    },
    {
        "id": "q18",
        "split": "test",
        "type": "truc_tiep",
        "page": 38,
        "question": "Khám phá tiên phong về bộ nhớ đa phương thức mở rộng sang các dạng dữ liệu nào?",
        "expected_answer": "Bao gồm hình ảnh, âm thanh và video bên cạnh văn bản truyền thống.",
        "kw": "Bộ nhớ đa phương thức",
        "filter": None
    },
    {
        "id": "q19",
        "split": "test",
        "type": "dien_dat_lai",
        "page": 39,
        "question": "Trong phần tóm tắt chương 3, hai thành phần bổ trợ cho nhau của hệ thống ghi nhớ là gì?",
        "expected_answer": "Bộ nhớ người dùng cá nhân hóa và cơ sở kiến thức RAG bên ngoài.",
        "kw": "Tóm tắt chương này",
        "filter": None
    },
    {
        "id": "q20",
        "split": "test",
        "type": "dieu_kien",
        "page": 40,
        "question": "Câu hỏi tư duy số 1 ở cuối chương 3 yêu cầu giải quyết tình huống nào trong hệ thống bộ nhớ?",
        "expected_answer": "Xử lý xung đột thông tin khi cùng một người dùng cung cấp thông tin mâu thuẫn trong các phiên khác nhau.",
        "kw": "Trong hệ thống bộ nhớ người dùng",
        "filter": None
    }
]

clean_questions = []
for spec in questions_spec:
    sentence = find_best_sentence(spec["page"], spec["kw"])
    clean_questions.append({
        "id": spec["id"],
        "split": spec["split"],
        "type": spec["type"],
        "question": spec["question"],
        "expected_answer": spec["expected_answer"],
        "gold_source_id": "ai-agents-chapter3",
        "gold_page": spec["page"],
        "evidence_quote": sentence,
        "filter": spec["filter"]
    })

with open("benchmark/questions.json", "w", encoding="utf-8") as f:
    json.dump(clean_questions, f, ensure_ascii=False, indent=2)

print("Đã làm sạch hoàn toàn 20 câu hỏi trong benchmark/questions.json!")
