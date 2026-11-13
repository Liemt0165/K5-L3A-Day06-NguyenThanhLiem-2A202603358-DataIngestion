# Báo cáo cá nhân — Day 6: Data Ingestion & Retrieval Evaluation

- **Họ và tên**: Nguyễn Thanh Liêm
- **MSSV**: 2A202603358
- **Lớp**: L3A

---

## 1. Dữ liệu và pipeline

* **Chủ đề, nguồn/quyền, số trang và quy tắc tính trang**:
  * Chủ đề: Hệ thống RAG và Bộ nhớ người dùng trong AI Agent.
  * Tài liệu: Trích xuất trọn vẹn Chương 3 (*Bộ nhớ người dùng và cơ sở kiến thức*) từ sách *Hiểu sâu về AI Agent: Nguyên lý thiết kế và thực hành kỹ thuật* (`data/AI-Agents-Chapter3.pdf`).
  * Giấy phép: Tài liệu học thuật tiếng Việt phục vụ nghiên cứu và thực hành cá nhân.
  * Quy tắc tính trang: Lấy đúng 40 trang kỹ thuật (tương ứng từ trang 106 đến 145 của sách gốc), loại bỏ hoàn toàn trang bìa và mục lục. Tổng số ký tự sạch sau chuẩn hóa: 129.035 ký tự (trung bình 3.225 ký tự/trang).

* **Chuẩn hóa, lỗi và giới hạn**:
  * Chuẩn hóa: Sử dụng Unicode `NFKC`, thu gọn các khoảng trắng thừa và dòng trống liên tiếp.
  * Lỗi font trích xuất: Font nhúng trong PDF khiến các dấu ngoặc kép `“` và `”` bị ánh xạ nhầm thành ký tự `o` và `p` (ví dụ `ocông typ` thay cho `“công ty”`). Các ký hiệu toán học và cấu trúc bảng (Bảng 3-2) được giữ nguyên vẹn.
  * Giới hạn: Chưa xử lý trích xuất hình vẽ sơ đồ (như Hình 3-7 HNSW, Hình 3-9 RRF), văn bản chỉ lấy phần text giải thích.

* **Embedding model, Vector store, lệnh chạy và chi phí**:
  * Embedding model: `Qwen3-Embedding-8B` (8 tỷ tham số, vector 4096 chiều; provider DeepInfra nhận diện qua ID `Qwen/Qwen3-Embedding-8B`).
  * Vector store: `ChromaDB v1.5.9` (Persistent local storage, metric Cosine distance: `hnsw:space = cosine`).
  * Lệnh chạy tái lập: `.venv\Scripts\python.exe -m starter.src.run`.
  * Chi phí API: Khoảng 0.0003 USD (~10 đồng VNĐ) cho toàn bộ 20 câu hỏi và 3 chiến lược.

* **Metadata bắt buộc, trường lọc bổ sung, nguồn giá trị và lý do chọn**:
  * Metadata bắt buộc: `document_id` (`"ai-agents-chapter3"`), `source` (`"data/AI-Agents-Chapter3.pdf"`), `page` (1–40).
  * Trường lọc bổ sung: `section` nhận 4 giá trị dựa trên 4 đề mục lớn của chương sách:
    * `user_memory` (Trang 1–11): Hệ thống bộ nhớ người dùng.
    * `rag_fundamentals` (Trang 12–21): Cơ bản về RAG, Chunking và Vector DB.
    * `advanced_retrieval` (Trang 22–38): Kỹ thuật truy xuất nâng cao và RAG thông minh.
    * `summary` (Trang 39–40): Tóm tắt chương và câu hỏi tư duy.
  * Lý do chọn: Cho phép cô lập không gian tìm kiếm chính xác theo ngữ cảnh câu hỏi của người dùng.

---

## 2. Cấu hình và benchmark

* **Ba cấu hình A / B / C (Thí nghiệm có kiểm soát)**:
  * **Cấu hình A (Fixed-size)**: Cắt cố định 500 ký tự, overlap 50 ký tự; gắn metadata nguồn và trang; không lọc (sinh ra 303 chunks).
  * **Cấu hình B (By-paragraph)**: Cắt theo ranh giới đoạn văn (`\n\n`), gom đoạn nhỏ, giới hạn tối đa 800 ký tự; cùng metadata; không lọc (sinh ra 184 chunks).
  * **Cấu hình C (B + Filter)**: Dùng lại chính xác 184 chunks của B; chỉ áp dụng bộ lọc `where={"section": ...}` khi câu hỏi có filter (cô lập duy nhất biến số lọc metadata).

* **Bộ benchmark $\ge 20$ câu**:
  * Phân chia: 15 câu Dev (`q01`–`q15`) và 5 câu Test (`q16`–`q20`).
  * Câu lọc metadata: 4 câu (`q02`, `q04`, `q08` trong Dev; `q17` trong Test).
  * Kiểm chứng evidence: Mỗi câu hỏi đều trích dẫn một câu văn hoàn chỉnh tồn tại thực tế tại `gold_page` trong PDF để hàm `is_hit()` đối soát tự động.

* **Tiêu chí chọn chiến lược đã chốt trước khi xem test**:
  * Ưu tiên 1: Evidence Hit@3 trên tập Dev cao nhất.
  * Ưu tiên 2: Tính mạch lạc Chunk Coherence cao (tối thiểu $\ge 1.5/2.0$).
  * Ưu tiên 3: Tối ưu dung lượng lưu trữ vector và thời gian nạp dữ liệu.

---

## 3. Kết quả

### Bảng 1: Hiệu năng và Độ chính xác truy xuất

| Strategy | Split | Hit@1 (n/N) | Evidence Hit@3 (n/N) | Chunk count | Thời gian | Chi phí |
|---|---|:---:|:---:|---:|---:|---:|
| **A** | Dev | 6/15 (40.0%) | 8/15 (53.3%) | 303 | 28.79 s | $\approx \$0.0003$ |
| **A** | Test | 4/5 (80.0%) | 4/5 (80.0%) | 303 | 28.79 s | $\approx \$0.0003$ |
| **B** | Dev | 7/15 (46.7%) | 8/15 (53.3%) | 184 | 7.69 s | $\approx \$0.0002$ |
| **B** | Test | 3/5 (60.0%) | 4/5 (80.0%) | 184 | 7.69 s | $\approx \$0.0002$ |
| **C** | Dev | 7/15 (46.7%) | 8/15 (53.3%) | 184 | 0.30 s | $\approx \$0.0002$ |
| **C** | Test | 3/5 (60.0%) | 4/5 (80.0%) | 184 | 0.30 s | $\approx \$0.0002$ |

### Bảng 2: Đánh giá định tính thủ công

| Strategy | Coherence trung bình / 2.0 ($n=10$) | Metadata đủ và đúng ($n=15$) | Filter tuân thủ ($n/N$) | Grounding 0/1/2 (số câu) và trung bình / 2.0 |
|---|:---:|:---:|:---:|:---:|
| **A** | 1.20 / 2.0 | 15/15 (100%) | N/A | 1 câu 0đ, 0 câu 1đ, 4 câu 2đ $\rightarrow$ **1.60 / 2.0** |
| **B** | 1.85 / 2.0 | 15/15 (100%) | N/A | 1 câu 0đ, 1 câu 1đ, 3 câu 2đ $\rightarrow$ **1.40 / 2.0** |
| **C** | 1.85 / 2.0 | 15/15 (100%) | 3/3 (100%) | 1 câu 0đ, 1 câu 1đ, 3 câu 2đ $\rightarrow$ **1.40 / 2.0** |

* **Đường dẫn lưu trữ**:
  * Traces chi tiết: `results/traces_strategy_a.json`, `traces_strategy_b.json`, `traces_strategy_c.json`.
  * Nhật ký tham số: `artifacts/experiment_log.csv`.
  * Mẫu kiểm tra Coherence: Chọn 10 vị trí đoạn văn tại các trang [1, 4, 7, 12, 14, 16, 23, 27, 34, 38].

---

## 4. Phân tích và kết luận

* **So sánh A–B và B–C; 2 ví dụ sai có ID câu và nguồn**:
  * *So sánh A vs B*: Chiến lược B giảm 39.3% số lượng chunk (184 vs 303) và tốc độ nạp nhanh gấp gần 4 lần (7.69s vs 28.79s). Tính mạch lạc Coherence của B vượt trội (1.85 vs 1.20) do không bị cắt đôi câu văn. Trên tập Test, cả A và B đều đạt Hit@3 là 80.0% (4/5 câu).
  * *So sánh B vs C*: Trên câu hỏi có filter `q17` (`section: advanced_retrieval`), Cấu hình C đã kích hoạt bộ lọc ép toàn bộ Top-3 kết quả nằm trong đúng phạm vi các trang 36–38, ngăn ngừa các trang lý thuyết mở đầu lọt vào kết quả.
  * *Ví dụ sai 1 (`q19` - Cả 3 cấu hình đều sai ở Test)*: Câu hỏi hỏi về 2 thành phần bổ trợ trong phần tóm tắt chương ở Trang 39. Tuy nhiên, Top-1 trả về lại rơi vào Trang 2 và Trang 4 do các trang này chứa mật độ từ khóa *"bộ nhớ người dùng"* và *"cơ sở kiến thức"* lặp lại rất cao.
  * *Ví dụ sai 2 (`q03` - Chiến lược A sai ở Dev)*: Câu hỏi về 4 định dạng lưu trữ bộ nhớ ở Trang 4. Cấu hình A cắt 500 ký tự làm tách rời danh sách gạch đầu dòng, khiến điểm tương đồng bị chia nhỏ và rớt khỏi Top-3.

* **Chiến lược chọn, đánh đổi, điều kiện áp dụng**:
  * **Chiến lược được chọn: Chiến lược C (By-paragraph + Metadata Filter)**.
  * Lý do: Đạt tính mạch lạc cao nhất (1.85/2.0), tiết kiệm tài nguyên lưu trữ, tốc độ nạp nhanh, đạt 80% Hit@3 trên tập Test, và đảm bảo kết quả truy xuất luôn tuân thủ đúng phân đoạn ngữ cảnh thông qua metadata filter.
  * Đánh đổi: Cắt theo đoạn văn đòi hỏi văn bản gốc phải có cấu trúc phân đoạn rõ ràng (`\n\n`), không phù hợp với văn bản OCR thô chưa phân đoạn.

* **Hạn chế của dữ liệu, nhãn, test nhỏ, đo thời gian/chi phí**:
  * Tập Test quy mô 5 câu chỉ mang tính chất minh họa đại diện (sample demonstration), độ tin cậy thống kê chưa thể khái quát hóa cho toàn bộ hệ thống lớn.
  * Hiện tượng lỗi font encoding PDF (`o...p`) gây nhiễu nhẹ cho các thuật toán so khớp chuỗi chính xác.

* **Nếu làm lại hoặc mở rộng, cải tiến gì?**:
  * Mở rộng kỹ thuật **Hybrid Retrieval**: Kết hợp dense vector của `Qwen3-Embedding-8B` với sparse lexical search `BM25` để xử lý các câu hỏi chứa tên riêng/mã hiệu chính xác.
  * Áp dụng **Cross-Encoder Re-ranker** (như `bge-reranker-large`) cho Top-10 trước khi lấy Top-3 để khắc phục triệt để lỗi của câu `q19`.
