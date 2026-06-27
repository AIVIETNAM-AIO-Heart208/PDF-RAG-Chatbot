# PDF RAG Chatbot

Ứng dụng hỏi đáp tài liệu PDF bằng mô hình chạy local qua Ollama. Pipeline gồm đọc PDF, chia nội dung thành chunks, tạo embedding, lưu vào ChromaDB và truy vấn ngữ cảnh để tạo câu trả lời bằng LLM.

## Tính năng

- Upload file PDF trực tiếp trên giao diện Streamlit.
- Trích xuất text từ PDF bằng `pypdf`.
- Chia văn bản thành các chunks có overlap để giữ ngữ cảnh.
- Tạo embedding bằng Ollama embedding model.
- Lưu và truy vấn vector bằng ChromaDB.
- Trả lời câu hỏi bằng LLM local qua Ollama.
- Có xử lý lỗi cho PDF rỗng, PDF không đọc được, Ollama chưa chạy, thiếu model và lỗi truy vấn.

## Yêu cầu

- Python 3.10 hoặc mới hơn.
- Ollama đã được cài và đang chạy.
- Các model Ollama cần dùng:

```powershell
ollama pull bge-m3
ollama pull vicuna:7b-v1.5-q5_1
```

Nếu model `vicuna:7b-v1.5-q5_1` tải lỗi hoặc máy không đủ tài nguyên, có thể dùng model khác như `llama3.2` hoặc `mistral` bằng cách đặt biến môi trường `LLM_MODEL`.

## Cài đặt

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Chạy ứng dụng

```powershell
streamlit run chatbot_app.py
```

Sau khi ứng dụng mở trong trình duyệt:

1. Upload một file PDF.
2. Bấm `Xử lý PDF`.
3. Đặt câu hỏi về nội dung tài liệu.

## Tùy chỉnh model

Mặc định project dùng:

- LLM: `vicuna:7b-v1.5-q5_1`
- Embedding: `bge-m3`

Có thể đổi model bằng biến môi trường:

```powershell
$env:LLM_MODEL="llama3.2"
$env:EMBED_MODEL="bge-m3"
streamlit run chatbot_app.py
```

## Cấu trúc chính

```text
.
├── chatbot_app.py       # Streamlit app và RAG pipeline
├── requirements.txt     # Python dependencies
├── README.md            # Hướng dẫn project
└── .gitignore           # File/folder local không đưa lên Git
```

## Lưu ý

- PDF dạng scan ảnh thường không có text để `pypdf` trích xuất. Với loại file đó cần OCR trước.
- ChromaDB trong project này đang chạy dạng in-memory, phù hợp cho demo và bài tập nhỏ.
- Câu trả lời phụ thuộc vào chất lượng text trích xuất từ PDF, model embedding và model LLM đã chọn.
