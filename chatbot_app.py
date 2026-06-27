import os
import tempfile
import time

import chromadb
import ollama
import pypdf
import streamlit as st


LLM_MODEL = os.getenv("LLM_MODEL", "vicuna:7b-v1.5-q5_1")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4

PROMPT = """Bạn là trợ lý hỏi đáp tài liệu. Chỉ dùng các đoạn ngữ cảnh dưới đây để trả lời câu hỏi.
Nếu ngữ cảnh không có thông tin, hãy nói rằng bạn không biết và không bịa thêm.
Trả lời ngắn gọn, chính xác, bằng tiếng Việt.

Ngữ cảnh:
{context}

Câu hỏi: {question}

Trả lời:"""


st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_state():
    defaults = {
        "collection": None,
        "pdf_name": "",
        "chat_history": [],
        "chunk_count": 0,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def embed(texts):
    if not texts:
        raise ValueError("Không có nội dung để embedding.")

    try:
        response = ollama.embed(model=EMBED_MODEL, input=texts)
    except Exception as exc:
        raise RuntimeError(
            f"Không thể tạo embedding bằng model '{EMBED_MODEL}'. "
            "Hãy kiểm tra Ollama đang chạy và model đã được pull."
        ) from exc

    embeddings = response.get("embeddings")
    if not embeddings:
        raise RuntimeError("Ollama không trả về embedding hợp lệ.")
    return embeddings


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    paragraphs = [paragraph.strip() for paragraph in text.splitlines() if paragraph.strip()]
    chunks = []
    current = ""

    for paragraph in paragraphs:
        while len(paragraph) > size:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.append(paragraph[:size].strip())
            paragraph = paragraph[size - overlap :]

        if len(current) + len(paragraph) + 1 <= size:
            current += paragraph + "\n"
            continue

        if current:
            chunks.append(current.strip())
        current = (current[-overlap:] + paragraph + "\n") if overlap else (paragraph + "\n")

    if current.strip():
        chunks.append(current.strip())
    return chunks


def extract_pdf_text(uploaded_file):
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name

        reader = pypdf.PdfReader(temp_path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as exc:
        raise RuntimeError("Không thể đọc nội dung từ file PDF này.") from exc
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

    text = text.strip()
    if not text:
        raise ValueError(
            "PDF không có text để trích xuất. Nếu đây là file scan ảnh, cần OCR trước khi dùng RAG."
        )
    return text


def process_pdf(uploaded_file):
    text = extract_pdf_text(uploaded_file)
    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("Không tạo được chunk nào từ PDF.")

    collection_name = f"rag_{int(time.time())}"
    collection = chromadb.Client().get_or_create_collection(collection_name)
    collection.add(
        ids=[str(index) for index in range(len(chunks))],
        documents=chunks,
        embeddings=embed(chunks),
    )
    return collection, len(chunks)


def rag(question, collection, k=TOP_K):
    if not question.strip():
        raise ValueError("Câu hỏi đang trống.")
    if collection is None:
        raise ValueError("Bạn cần xử lý PDF trước khi đặt câu hỏi.")

    try:
        result = collection.query(query_embeddings=embed([question]), n_results=k)
        documents = result.get("documents") or [[]]
        context = "\n\n".join(documents[0])
    except Exception as exc:
        raise RuntimeError("Không thể truy vấn dữ liệu từ ChromaDB.") from exc

    if not context.strip():
        return "Mình không tìm thấy ngữ cảnh phù hợp trong tài liệu để trả lời câu hỏi này."

    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": PROMPT.format(context=context, question=question),
                }
            ],
            options={"temperature": 0},
        )
    except Exception as exc:
        raise RuntimeError(
            f"Không thể gọi model '{LLM_MODEL}'. Hãy kiểm tra Ollama đang chạy và model đã được pull."
        ) from exc

    answer = response.get("message", {}).get("content", "").strip()
    if not answer:
        raise RuntimeError("Model không trả về câu trả lời hợp lệ.")
    return answer


init_state()

st.title("PDF RAG Assistant")

with st.sidebar:
    st.subheader("📄 Tài liệu")
    uploaded_file = st.file_uploader("Chọn file PDF", type="pdf")

    if uploaded_file and st.button("🔄 Xử lý PDF", use_container_width=True):
        with st.spinner("Đang đọc PDF, tạo chunks và embedding..."):
            try:
                collection, chunk_count = process_pdf(uploaded_file)
            except Exception as exc:
                st.session_state.collection = None
                st.session_state.pdf_name = ""
                st.session_state.chunk_count = 0
                st.error(str(exc))
            else:
                st.session_state.collection = collection
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.chunk_count = chunk_count
                st.session_state.chat_history = []
                st.success(f"✅ Đã xử lý {chunk_count} chunks.")

    if st.session_state.pdf_name:
        st.info(f"📄 Đang dùng: {st.session_state.pdf_name}")
        st.caption(f"Chunks: {st.session_state.chunk_count}")
    else:
        st.info("📄 Chưa có tài liệu.")

    st.divider()
    st.caption(f"LLM: {LLM_MODEL}")
    st.caption(f"Embedding: {EMBED_MODEL}")

    if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
        st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.collection is None:
    st.info("🔄 Upload và xử lý PDF trước khi chat.")
    st.chat_input("Nhập câu hỏi...", disabled=True)
else:
    question = st.chat_input("Nhập câu hỏi của bạn...")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Đang tìm ngữ cảnh và tạo câu trả lời..."):
                try:
                    answer = rag(question, st.session_state.collection)
                except Exception as exc:
                    answer = f"Lỗi: {exc}"
                    st.error(answer)
                else:
                    st.write(answer)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
