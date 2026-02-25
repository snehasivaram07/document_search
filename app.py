import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

st.set_page_config(layout="wide")
st.title("📘 RAG Document Chatbot (TF-IDF Based)")

# --------------- PDF PROCESSING ---------------
def extract_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def chunk_text(text, chunk_size=600):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


# --------------- SIDEBAR: UPLOAD ---------------
uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    text = extract_text(uploaded_file)

    if not text.strip():
        st.error("Couldn't extract any text from this PDF.")
    else:
        chunks = chunk_text(text)

        # Fit TF-IDF on document chunks
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(chunks)

        st.sidebar.success(f"Document indexed successfully with {len(chunks)} chunks ✅")

        # --------------- QUESTION INPUT ---------------
        query = st.text_input("Ask a question from the document")

        if query:
            # Vectorize query
            query_vec = vectorizer.transform([query])

            # Cosine similarity between query and all chunks
            scores = cosine_similarity(query_vec, tfidf_matrix)[0]

            # Top 3 most relevant chunks
            top_k = 3
            top_indices = np.argsort(scores)[::-1][:top_k]

            relevant_chunks = []
            for idx in top_indices:
                if scores[idx] > 0:
                    relevant_chunks.append(chunks[idx])

            if relevant_chunks:
                st.subheader("Answer from document:")
                # Just join the best chunks as the answer
                st.write(" ".join(relevant_chunks))

                with st.expander("Show supporting context"):
                    for i, c in enumerate(relevant_chunks, start=1):
                        st.markdown(f"**Chunk {i}:**")
                        st.write(c)
                        st.markdown("---")
            else:
                st.warning("I couldn't find anything relevant to your question in this document.")
else:
    st.info("📄 Please upload a PDF from the sidebar to start.")