import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

st.set_page_config(layout="wide")
st.title("📘 RAG Document Chatbot")

# Load models
@st.cache_resource
def load_models():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    generator = pipeline("text2text-generation", model="google/flan-t5-base")
    return embed_model, generator

embed_model, generator = load_models()

# Extract PDF text
def extract_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

# Chunk text
def chunk_text(text, chunk_size=500):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

# Sidebar upload
uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    text = extract_text(uploaded_file)
    chunks = chunk_text(text)

    embeddings = embed_model.encode(chunks)
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    st.sidebar.success("Document indexed successfully!")

    query = st.text_input("Ask a question from the document")

    if query:
        query_vector = embed_model.encode([query])
        distances, indices = index.search(np.array(query_vector), k=3)

        context = " ".join([chunks[i] for i in indices[0]])

        prompt = f"""
        Answer the question using only the context below.

        Context:
        {context}

        Question:
        {query}
        """

        result = generator(prompt, max_length=256)
        st.write(result[0]["generated_text"])