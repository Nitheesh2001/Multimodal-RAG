import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
import google.generativeai as genai
from dotenv import load_dotenv
import os
import faiss
import numpy as np
from typing import List

# --- Load API key from .env ---
load_dotenv()
gemini_key = os.getenv("GEMINI_API_KEY")
# You should set your real embedding API key here if needed
embedding_key = os.getenv("EMBEDDING_API_KEY")  # Optional, depends on embedding provider

# --- Configure Gemini ---
if gemini_key:
    genai.configure(api_key=gemini_key)

# --- Constants ---
EMBEDDING_DIM = 1536  # Adjust according to your embedding model dimension

# --- Dummy embedder function - replace with your real embedding function ---
def get_text_embedding(text: str) -> np.ndarray:
    # Example: call your embedding API here and return np.array of shape (EMBEDDING_DIM,)
    # Below is a random vector for demonstration only — DO NOT USE IN PRODUCTION
    return np.random.rand(EMBEDDING_DIM).astype('float32')

# --- Text chunking helper ---
def chunk_text(text: str, max_len: int = 1000) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_len, len(text))
        if end < len(text):
            space_pos = text.rfind(' ', start, end)
            if space_pos > start:
                end = space_pos
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
    return chunks

# --- Streamlit UI ---
st.set_page_config(page_title="Multimodal RAG with FAISS - Single Submit", layout="wide")
st.title("📄🖼️✍️ Multimodal RAG - Upload PDF/Image/Text & Ask Question")

col1, col2 = st.columns(2)
with col1:
    pdf_file = st.file_uploader("📄 Upload a PDF document", type=["pdf"])
    text_input = st.text_area("✍️ Or enter text here", height=150)
with col2:
    image_file = st.file_uploader("🖼️ Upload an image", type=["jpg", "jpeg", "png"])

question = st.text_input("💬 Ask your question based on uploaded data:")

# --- Main processing ---
if gemini_key and st.button("Submit"):
    if not question.strip():
        st.warning("Please enter a question to ask.")
    else:
        context_texts = []

        # Extract and chunk PDF text
        if pdf_file:
            try:
                pdf_reader = PdfReader(pdf_file)
                full_text = "\n".join(page.extract_text() or "" for page in pdf_reader.pages)
                pdf_chunks = chunk_text(full_text)
                context_texts.extend(pdf_chunks)
            except Exception as e:
                st.error(f"Failed to read PDF: {e}")

        # Chunk text input
        if text_input.strip():
            text_chunks = chunk_text(text_input)
            context_texts.extend(text_chunks)

        if not context_texts and not image_file:
            st.warning("Please upload a PDF, enter text, or upload an image to provide context.")
        else:
            # Build FAISS index on the fly
            if context_texts:
                embeddings = np.array([get_text_embedding(t) for t in context_texts])
                index = faiss.IndexFlatL2(EMBEDDING_DIM)
                index.add(embeddings)
            else:
                index = None

            # Embed question and search relevant chunks
            retrieved_chunks = []
            if index and index.ntotal > 0:
                q_emb = get_text_embedding(question)
                D, I = index.search(np.array([q_emb]), k=min(3, index.ntotal))
                retrieved_chunks = [context_texts[i] for i in I[0] if i < len(context_texts)]

            # Prepare content for Gemini: retrieved chunks + image + question
            context_parts = []
            for chunk in retrieved_chunks:
                context_parts.append(chunk)

            if image_file:
                try:
                    image = Image.open(image_file)
                    context_parts.append(image)
                except Exception as e:
                    st.error(f"Failed to load image: {e}")

            context_parts.append(question)

            # Generate answer
            try:
                model = genai.GenerativeModel("models/gemini-1.5-flash")
                response = model.generate_content(contents=context_parts)
                st.subheader("🧠 Answer:")
                st.write(response.text)
            except Exception as e:
                st.error(f"Error generating response: {e}")
else:
    if not gemini_key:
        st.error("Gemini API Key not found. Please set it in the .env file.")
