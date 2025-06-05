1
# 📄🖼️✍️ Multimodal RAG App

This project is a **Multimodal Retrieval-Augmented Generation (RAG)** app built with **Streamlit**, using **Gemini 1.5 Flash** to answer questions based on:

- Uploaded **PDF documents**
- Uploaded **Images**
- Directly entered **Text**

All these modalities are processed, embedded, and queried to generate contextual answers using Google’s Gemini API. The app also uses **FAISS** for embedding and retrieval.

---

## 🚀 Features

- Upload **PDFs** and extract content automatically.
- Upload **images** and allow Gemini to understand visual content.
- Input **text** manually to provide additional context.
- Ask a **question** based on any or all of the above.
- All context is embedded and stored using **FAISS**.
- Powered by **Gemini 1.5 Flash** (via `google.generativeai`).

---

## 🧰 Tech Stack

- `Python 3.10+`
- `Streamlit`
- `Pillow`
- `PyPDF2`
- `google-generativeai`
- `python-dotenv`
- `faiss-cpu`

---

## 📦 Installation

1. **Clone the repository**:

```bash
git clone https://github.com/yourusername/multimodal-rag-app.git
cd multimodal-rag-app



2 Create a virtual environment:
python3 -m venv venv
source venv/bin/activate  # On Mac/Linux
venv\Scripts\activate     # On Windows


3 Install dependencies:
pip install -r requirements.txt


4 Set up your .env file:
Create a .env file in the root folder and add your Gemini API key:

GEMINI_API_KEY=your_google_generativeai_api_key


5 🧪 Running the App

streamlit run app.py


6 📁 Project Structure

multimodal_rag_app/
├── app.py                # Main Streamlit app
├── .env                  # API keys stored here
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation


7  🧠 About RAG

Retrieval-Augmented Generation (RAG) is a technique that enhances large language models (LLMs) by incorporating external knowledge sources at runtime. This allows the model to generate more accurate, up-to-date, and grounded responses.
This app simulates a lightweight RAG system by storing uploaded context (text, PDF, image) into memory, enabling Gemini to generate answers using multimodal context.