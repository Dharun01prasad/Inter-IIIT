# Smart Study Buddy

An AI-powered PDF learning assistant that helps students interact with study materials intelligently. Upload PDFs, ask contextual questions, view source citations, track activity history, and download study logs — powered by FastAPI, Gemini AI, and semantic search.

---

## Features

- Upload PDF documents
- Ask questions about uploaded PDFs
- Semantic chunk search using embeddings
- Page-wise source citations
- Voice input support
- Dashboard with activity history
- Download study activity as Excel
- FastAPI backend
- Interactive frontend UI
- Gemini 2.0 integration

---

## Tech Stack

### Backend
- Python
- FastAPI
- Gemini 2.0 Flash
- Pandas
- HuggingFace
- Uvicorn

### Frontend
- HTML
- CSS
- JavaScript

---

## Project Structure

```bash
Smart-Study-Buddy/
│
├── backend/
│   ├── main.py
│   ├── nlp/
│   ├── uploads/
│   └── requirements.txt
│
├── frontend/
│   ├── index.html
│   ├── dashboard.html
│   ├── images/
│   └── styles/
│
└── README.md
