import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        if text.strip():  
            pages.append({
                'page_num': page_num + 1,  
                'text': text
            })
    
    doc.close()
    return pages

def chunk_texts(pages, chunk_size=200, overlap=50):
    chunks = []
    chunk_id = 0
    
    for page in pages:
        words = page['text'].split()
        page_num = page['page_num']
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            if len(chunk_words) > 20: 
                chunk_text = ' '.join(chunk_words)
                chunks.append({
                    'page': page_num,
                    'chunk_id': chunk_id,
                    'text': chunk_text
                })
                chunk_id += 1
    
    return chunks

def build_embeddings(chunks):
    texts = [chunk['text'] for chunk in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True)
    
    dimension = embeddings.shape[1] 
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    
    return chunks, embeddings, index

def extract_and_embed(pdf_bytes):
    print("📄 Extracting text from PDF...")
    pages = extract_text_from_pdf(pdf_bytes)
    print(f"✅ Extracted {len(pages)} pages")
    
    print("✂️ Chunking text...")
    chunks = chunk_texts(pages)
    print(f"✅ Created {len(chunks)} chunks")
    
    print("🧠 Building embeddings...")
    chunks, embeddings, index = build_embeddings(chunks)
    print(f"✅ Built embeddings with shape {embeddings.shape}")
    
    return chunks, index

def embed_query(query):
    embedding = model.encode([query], convert_to_numpy=True)
    return embedding[0]

def search_similar_chunks(query, chunks, index, top_k=3):
    query_embedding = embed_query(query)
    query_embedding = query_embedding.reshape(1, -1).astype('float32')
    
    distances, indices = index.search(query_embedding, top_k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(chunks):  
            chunk = chunks[idx].copy()
            chunk['distance'] = float(distances[0][i])
            results.append(chunk)
    
    return results