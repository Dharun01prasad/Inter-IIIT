from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import google.generativeai as genai
import pandas as pd
from fastapi.responses import Response
import io
from datetime import datetime
import json

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nlp.processor import extract_and_embed, search_similar_chunks

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DOCS = []
INDEX = None
ACTIVITY_LOG = []
PDF_HISTORY = []
CURRENT_PDF_ID = None

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-2.0-flash-exp')

print("✅ Smart Study Buddy API ready with Gemini 2.0!")

class QueryRequest(BaseModel):
    query: str

class Citation(BaseModel):
    page: int
    text: str

class ActivityEntry(BaseModel):
    timestamp: str
    question: str
    answer: str
    sources: list
    context_chunks: int
    pdf_id: str
    citations: list = []

class PDFInfo(BaseModel):
    id: str
    filename: str
    upload_time: str
    chunks: int
    pages: int

@app.get("/")
async def root():
    return {"message": "Smart Study Buddy API", "status": "running"}

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload PDF and process it
    """
    global DOCS, INDEX, PDF_HISTORY, CURRENT_PDF_ID
    
    try:
        # Check file type
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Read PDF bytes
        pdf_bytes = await file.read()
        
        # Process PDF using Person 3's functions
        chunks, index = extract_and_embed(pdf_bytes)
        
        # Store globally
        DOCS = chunks
        INDEX = index
        
        # Extract page count
        pages = len(set([chunk['page'] for chunk in chunks]))
        
        # Generate unique PDF ID
        pdf_id = f"pdf_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        CURRENT_PDF_ID = pdf_id
        
        pdf_info = PDFInfo(
            id=pdf_id,
            filename=file.filename,
            upload_time=datetime.now().isoformat(),
            chunks=len(chunks),
            pages=pages
        )
        PDF_HISTORY.append(pdf_info.dict())
        
        return {
            "status": "success",
            "message": f"PDF uploaded successfully! Processed {len(chunks)} chunks from {pages} pages.",
            "chunks": len(chunks),
            "pages": pages,
            "pdf_id": pdf_id
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.post("/ask/")
async def ask_question(request: QueryRequest):

    global DOCS, INDEX, ACTIVITY_LOG, CURRENT_PDF_ID
    
    if not DOCS or INDEX is None:
        raise HTTPException(status_code=400, detail="Please upload a PDF first!")
    
    try:
        query = request.query
        
        results = search_similar_chunks(query, DOCS, INDEX, top_k=3)
        
        if not results:
            response_data = {
                "answer": "Could not find relevant information in the PDF.",
                "sources": [],
                "citations": []
            }
            
            activity_entry = ActivityEntry(
                timestamp=datetime.now().isoformat(),
                question=query,
                answer=response_data["answer"],
                sources=[],
                context_chunks=0,
                pdf_id=CURRENT_PDF_ID,
                citations=[]
            )
            ACTIVITY_LOG.append(activity_entry.dict())
            
            return response_data
        
        context_parts = []
        citations = []
        
        for chunk in results:
            context_parts.append(f"[Page {chunk['page']}]: {chunk['text']}")
            citations.append({
                "page": chunk['page'],
                "text": chunk['text']
            })
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""You are a helpful study assistant. Answer the question based on the context provided from a PDF document. 

IMPORTANT: Always cite the page numbers in your answer using the format [Page X].

Context from the document:
{context}

Question: {query}

Provide a clear, concise answer based on the context above. Make sure to cite relevant page numbers."""

        try:
            response = model.generate_content(prompt)
            answer = response.text
            print(f"✅ Gemini response generated")
        except Exception as e:
            print(f"❌ Gemini error: {str(e)}")
            answer = f"Based on the document:\n\n" + "\n\n".join([
                f"**Page {chunk['page']}:** {chunk['text'][:300]}..." 
                for chunk in results
            ])
        
        pages = sorted(list(set([chunk['page'] for chunk in results])))
        
        activity_entry = ActivityEntry(
            timestamp=datetime.now().isoformat(),
            question=query,
            answer=answer,
            sources=pages,
            context_chunks=len(results),
            pdf_id=CURRENT_PDF_ID,
            citations=citations
        )
        ACTIVITY_LOG.append(activity_entry.dict())
        
        return {
            "answer": answer,
            "sources": pages,
            "context_chunks": len(results),
            "citations": citations  
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.get("/download_activity/")
async def download_activity():
    if not ACTIVITY_LOG:
        raise HTTPException(status_code=404, detail="No activity data available")
    
    try:
        excel_data = []
        for activity in ACTIVITY_LOG:
            citations_text = ""
            if activity.get('citations'):
                citations_text = "\n\n---CITATIONS---\n" + "\n\n".join([
                    f"Page {cite['page']}:\n{cite['text']}"
                    for cite in activity['citations']
                ])
            
            excel_data.append({
                'Timestamp': activity['timestamp'],
                'Question': activity['question'],
                'Answer': activity['answer'],
                'Sources (Pages)': ', '.join(map(str, activity['sources'])),
                'Context Chunks': activity['context_chunks'],
                'PDF ID': activity.get('pdf_id', 'N/A'),
                'Source Citations': citations_text
            })
        
        df = pd.DataFrame(excel_data)
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Study Activity', index=False)
            
            worksheet = writer.sheets['Study Activity']
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 80)
                worksheet.column_dimensions[column_letter].width = adjusted_width
        
        output.seek(0)
        
        filename = f"study_buddy_activity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        return Response(
            content=output.getvalue(),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating Excel file: {str(e)}")



@app.get("/dashboard/")
async def get_dashboard():

    return {
        "pdf_history": PDF_HISTORY,
        "activity_log": ACTIVITY_LOG,
        "current_pdf": PDF_HISTORY[-1] if PDF_HISTORY else None,
        "stats": {
            "total_pdfs": len(PDF_HISTORY),
            "total_questions": len(ACTIVITY_LOG),
            "current_chunks": len(DOCS) if DOCS else 0
        }
    }

@app.get("/dashboard/pdf_history")
async def get_pdf_history():
    return {"pdf_history": PDF_HISTORY}

@app.get("/dashboard/activity")
async def get_activity(pdf_id: str = None):

    if pdf_id:
        filtered_activity = [activity for activity in ACTIVITY_LOG if activity['pdf_id'] == pdf_id]
        return {"activity_log": filtered_activity}
    else:
        return {"activity_log": ACTIVITY_LOG}

@app.get("/dashboard/stats")
async def get_stats():

    return {
        "total_pdfs": len(PDF_HISTORY),
        "total_questions": len(ACTIVITY_LOG),
        "current_chunks": len(DOCS) if DOCS else 0,
        "last_activity": ACTIVITY_LOG[-1]['timestamp'] if ACTIVITY_LOG else None
    }

@app.delete("/dashboard/clear_all")
async def clear_all_data():

    global DOCS, INDEX, ACTIVITY_LOG, PDF_HISTORY, CURRENT_PDF_ID
    DOCS = []
    INDEX = None
    CURRENT_PDF_ID = None
    activity_count = len(ACTIVITY_LOG)
    pdf_count = len(PDF_HISTORY)
    ACTIVITY_LOG = []
    PDF_HISTORY = []
    
    return {
        "status": "success",
        "cleared_activities": activity_count,
        "cleared_pdfs": pdf_count
    }

@app.get("/status/")
async def get_status():

    return {
        "pdf_loaded": len(DOCS) > 0,
        "chunks": len(DOCS) if DOCS else 0,
        "activity_entries": len(ACTIVITY_LOG)
    }

@app.get("/clear_activity/")
async def clear_activity():

    global ACTIVITY_LOG
    count = len(ACTIVITY_LOG)
    ACTIVITY_LOG = []
    return {"status": "success", "cleared_entries": count}

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("🚀 Smart Study Buddy - Server Starting!")
    print("="*60)
    print("📊 Backend API:  http://localhost:8000")
    print("🎨 Frontend URL: http://localhost:3000")
    print("📈 Dashboard:    http://localhost:3000/dashboard.html")
    print("="*60)
    print("💡 To start the frontend, run: python -m http.server 3000")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)