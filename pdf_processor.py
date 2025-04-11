import os
import fitz
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

def process_pdfs(data, db_dir):
    pdf_files = [f for f in os.listdir(data) if f.endswith('.pdf')]
    
    if not pdf_files:
        return None
    
    all_chunks = []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "!", "؟", "،", " ", ""],  
        length_function=len,
    )
    
    for pdf_file in tqdm(pdf_files, desc="پردازش فایل‌های PDF"):
        pdf_path = os.path.join(data, pdf_file)
        text = get_pdf_text(pdf_path)
        
        
        print(f"نمونه متن از {pdf_file}: {text[:200]}")

        doc = Document(page_content=text, metadata={"source": pdf_file})
        
        chunks = text_splitter.split_documents([doc])
        
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk"] = i
            chunk.metadata["total_chunks"] = len(chunks)
        
        all_chunks.extend(chunks)
    
    print(f"Total documents processed: {len(pdf_files)}")
    print(f"Total chunks created: {len(all_chunks)}")
    
    return all_chunks

def get_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    
    return text