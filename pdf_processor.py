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
    
    documents = []
    
    for pdf_file in tqdm(pdf_files, desc="پردازش فایل‌های PDF"):
        pdf_path = os.path.join(data, pdf_file)
        text = get_pdf_text(pdf_path)
        
        metadata = {"source": pdf_file}
        
        doc = Document(page_content=text, metadata=metadata)
        documents.append(doc)
    
    
    return documents


def get_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
        
    return text