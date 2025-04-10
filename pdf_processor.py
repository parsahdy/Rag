import os
import fitz
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

def get_pdf_text(pdf_path):
    doc = fitz.Document(pdf_path)
    text = ""
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
        
    return text

def process_pdfs(data_dir, db_dir):
    pdf_files = [f for f in os.listdir(data_dir) if f.endswith('.pdf')]
    
    if not pdf_files:
        return None
    
    documents = []
    
    for pdf_file in tqdm(pdf_files, desc="پردازش فایل‌های PDF"):
        pdf_path = os.path.join(data_dir, pdf_file)
        text = get_pdf_text(pdf_path)
        
        metadata = {"source": pdf_file}
        
        doc = Document(page_content=text, metadata=metadata)
        documents.append(doc)
    
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  
    chunk_overlap=100,  
    length_function=len,
    )
    
   
    split_docs = text_splitter.split_documents(documents)
    
    from rag_manager import setup_embeddings
    embeddings = setup_embeddings()
    
    Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=db_dir
    )
    
    return split_docs


def get_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    
    print(f"نمونه متن از {pdf_path}: {text[:200]}")
        
    return text