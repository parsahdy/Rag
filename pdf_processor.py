import os
import fitz
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

def process_pdfs_with_pdfminer(data_dir, db_dir):
    from pdfminer.high_level import extract_text
    from bidi.algorithm import get_display
    from langchain.schema import Document
    import os
    
    if not os.path.exists(data_dir):
        return []
    
    documents = []
    
    try:
        for filename in os.listdir(data_dir):
            if filename.endswith('.pdf'):
                pdf_path = os.path.join(data_dir, filename)
                
                try:
                    text = extract_text(pdf_path)
                    
                    text = get_display(text)
                    
                    if text.strip():
                        metadata = {"source": filename}
                        documents.append(Document(page_content=text, metadata=metadata))
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
        
        return documents
    except Exception as e:
        print(f"Error accessing directory {data_dir}: {e}")
        return []

def get_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    
    return text