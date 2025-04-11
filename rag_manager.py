import os
import shutil
import time

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub
from langchain.text_splitter import RecursiveCharacterTextSplitter

import torch

import bidi.algorithm as bidi  
from dotenv import load_dotenv

from pdf_processor import process_pdfs

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

load_dotenv()
token = os.getenv("HUGGINGFACE_API_TOKEN")

os.environ["HUGGINGFACEHUB_API_TOKEN"] = token

def setup_embeddings():
    model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    model_kwargs = {"device": DEVICE}
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs={"normalize_embeddings": True}
    )
    
    return embeddings

def get_llm():
    llm = HuggingFaceHub(
        repo_id="HooshvareLab/gpt2-fa",  
        model_kwargs={"temperature": 0.5, "max_length": 1024}
    )
    return llm

def correct_rtl_text(text):
    return bidi.get_display(text)

def format_docs(docs):
    formatted_docs = []
    for i, doc in enumerate(docs):
        content = correct_rtl_text(doc.page_content)
        formatted_docs.append(f"Document {i+1}:\n{content}")
        
    return "\n\n".join(formatted_docs)

class RAGManager: 
    def __init__(self, db_dir, llm, embeddings):
        self.db_dir = db_dir
        self.llm = llm
        self.embeddings = embeddings
        
        try:
            self.vectordb = Chroma(
                persist_directory=db_dir,
                embedding_function=embeddings
            )

        except Exception as e:
            import shutil
            if os.path.exists(db_dir):
                shutil.rmtree(db_dir)
                os.makedirs(db_dir, exist_ok=True)
             
            self.vectordb = Chroma(
                persist_directory=db_dir,
                embedding_function=embeddings
            )
        

        self.retriever = self.vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        

        self.template = """
        <div dir="rtl">
        شما یک دستیار هوشمند فارسی‌زبان متخصص هستید.
        
        ### دستورالعمل‌ها:
        - با دقت متن‌های ارائه شده را بخوانید
        - فقط از اطلاعات موجود در متن‌ها برای پاسخ استفاده کنید
        - به قسمت‌های مرتبط با سؤال در متن‌ها استناد کنید
        - اگر اطلاعات کافی برای پاسخ در متن‌ها وجود ندارد، به صراحت بگویید
        - از پاسخ‌های خیالی یا نامربوط خودداری کنید
        - پاسخ را گام به گام توضیح دهید
        
        ### اطلاعات مرتبط:
        {context}
        
        ### سؤال کاربر: 
        {question}
        
        ### پاسخ دقیق (با استناد به متن‌های بالا):
        </div>
        """
        
        self.prompt = PromptTemplate(
            template=self.template,
            input_variables=["context", "question"]
        )
        

        self.rag_chain = (
            {
                "context": self.retriever | RunnableLambda(format_docs), 
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    def preprocess_text(self, text):
        return text
    
    def add_documents(self, documents):
        for doc in documents:
            doc.page_content = self.preprocess_text(doc.page_content)
        

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", "!", "؟", "،", " ", ""],  
            length_function=len,
        )
        
        chunks = text_splitter.split_documents(documents)
        self.vectordb.add_documents(chunks)
        self.vectordb.persist()
        return len(chunks)
    
    def get_response(self, query):
        try:
            response = self.rag_chain.invoke(query)
            return response
        except Exception as e:
            error_msg = str(e)
            if "dimension" in error_msg.lower():
                return "خطا: ابعاد امبدینگ با پایگاه داده مطابقت ندارد. لطفاً پایگاه داده را بازنشانی کنید."
            elif "api" in error_msg.lower() or "server" in error_msg.lower():
                return "خطا در ارتباط با سرور هاگینگ‌فیس. لطفاً اتصال اینترنت خود را بررسی کنید یا دوباره تلاش کنید."
            else:
                return f"خطا در پردازش پرسش شما: {error_msg}"
    
    def get_similar_documents(self, query, k=5):
        try:
            docs = self.vectordb.similarity_search(query, k=k)
            for doc in docs:
                doc.page_content = correct_rtl_text(doc.page_content)
            return docs
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return []


def examine_retrieved_docs(rag_manager, query):
    docs = rag_manager.get_similar_documents(query)
    
    print(f"Query: {query}")
    print("Retrieved Documents:")
    for i, doc in enumerate(docs):
        print(f"\nDocument {i+1}:")
        print("-" * 40)
        print(doc.page_content)
        print("-" * 40)

if __name__ == "__main__":
    data_dir = "docs"
    db_dir = "db"

    embeddings = setup_embeddings()
    llm = get_llm()
    rag_manager = RAGManager(db_dir=db_dir, llm=llm, embeddings=embeddings)

    documents = process_pdfs(data_dir, db_dir)
    if documents:
        num_chunks = rag_manager.add_documents(documents)
        print(f"{num_chunks} بخش به پایگاه برداری اضافه شد.")
    else:
        print("هیچ فایل PDF معتبری یافت نشد.")

    query = "هدف الگوریتم PageRank چیست؟"
    examine_retrieved_docs(rag_manager, query)
    print("\nپاسخ نهایی:")
    print(rag_manager.get_response(query))