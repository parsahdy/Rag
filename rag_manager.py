import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
import torch
from transformers import pipeline
from langchain_community.llms import HuggingFaceHub

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

load_dotenv()
token = os.getenv("HUGGINGFACE_API_TOKEN")

os.environ["HUGGINGFACEHUB_API_TOKEN"] = token

def setup_embeddings():
    model_name = "sentence-transformers/distiluse-base-multilingual-cased-v1"
    
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
        model_kwargs={"temperature": 0.7, "max_length": 512}
    )
    return llm

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
            search_kwargs={"k": 3}  
        )
        

        self.template = """
        شما یک دستیار هوشمند فارسی‌زبان متخصص هستید. 
        بر اساس اطلاعات زیر، به سؤال کاربر پاسخ دهید.
        اگر نمی‌توانید پاسخ دقیقی از اطلاعات پیدا کنید، بگویید "متأسفم، نمی‌توانم پاسخ دقیقی به این سؤال بدهم."
        از پاسخ‌های خیالی یا نامربوط خودداری کنید.

        اطلاعات مرتبط:
        -----------------
        {context}
        -----------------

        سؤال کاربر: {question}

        پاسخ دقیق:
        """
        
        self.prompt = PromptTemplate(
            template=self.template,
            input_variables=["context", "question"]
        )
        

        self.rag_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    def get_response(self, query):
        try:
            import asyncio
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
    
    def get_similar_documents(self, query, k=3):
        try:
            return self.vectordb.similarity_search(query, k=k)
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return []