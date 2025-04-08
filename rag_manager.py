import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
import torch


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def setup_embeddings():
    model_name = "distiluse-base-multilingual-cased-v1"
    
    model_kwargs = {"device": DEVICE}
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs
    )
    
    return embeddings

def get_llm():
    llm = HuggingFaceEndpoint(
        repo_id="HooshvareLab/bert-base-parsbert-uncased", 
        max_length=2048,
        temperature=0.5, 
    )
    return llm

class RAGManager:
    
    def __init__(self, db_dir, llm, embeddings):
        self.db_dir = db_dir
        self.llm = llm
        self.embeddings = embeddings
        
        #
        self.vectordb = Chroma(
            persist_directory=db_dir,
            embedding_function=embeddings
        )
        
        
        self.retriever = self.vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}  
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
            return self.rag_chain.invoke(query)
        except Exception as e:
            return f"خطا در پردازش پرسش شما: {str(e)}"
    
    def get_similar_documents(self, query, k=3):
        return self.vectordb.similarity_search(query, k=k)