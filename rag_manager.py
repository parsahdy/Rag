import os
import re
import traceback

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.llms import Ollama

import torch
from dotenv import load_dotenv

from pdf_processor import process_pdfs_with_pdfminer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

load_dotenv()

def setup_embeddings():
    model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    model_kwargs = {"device": DEVICE}
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs={"normalize_embeddings": True}
    )
    
    return embeddings

def get_ollama_llm():
    """Use Ollama with Llama 3.1 model"""
    print("Setting up Ollama with Llama 3.1...")
    
    try:
        llm = Ollama(
            model="llama3.1",  
            temperature=0.2,   
            top_p=0.9,
            top_k=40,
            repeat_penalty=1.2,  
            num_ctx=4096,      
            num_predict=512    
        )
        
        print("Ollama with Llama 3.1 loaded successfully!")
        return llm
    except Exception as e:
        print(f"Error loading Ollama: {e}")
        print(traceback.format_exc())
        raise e

class RAGManager: 
    def __init__(self, db_dir, embeddings, use_local_model=True):
        self.db_dir = db_dir
        self.embeddings = embeddings
        
        try:
            self.llm = get_ollama_llm()
        except Exception as e:
            print(f"Error initializing Ollama: {e}")
            raise e
        
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
        <s>[INST]
        شما یک دستیار آموزشی هوشمند هستید که به سوالات دانش‌آموزان پاسخ می‌دهد.
        
        با توجه به اطلاعات زیر، به سوال دانش‌آموز پاسخ دهید:
        
        اطلاعات مرتبط:
        {context}
        
        سوال دانش‌آموز:
        {question}
        [/INST]
        """
        
        self.prompt = PromptTemplate(
            template=self.template,
            input_variables=["context", "question"]
        )
        
        self.rag_chain = (
            {
                "context": self.retriever | RunnableLambda(lambda docs: self.format_docs(docs)),   
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    def format_docs(self, docs):
        """Format retrieved documents in a way suitable for the LLM"""
        if not docs:
            return "هیچ سند مرتبطی یافت نشد."
            
        formatted_docs = []
        
        for i, doc in enumerate(docs):
            try:
                content = doc.page_content.strip()
                formatted_docs.append(f"متن {i+1}:\n{content}")
                
            except Exception as e:
                print(f"Error formatting document {i}: {e}")
        
        if not formatted_docs:
            return "دسترسی به محتوای اسناد با مشکل مواجه شد."
            
        return "\n\n".join(formatted_docs)

    def get_response(self, query):
        """Get a response using the RAG chain with Ollama"""
        try:
            docs = self.vectordb.similarity_search(query, k=3)
            
            if not docs:
                return "متأسفانه اطلاعات مرتبطی با سوال شما در پایگاه داده یافت نشد."
            
            context = self.format_docs(docs)
            
            prompt = f"""<s>[INST]
            شما یک دستیار آموزشی فارسی زبان هستید که به سوالات دانش‌آموزان پاسخ می‌دهد.
            
            اطلاعات مرتبط:
            {context}
            
            سوال دانش‌آموز:
            {query}
            
            پاسخ دهید:
            [/INST]
            """
            
            response = self.llm.invoke(prompt)
            
            response = response.strip()
            
            return response
            
        except Exception as e:
            error_msg = str(e)
            print(f"Error in get_response: {error_msg}")
            print(traceback.format_exc())
            return f"خطا در پردازش پرسش شما: {str(e)}"
            
    def get_similar_documents(self, query, k=5):
        """Get similar documents for a query"""
        try:
            docs = self.vectordb.similarity_search(query, k=k)
            return docs
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return []
        
    def debug_vector_retrieval(self, query):
        """Debug function to check vector retrieval"""
        print("\n--- DEBUGGING VECTOR RETRIEVAL ---")
        print(f"Query: {query}")
        
        try:
            docs = self.vectordb.similarity_search(query, k=3)
            print(f"Retrieved {len(docs)} documents")
            
            for i, doc in enumerate(docs):
                print(f"\nDocument {i+1}:")
                print("-" * 40)
                print(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)
                print("-" * 40)
                
            return docs
        except Exception as e:
            print(f"Error in vector retrieval: {e}")
            traceback_str = traceback.format_exc()
            print(traceback_str)
            return []