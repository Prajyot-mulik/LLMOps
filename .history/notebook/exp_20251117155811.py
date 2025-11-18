import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer  # ✅ Free embedding model

# Load environment variables (if needed)
load_dotenv()

# 1. Initialize free embedding model
class LocalEmbeddings:
    def __init__(self):
        # ✅ Free Hugging Face model
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_numpy=True).tolist()
    
    def embed_query(self, text):
        return self.model.encode([text], convert_to_numpy=True)[0].tolist()

# 2. Load documents
loader = TextLoader(r"C:\Users\prajyot\Documents\LLMOps\data\ml.txt", encoding="utf-8")
doc = loader.load()

# 3. Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
text_chunks = text_splitter.split_documents(doc)

# 4. Create embeddings and FAISS vector store
local_embeddings = LocalEmbeddings()
vectorstore = FAISS.from_documents(text_chunks, local_embeddings)

print("✅ Vector store created with free embeddings!")
