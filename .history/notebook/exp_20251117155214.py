import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from google.generative_ai import Client  # ✅ Key fix: Correct client library

# Load API key and project ID
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
google_cloud_project = os.getenv("GOOGLE_CLOUD_PROJECT")

# Initialize Gemini client
client = Client(api_key=api_key)

# 4. Custom Gemini Embedding Class
class GeminiEmbeddings:
    def __init__(self):
        self.client = client  # Reuse the global client
    
    def embed_documents(self, texts):
        return self._embed_texts(texts)
    
    def embed_query(self, text):
        return self._embed_texts([text])[0]
    
    def _embed_texts(self, texts):
        return [
            client.models.embed_content(
                model="gemini-embedding-001",  # ✅ Correct model name per Google docs
                contents=text
            ).embeddings for text in texts
        ]

# Load documents
loader = TextLoader(r"C:\Users\prajyot\Documents\LLMOps\data\ml.txt", encoding="utf-8")
doc = loader.load()

# Split text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
text_chunks = text_splitter.split_documents(doc)

# Create embeddings and vector store
gemini_embeddings = GeminiEmbeddings()
vectorstores = FAISS.from_documents(text_chunks, gemini_embeddings)