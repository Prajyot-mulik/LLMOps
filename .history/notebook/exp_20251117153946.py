import os
from dotenv import load_dotenv
from langchain_google_vertexai import VertexAIEmbeddings  # ✅ Use VertexAI for Gemini
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
load_dotenv() #doc load 
api_key = os.getenv("GEMINI_API_KEY")


loader = TextLoader(r"C:\Users\prajyot\Documents\LLMOps\data\ml.txt", encoding="utf-8")
doc = loader.load() #extract the data




# print(doc[0].page_content[:20000])


#create a text splitter and chunk size
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap = 20)
# print(text_splitter)

text_chunk=text_splitter.split_documents(doc)
# print(text_chunk)

embeddings = embeddings = VertexAIEmbeddings(
    model="text-embedding-004",  # Gemini's embedding model
    task="embedding"
)

vectorstores=FAISS.from_documents(text_chunk , embeddings)


