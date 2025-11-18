import os
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.embeddings import GooglePalmEmbeddings
from langchain.vectorstores import FAISS
import requests

load_dotenv() #doc load 
api_key = os.getenv("GOOGLE_API_KEY")


loader = TextLoader(r"C:\Users\prajyot\Documents\LLMOps\data\ml.txt", encoding="utf-8")
doc = loader.load() #extract the data



# print(doc[0].page_content[:20000])


#create a text splitter and chunk size
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap = 20)
# print(text_splitter)
text_chunk=text_splitter.split_documents(doc)
# print(text_chunk)


embeddings = []
for chunk in text_chunk:
    resp = requests.post(
        "https://openrouter.ai/api/v1/embeddings",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": "qwen/qwen3-embedding-0.6b", "input": chunk.page_content}
    )
    embeddings.append(resp.json()["data"][0]["embedding"])

