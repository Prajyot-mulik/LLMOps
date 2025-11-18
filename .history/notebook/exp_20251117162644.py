import os
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter


from langchain_community.vectorstores import FAISS

import requests

load_dotenv() #doc load 
api_key = os.getenv("OPENROUTER_API_KEY")


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
    result = resp.json()
    if "data" in result:
        embeddings.append(result["data"][0]["embedding"])
    else:
        print("Error response:", result)

text_embeddings = [(chunk.page_content, emb) for chunk, emb in zip(text_chunk, embeddings)]
vectorstore = FAISS.from_embeddings(text_embeddings, embedding=None)

query = "Explain supervised learning"
results = vectorstore.similarity_search(query, k=5)
for r in results:
    print(r.page_content)