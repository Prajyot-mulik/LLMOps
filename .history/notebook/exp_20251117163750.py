import os
import requests
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# 1. Load API key
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

# 2. Load document
loader = TextLoader(r"C:\Users\prajyot\Documents\LLMOps\data\ml.txt", encoding="utf-8")
doc = loader.load()

# 3. Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
text_chunks = text_splitter.split_documents(doc)

# 4. Collect all chunk texts
texts = [chunk.page_content for chunk in text_chunks]

# 5. Make ONE request for all embeddings
resp = requests.post(
    "https://openrouter.ai/api/v1/embeddings",
    headers={"Authorization": f"Bearer {api_key}"},
    json={"model": "openai/text-embedding-3-small", "input": texts}  # batch input
)

result = resp.json()
print("Embedding response:", result)  # 👀 Debug

embeddings = [item["embedding"] for item in result.get("data", [])]

# 6. Build FAISS vectorstore
text_embeddings = list(zip(texts, embeddings))
vectorstore = FAISS.from_embeddings(text_embeddings, embedding=None)

# 7. Embed query manually
query = "Explain supervised learning"
resp = requests.post(
    "https://openrouter.ai/api/v1/embeddings",
    headers={"Authorization": f"Bearer {api_key}"},
    json={"model": "openai/text-embedding-3-small", "input": query}
)
query_embedding = resp.json()["data"][0]["embedding"]

# 8. Search by vector
results = vectorstore.similarity_search_by_vector(query_embedding, k=5)
for r in results:
    print(r.page_content)
