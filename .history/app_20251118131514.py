import os
import requests
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

api_key = "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"

# ------------------------------------------------------
# 1. SAMPLE TEXTS
# ------------------------------------------------------
texts = [
    "Docker is used to containerize applications.",
    "AWS EC2 provides virtual machines in the cloud.",
    "Python is widely used for AI and data science."
]

# ------------------------------------------------------
# 2. GET EMBEDDINGS (batch)
# ------------------------------------------------------
resp = requests.post(
    "https://openrouter.ai/api/v1/embeddings",
    headers={"Authorization": f"Bearer {api_key}"},
    json={
        "model": "qwen/qwen3-embedding-0.6b",
        "input": texts
    }
)

result = resp.json()
embeddings = [item["embedding"] for item in result.get("data", [])]

# ------------------------------------------------------
# 3. Build FAISS Vectorstore
# ------------------------------------------------------
text_embeddings = list(zip(texts, embeddings))
vectorstore = FAISS.from_embeddings(text_embeddings, embedding=None)

# ------------------------------------------------------
# 4. Query
# ------------------------------------------------------
query = "What is Docker used for?"

# Embed query manually
qresp = requests.post(
    "https://openrouter.ai/api/v1/embeddings",
    headers={"Authorization": f"Bearer {api_key}"},
    json={
        "model": "qwen/qwen3-embedding-0.6b",
        "input": query
    }
)

qvec = qresp.json()["data"][0]["embedding"]

# Retrieve similar docs
docs_and_scores = vectorstore.similarity_search_with_score_by_vector(qvec, k=3)

print("\n=== Retrieved Docs ===")
context = ""
for doc, score in docs_and_scores:
    print(f"- {doc} (score={score})")
    context += doc + "\n"

# ------------------------------------------------------
# 5. LLM Call using OpenRouter
# ------------------------------------------------------
prompt = f"Context:\n{context}\nQuestion: {query}\nAnswer:"

llm_resp = requests.post(
    "https://openrouter.ai/api/v1/chat/completions",
    headers={"Authorization": f"Bearer {api_key}"},
    json={
        "model": "nvidia/nemotron-nano-12b-v2-vl:free",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0
    }
)

answer = llm_resp.json()["choices"][0]["message"]["content"]

print("\n=== LLM Answer ===")
print(answer)
