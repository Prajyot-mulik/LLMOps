import os
import requests
from langchain_community.vectorstores import FAISS

api_key = "YOUR_OPENROUTER_API_KEY"

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
    json={"model": "qwen/qwen3-embedding-0.6b", "input": texts}
)

result = resp.json()
print("Embedding API Response:", result)  # DEBUG

# If API request failed
if "data" not in result or len(result["data"]) == 0:
    raise ValueError("❌ OpenRouter returned no embeddings. Check API key / model name.")

embeddings = [item.get("embedding") for item in result.get("data", [])]

# Remove any None
text_embeddings = [(txt, emb) for txt, emb in zip(texts, embeddings) if emb]

if len(text_embeddings) == 0:
    raise ValueError("❌ No embeddings available to insert into FAISS.")

# ------------------------------------------------------
# 3. Build FAISS Vectorstore
# ------------------------------------------------------
vectorstore = FAISS.from_embeddings(text_embeddings, embedding=None)

# ------------------------------------------------------
# 4. Query
# ------------------------------------------------------
query = "What is Docker used for?"

# Embed query manually
qresp = requests.post(
    "https://openrouter.ai/api/v1/embeddings",
    headers={"Authorization": f"Bearer {api_key}"},
    json={"model": "qwen/qwen3-embedding-0.6b", "input": query}
)

qjson = qresp.json()
print("Query Embedding Response:", qjson)  # DEBUG

if "data" not in qjson:
    raise ValueError("❌ Query embedding failed.")

qvec = qjson["data"][0]["embedding"]

# Retrieve docs
docs_and_scores = vectorstore.similarity_search_with_score_by_vector(qvec, k=3)

print("\n=== Retrieved Docs ===")
context = ""
for (doc, score) in docs_and_scores:
    print(f"- {doc}  (score={score})")
    context += doc + "\n"

# ------------------------------------------------------
# 5. LLM (OpenRouter)
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
