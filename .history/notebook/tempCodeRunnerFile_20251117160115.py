import os
import requests

# Make sure you set your OpenRouter API key in environment
api_key = os.getenv("OPENROUTER_API_KEY")

headers = {
    "Authorization": f"Bearer {api_key}",
    "HTTP-Referer": "http://localhost",   # optional
    "X-Title": "Embedding Test",          # optional
}

# Input text to embed
data = {
    "model": "qwen/qwen3-embedding-0.6b",
    "input": "Machine learning is a subset of artificial intelligence."
}

# Call OpenRouter embeddings endpoint
resp = requests.post("https://openrouter.ai/api/v1/embeddings", headers=headers, json=data)

# Print embedding vector length and first few values
result = resp.json()
embedding = result["data"][0]["embedding"]

print("Embedding length:", len(embedding))
print("First 10 values:", embedding[:10])
