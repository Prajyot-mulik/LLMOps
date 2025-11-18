import os
import requests
from typing import List, Sequence
from langchain.embeddings.base import Embeddings


class OpenRouterEmbeddingsClient(Embeddings):
    """LangChain-compatible embeddings that call OpenRouter /embeddings endpoint."""

    def __init__(self, model: str, api_key: str, base_url: str = "https://openrouter.ai/api/v1", timeout: int = 60):
        if not api_key:
            raise ValueError("OpenRouter API key is required")
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _embed(self, inputs: Sequence[str]) -> List[List[float]]:
        """Call OpenRouter embeddings API and return list of vectors."""
        resp = requests.post(
            f"{self.base_url}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "input": list(inputs)},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        if "data" not in data or not isinstance(data["data"], list):
            raise ValueError(f"No embedding data received: {data}")
        # Each item should contain an "embedding" list
        vectors = [item.get("embedding") for item in data["data"]]
        if any(v is None for v in vectors):
            raise ValueError(f"Missing embedding vectors in response items: {data['data']}")
        return vectors

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents."""
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        vectors = self._embed([text])
        return vectors[0]
