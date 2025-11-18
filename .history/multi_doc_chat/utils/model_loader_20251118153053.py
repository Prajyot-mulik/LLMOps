import os
from typing import Optional
from multi_doc_chat.utils.config_loader import load_config
from multi_doc_chat.logger import GLOBAL_LOGGER as log
from multi_doc_chat.exceptions.custom_exception import DocumentPortalException
from langchain_openai import ChatOpenAI
from multi_doc_chat.utils.openrouter_embeddings import OpenRouterEmbeddingsClient


class ModelLoader:
    """Loads embedding models and LLMs from OpenRouter using runtime key."""

    def __init__(self, runtime_key: Optional[str] = None):
        if not runtime_key:
            log.error("OpenRouter API key must be provided at runtime")
            raise DocumentPortalException("OpenRouter API key must be provided at runtime")
        self.api_key = runtime_key
        self.config = load_config()
        log.info(f"YAML config loaded. Keys: {list(self.config.keys())}")

    def load_embeddings(self):
        """Return a LangChain Embeddings object that calls OpenRouter."""
        try:
            model_name = self.config["embedding_model"]["model_name"]
            log.info(f"Loading embedding model: {model_name}")
            return OpenRouterEmbeddingsClient(
                model=model_name,
                api_key=self.api_key,
                base_url="https://openrouter.ai/api/v1",
            )
        except Exception as e:
            log.error(f"Error loading embedding model: {e}")
            raise DocumentPortalException("Failed to load embedding model", e)

    def load_llm(self):
        """Load and return the configured LLM model from OpenRouter."""
        try:
            llm_block = self.config["llm"]
            provider_key = os.getenv("LLM_PROVIDER", "openrouter")
            if provider_key not in llm_block:
                log.error(f"LLM provider not found in config: {provider_key}")
                raise ValueError(f"LLM provider '{provider_key}' not found in config")

            llm_config = llm_block[provider_key]
            model_name = llm_config.get("model_name")
            temperature = llm_config.get("temperature", 0)
            max_tokens = llm_config.get("max_output_tokens", 2048)

            log.info(f"Loading LLM (provider=openrouter, model={model_name})")
            return ChatOpenAI(
                model=model_name,
                api_key=self.api_key,
                base_url="https://openrouter.ai/api/v1",
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            log.error(f"Error loading LLM: {e}")
            raise DocumentPortalException("Failed to load LLM", e)
