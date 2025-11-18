import os
from dotenv import load_dotenv
from multi_doc_chat.utils.config_loader import load_config
from multi_doc_chat.logger import GLOBAL_LOGGER as log
from multi_doc_chat.exceptions.custom_exception import DocumentPortalException
from langchain_openai import ChatOpenAI
from multi_doc_chat.utils.openrouter_embeddings import OpenRouterEmbeddingsClient

class ApiKeyManager:
    REQUIRED_KEYS = ["OPENROUTER_API_KEY"]

    def __init__(self, runtime_key: str = None):
        self.api_keys = {}
        if runtime_key:
            self.api_keys["OPENROUTER_API_KEY"] = runtime_key
        else:
            if os.getenv("ENV", "local").lower() != "production":
                load_dotenv()
            for key in self.REQUIRED_KEYS:
                val = os.getenv(key)
                if val: self.api_keys[key] = val

        missing = [k for k in self.REQUIRED_KEYS if not self.api_keys.get(k)]
        if missing:
            raise DocumentPortalException("Missing API keys", missing)

    def get(self, key: str) -> str:
        val = self.api_keys.get(key)
        if not val: raise KeyError(f"API key for {key} is missing")
        return val

class ModelLoader:
    """Loads embedding models and LLMs from OpenRouter."""

    def __init__(self, api_key: str = None):
        self.api_key_mgr = ApiKeyManager(runtime_key=api_key)
        self.config = load_config()
        log.info(f"YAML config loaded. Keys: {list(self.config.keys())}")

    def load_embeddings(self):
        model_name = self.config["embedding_model"]["model_name"]
        api_key = self.api_key_mgr.get("OPENROUTER_API_KEY")
        return OpenRouterEmbeddingsClient(model=model_name, api_key=api_key, base_url="https://openrouter.ai/api/v1")

    def load_llm(self):
        llm_block = self.config["llm"]
        provider_key = os.getenv("LLM_PROVIDER","openrouter")
        if provider_key not in llm_block: raise ValueError(f"LLM provider '{provider_key}' not found")
        cfg = llm_block[provider_key]
        return ChatOpenAI(
            model=cfg.get("model_name"),
            api_key=self.api_key_mgr.get("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            temperature=cfg.get("temperature",0),
            max_tokens=cfg.get("max_output_tokens",2048)
        )
