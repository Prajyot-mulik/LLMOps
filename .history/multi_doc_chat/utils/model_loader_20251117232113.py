import os
import sys
from dotenv import load_dotenv
from multi_doc_chat.utils.config_loader import load_config
from multi_doc_chat.logger import GLOBAL_LOGGER as log
from multi_doc_chat.exceptions.custom_exception import DocumentPortalException
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


class ApiKeyManager:
    REQUIRED_KEYS = ["OPENROUTER_API_KEY"]

    def __init__(self):
        self.api_keys = {}
        # Load from .env in local mode
        if os.getenv("ENV", "local").lower() != "production":
            load_dotenv()
            log.info("Running in LOCAL mode: .env loaded")
        else:
            log.info("Running in PRODUCTION mode")

        for key in self.REQUIRED_KEYS:
            env_val = os.getenv(key)
            if env_val:
                self.api_keys[key] = env_val
                log.info(f"Loaded {key} from env var")

        missing = [k for k in self.REQUIRED_KEYS if not self.api_keys.get(k)]
        if missing:
            log.error("Missing required API keys", missing_keys=missing)
            raise DocumentPortalException("Missing API keys", sys)

        log.info("API keys loaded", keys={k: v[:6] + "..." for k, v in self.api_keys.items()})

    def get(self, key: str) -> str:
        val = self.api_keys.get(key)
        if not val:
            raise KeyError(f"API key for {key} is missing")
        return val


class ModelLoader:
    """Loads embedding models and LLMs from OpenRouter."""

    def __init__(self):
        self.api_key_mgr = ApiKeyManager()
        self.config = load_config()
        log.info("YAML config loaded", config_keys=list(self.config.keys()))

    def load_embeddings(self):
        """Load and return embedding model from OpenRouter."""
        try:
            model_name = self.config["embedding_model"]["model_name"]
            log.info("Loading embedding model", model=model_name)
            return OpenAIEmbeddings(
                model=model_name,
                api_key=self.api_key_mgr.get("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1"
            )
        except Exception as e:
            log.error("Error loading embedding model", error=str(e))
            raise DocumentPortalException("Failed to load embedding model", sys)

    def load_llm(self):
        """Load and return the configured LLM model from OpenRouter."""
        try:
            llm_block = self.config["llm"]
            provider_key = os.getenv("LLM_PROVIDER", "openrouter")
            if provider_key not in llm_block:
                log.error("LLM provider not found in config", provider=provider_key)
                raise ValueError(f"LLM provider '{provider_key}' not found in config")

            llm_config = llm_block[provider_key]
            model_name = llm_config.get("model_name")
            temperature = llm_config.get("temperature", 0)
            max_tokens = llm_config.get("max_output_tokens", 2048)

            log.info("Loading LLM", provider="openrouter", model=model_name)

            return ChatOpenAI(
                model=model_name,
                api_key=self.api_key_mgr.get("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
                temperature=temperature,
                max_tokens=max_tokens
            )
        except Exception as e:
            log.error("Error loading LLM", error=str(e))
            raise DocumentPortalException("Failed to load LLM", sys)


if __name__ == "__main__":
    loader = ModelLoader()

    # Test Embedding
    embeddings = loader.load_embeddings()
    print(f"Embedding Model Loaded: {embeddings}")
    result = embeddings.embed_query("Hello, how are you?")
    print(f"Embedding Result: {result[:5]}...")  # print first 5 dims

    # Test LLM
    llm = loader.load_llm()
    print(f"LLM Loaded: {llm}")
    result = llm.invoke("Hello, how are you?")
    print(f"LLM Result: {result.content}")
