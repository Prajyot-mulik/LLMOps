import os
from dotenv import load_dotenv
from multi_doc_chat.utils.config_loader import load_config
from multi_doc_chat.logger import GLOBAL_LOGGER as log
from multi_doc_chat.exceptions.custom_exception import DocumentPortalException
from langchain_openai import ChatOpenAI
from multi_doc_chat.utils.openrouter_embeddings import OpenRouterEmbeddingsClient


class ApiKeyManager:
    REQUIRED_KEYS = ["OPENROUTER_API_KEY"]

    def __init__(self):
        self.api_keys = {}
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
            log.error(f"Missing required API keys: {missing}")
            raise DocumentPortalException("Missing API keys", missing)

        masked_keys = {k: v[:6] + "..." for k, v in self.api_keys.items()}
        log.info(f"API keys loaded: {masked_keys}")

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
        log.info(f"YAML config loaded. Keys: {list(self.config.keys())}")

    def load_embeddings(self):
        """Return a LangChain Embeddings object that calls OpenRouter (Qwen)."""
        try:
            model_name = self.config["embedding_model"]["model_name"]
            api_key = self.api_key_mgr.get("OPENROUTER_API_KEY")
            log.info(f"Loading embedding model: {model_name}")
            return OpenRouterEmbeddingsClient(
                model=model_name,
                api_key=api_key,
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
                api_key=self.api_key_mgr.get("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            log.error(f"Error loading LLM: {e}")
            raise DocumentPortalException("Failed to load LLM", e)


if __name__ == "__main__":
    loader = ModelLoader()

    # Test Embedding
    embeddings = loader.load_embeddings()
    print(f"Embedding Model Loaded: {embeddings}")
    vec = embeddings.embed_query("Hello, how are you?")
    print(f"Embedding Result dims: {len(vec)}")

    # Test LLM
    llm = loader.load_llm()
    print(f"LLM Loaded: {llm}")
    result = llm.invoke("Hello, how are you?")
    print(f"LLM Result: {result.content}")
