import logging

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)

# Expose a global logger instance
GLOBAL_LOGGER = logging.getLogger("multi_doc_chat")
