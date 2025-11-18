from pathlib import Path
from typing import List, Iterable
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from multi_doc_chat.logger import GLOBAL_LOGGER as log
from multi_doc_chat.exceptions.custom_exception import DocumentPortalException

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def load_documents(paths: Iterable[Path]) -> List[Document]:
    """Load docs using appropriate loader based on extension."""
    docs: List[Document] = []
    try:
        for p in paths:
            ext = p.suffix.lower()
            if ext == ".pdf":
                loader = PyPDFLoader(str(p))
            elif ext == ".docx":
                loader = Docx2txtLoader(str(p))
            elif ext == ".txt":
                loader = TextLoader(str(p), encoding="utf-8")
            else:
                log.warning(f"Unsupported extension skipped. path={str(p)}")
                continue
            docs.extend(loader.load())
        log.info(f"Documents loaded. count={len(docs)}")
        return docs
    except Exception as e:
        log.error(f"Failed loading documents: {e}")
        raise DocumentPortalException("Error loading documents", e) from e
