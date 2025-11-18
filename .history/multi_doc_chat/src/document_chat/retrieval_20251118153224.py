from multi_doc_chat.utils.model_loader import ModelLoader
from multi_doc_chat.exceptions.custom_exception import DocumentPortalException
from multi_doc_chat.promts.prompt_library import PROMPT_REGISTRY
from multi_doc_chat.model.models import PromptType, ChatAnswer
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from pydantic import ValidationError
import sys
import os
from operator import itemgetter
from typing import List, Optional, Dict, Any


class ConversationalRAG:
    def __init__(self, session_id: Optional[str], runtime_key: str, retriever=None):
        try:
            self.session_id = session_id
            self.runtime_key = runtime_key  # store runtime key

            # Load LLM (OpenRouter) and prompts
            self.llm = self._load_llm()
            self.contextualize_prompt: ChatPromptTemplate = PROMPT_REGISTRY[PromptType.CONTEXTUALIZE_QUESTION.value]
            self.qa_prompt: ChatPromptTemplate = PROMPT_REGISTRY[PromptType.CONTEXT_QA.value]

            # Lazy retriever
            self.retriever = retriever
            self.chain = None
            if self.retriever is not None:
                self._build_lcel_chain()

        except Exception as e:
            raise DocumentPortalException("Initialization error in ConversationalRAG", e) from e

    def _load_llm(self, runtime_key: str):

        try:
            
            llm = ModelLoader(runtime_key=runtime_key).load_llm()
            if not llm:
                raise ValueError("LLM could not be loaded")
            log.info(f"LLM loaded successfully. session_id={self.session_id}")
            return llm
        except Exception as e:
            log.error(f"Failed to load LLM: {e}")
            raise DocumentPortalException("LLM loading error in ConversationalRAG", e) from e

    def load_retriever_from_faiss(self, index_path: str, k: int = 5, search_type: str = "mmr", fetch_k: int = 20, lambda_mult: float = 0.5, search_kwargs: Optional[Dict[str, Any]] = None):
        embeddings = ModelLoader(runtime_key=self.runtime_key).load_embeddings()
        vectorstore = FAISS.load_local(
            index_path,
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )
        if search_kwargs is None:
            search_kwargs = {"k": k}
            if search_type == "mmr":
                search_kwargs["fetch_k"] = fetch_k
                search_kwargs["lambda_mult"] = lambda_mult

        self.retriever = vectorstore.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
        self._build_lcel_chain()
        return self.retriever

    def invoke(self, user_input: str, chat_history: Optional[List[BaseMessage]] = None) -> str:
        if self.chain is None:
            raise DocumentPortalException("RAG chain not initialized. Call load_retriever_from_faiss() before invoke().", sys)
        chat_history = chat_history or []
        payload = {"input": user_input, "chat_history": chat_history}
        answer = self.chain.invoke(payload)
        try:
            validated = ChatAnswer(answer=str(answer))
            answer = validated.answer
        except ValidationError as ve:
            raise DocumentPortalException("Invalid chat answer", ve) from ve
        return answer

    @staticmethod
    def _format_docs(docs) -> str:
        return "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)

    def _build_lcel_chain(self):
        if self.retriever is None:
            raise DocumentPortalException("No retriever set before building chain", sys)
        question_rewriter = (
            {"input": itemgetter("input"), "chat_history": itemgetter("chat_history")}
            | self.contextualize_prompt
            | self.llm
            | StrOutputParser()
        )
        retrieve_docs = question_rewriter | self.retriever | self._format_docs
        self.chain = (
            {
                "context": retrieve_docs,
                "input": itemgetter("input"),
                "chat_history": itemgetter("chat_history"),
            }
            | self.qa_prompt
            | self.llm
            | StrOutputParser()
        )
