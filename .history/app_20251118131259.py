from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.retrievers import MultiQueryRetriever
import os

# ============================================================
# 1. SET API KEY
# ============================================================
os.environ["OPENROUTER_API_KEY"] = "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"

# ============================================================
# 2. EMBEDDING MODEL (openrouter)
# ============================================================
embeddings = OpenAIEmbeddings(
    openai_api_key=os.environ["OPENROUTER_API_KEY"],
    model="qwen/qwen3-embedding-0.6b",
    base_url="https://openrouter.ai/api/v1"
)

# ============================================================
# 3. TEST DATA
# ============================================================
documents = [
    Document(page_content="Python is a popular programming language used for AI and automation."),
    Document(page_content="AWS EC2 provides scalable virtual servers in the cloud."),
    Document(page_content="Docker is used to containerize applications."),
    Document(page_content="Qwen models are strong for embeddings and language tasks."),
]

# Build vector DB
vectorstore = FAISS.from_documents(documents, embeddings)

# ============================================================
# 4. RETRIEVER CONFIG
# ============================================================
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 10,
        "fetch_k": 20,
        "lambda_mult": 0.5
    }
)

# ============================================================
# 5. LLM MODEL (openrouter)
# ============================================================
llm = ChatOpenAI(
    openai_api_key=os.environ["OPENROUTER_API_KEY"],
    model="nvidia/nemotron-nano-12b-v2-vl:free",
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
    max_tokens=2048
)

# ============================================================
# 6. TEST QUERY
# ============================================================
query = "What is Docker used for?"

# Retrieve
results = retriever.get_relevant_documents(query)
print("\n===== Retrieved Docs =====")
for r in results:
    print("-", r.page_content)

# LLM Response
prompt = f"Answer the question based on context:\n{[r.page_content for r in results]}\n\nQuestion: {query}"

response = llm.invoke(prompt)

print("\n===== LLM Answer =====")
print(response.content)
