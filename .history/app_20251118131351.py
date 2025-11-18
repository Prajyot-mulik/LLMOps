import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

# --------------------------
# 1. API KEY
# --------------------------
os.environ["OPENROUTER_API_KEY"] = "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"

# --------------------------
# 2. EMBEDDINGS
# --------------------------
embeddings = OpenAIEmbeddings(
    model="qwen/qwen3-embedding-0.6b",
    openai_api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1"
)

# --------------------------
# 3. SAMPLE DOCUMENTS
# --------------------------
docs = [
    Document(page_content="Docker helps to containerize applications."),
    Document(page_content="EC2 is a cloud compute service from AWS."),
    Document(page_content="Python is widely used for AI and machine learning."),
]

# Build vector DB
vectordb = FAISS.from_documents(docs, embeddings)

# Simple retriever (MMR optional)
retriever = vectordb.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.5}
)

# --------------------------
# 4. LLM (OpenRouter)
# --------------------------
llm = ChatOpenAI(
    model="nvidia/nemotron-nano-12b-v2-vl:free",
    temperature=0,
    max_tokens=2048,
    openai_api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1"
)

# --------------------------
# 5. TEST QUERY
# --------------------------
query = "What is Docker used for?"
results = retriever.get_relevant_documents(query)

print("\n=== Retrieved Docs ===")
for d in results:
    print("-", d.page_content)

# Build prompt
context = "\n".join([d.page_content for d in results])
prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer briefly."

response = llm.invoke(prompt)

print("\n=== LLM Answer ===")
print(response.content)
