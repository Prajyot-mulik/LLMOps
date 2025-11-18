To integrate **Gemini embeddings directly** into your LangChain pipeline (without `VertexAIEmbeddings`), follow these steps. This approach avoids deprecated libraries and uses Google's current API:

---

### 1. **Install Required Packages**
```bash
# In your UV environment
uv pip install google-generative-ai langchain langchain-community python-dotenv
```

---

### 2. **Correct Environment Setup**
Create a `.env` file:
```env
GEMINI_API_KEY=your-vertex-ai-api-key
GOOGLE_CLOUD_PROJECT=your-google-project-id
```

---

### 3. **Updated Code (`exp.py`)**
```python
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from google.generative_ai import Client  # ✅ Key fix: Correct client library

# Load API key and project ID
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
google_cloud_project = os.getenv("GOOGLE_CLOUD_PROJECT")

# Initialize Gemini client
client = Client(api_key=api_key)

# 4. Custom Gemini Embedding Class
class GeminiEmbeddings:
    def __init__(self):
        self.client = client  # Reuse the global client
    
    def embed_documents(self, texts):
        return self._embed_texts(texts)
    
    def embed_query(self, text):
        return self._embed_texts([text])[0]
    
    def _embed_texts(self, texts):
        return [
            client.models.embed_content(
                model="gemini-embedding-001",  # ✅ Correct model name per Google docs
                contents=text
            ).embeddings for text in texts
        ]

# Load documents
loader = TextLoader(r"C:\Users\prajyot\Documents\LLMOps\data\ml.txt", encoding="utf-8")
doc = loader.load()

# Split text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
text_chunks = text_splitter.split_documents(doc)

# Create embeddings and vector store
gemini_embeddings = GeminiEmbeddings()
vectorstores = FAISS.from_documents(text_chunks, gemini_embeddings)
```

---

### 4. **Why This Works**
- **Removed `VertexAIEmbeddings`**: Directly uses Google's `genai` client.
- **No Conflicts**: Uses `google-generative-ai` (not the deprecated package).
- **Flexible Embeddings**: Works with any Gemini embedding model (`gemini-embedding-001`, `gemini-pro`), but ensure the model is available in your region.

---

### 5. **Run the Script**
In your UV environment:
```bash
uv run python exp.py
```

---

### 🔒 API Requirements
Ensure your Google Cloud project has:
- **Gemini API** enabled.
- The Vertex AI service account has `roles/aiplatform.user` permissions.
- The model (`text-embedding-004` or `gemini-embedding-001`) is available.

---

### 🚫 Common Pitfalls
- **Wrong Model Name**: Use `gemini-embedding-001` (not `gemini-pro`).
- **Installation Errors**: Avoid mixing `pip install --no-index` or site-packages conflicts.
- **API Key Scope**: Ensure the key has `Vertex AI User` service permissions.

Let me know if you encounter specific errors! 🛠️