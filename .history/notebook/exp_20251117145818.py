import os
from dotenv import load_dotenv
from langchain.document_loaders import text

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

