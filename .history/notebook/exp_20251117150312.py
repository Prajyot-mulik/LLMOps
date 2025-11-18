import os
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

loader = ("data\\ml.txt", encoding="utf8")

