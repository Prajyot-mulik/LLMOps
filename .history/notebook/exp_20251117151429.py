import os
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

loader = TextLoader(r"C:\Users\prajyot\Documents\LLMOps\data\ml.txt", encoding="utf-8")
doc = loader.load()


# print(doc[0].page_content[:20000])

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap = 20)
# print(text_splitter)
