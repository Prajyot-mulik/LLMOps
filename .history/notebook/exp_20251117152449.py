import os
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.embeddings import 
from langchain_community.vectorstores import FAISS

load_dotenv() #doc load 
api_key = os.getenv("GOOGLE_API_KEY")


loader = TextLoader(r"C:\Users\prajyot\Documents\LLMOps\data\ml.txt", encoding="utf-8")
doc = loader.load() #extract the data




# print(doc[0].page_content[:20000])


#create a text splitter and chunk size
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap = 20)
# print(text_splitter)

text_chunk=text_splitter.split_documents(doc)
# print(text_chunk)

embeddings = GooglePalmEmbeddings(
    google_api_key=api_key
)


