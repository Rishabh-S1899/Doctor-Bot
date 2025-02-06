import os
import sys
from langchain import HuggingFaceHub
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import CTransformers
import pickle
import faiss
from langchain.vectorstores import FAISS
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
import warnings
warnings.filterwarnings("ignore")
import pickle
# from google.colab import drive
# drive.mount('/content/drive')
from langchain_community.document_loaders import JSONLoader
import json
from pathlib import Path
from pprint import pprint
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma



def create_embeddings_with_chroma(split_docs,embedding_model_name,model_kwargs,db):
    embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs=model_kwargs
    )
    vector_store = Chroma.from_documents(split_docs, embeddings,persist_directory=db)
    return vector_store


os.environ["HUGGINGFACEHUB_API_TOKEN"] = "Your API key"

embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "gpu"}

llm = HuggingFaceHub(
    repo_id="google/flan-t5-large",  # Change to "flan-t5-large", "flan-t5-xl", or "flan-t5-xxl" as needed
    model_kwargs={"temperature": 0.55, "max_length": 512}
)

