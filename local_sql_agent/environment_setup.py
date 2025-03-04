import os
langsmith_api_key = os.environ.get("LANGSMITH_API_KEY")

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Local SQL Agent"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"

from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.1:8b")  #"llama3.1"

from examples import *
from prompt import *
from agent import *