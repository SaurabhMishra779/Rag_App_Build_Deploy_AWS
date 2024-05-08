import json
import sys
import streamlit as streamlit
import os
import boto3


from langchain_community.embeddings import BedrockEmbedding 

from langchain.llms.Bedrock import Bedrock
from langchain.prompt import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS