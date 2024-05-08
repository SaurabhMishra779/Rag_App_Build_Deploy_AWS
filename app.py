import json
import sys
import streamlit as streamlit
import os
import boto3


from langchain_community.embeddings import BedrockEmbeddings

from langchain_community.llms import Bedrock
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS