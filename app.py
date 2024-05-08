import json
import sys
import streamlit as st
import os
import boto3


from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from QASystem.ingestion import get_vector_store
from QASystem.ingestion import data_ingestion, bedrock_embeddings

from QASystem.retrieval_generation import get_llama3_llm
from QASystem.retrieval_generation import get_response_llm


def main():
    st.set_page_config("QA with Doc")
    st.header("QA with doc using lanchain and AWSBedrock service")

    user_question = st.text_input("Ask a question from pdf files")
    with st.sidebar:
        st.title("update pr create the vector store")
        if st.button("vectos update"):
            with st.spinner("Processing ....."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("done")




        if st.button("llama3 model "):
            with st.spinner("processing ...."):
                faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings,allow_dangerous_deserialization=True)
                llm = get_llama3_llm()

                st.write(get_response_llm(llm, faiss_index,user_question ))
                st.success("Done")

if __name__=="__main__":
    main()
