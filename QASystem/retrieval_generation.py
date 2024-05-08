from langchain.chains import RetrievalQA
from langchain.vectorstores import faiss , FAISS
from langchain.llms.bedrock import Bedrock
import boto3
from langchain_community.embeddings import BedrockEmbeddings

from langchain.prompts import PromptTemplate
from QASystem.ingestion import get_vector_store
from QASystem.ingestion import data_ingestion


bedrock=boto3.client(service_name="bedrock-runtime")
bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-image-v1",client=bedrock)

prompt_template = """

Human : Use the following pieces of context to provide a concise answers to the question at the end but use
atleast summarize with 250 words with detailed explanation.If you don't know the answers just say don't know ,
don't try to make up the answer.
{context}
</context

Question : {question}

Assistant:"""

PROMPT = PromptTemplate(template = prompt_template,input_variables = ["context","question"])

def get_llama3_llm():
    llm = Bedrock(model_id ="meta.llama3-70b-instruct-v1:0", client=bedrock)
    return llm


def get_response_llm(llm ,vectorstore_faiss,query):
    qa = RetrievalQA.from_chain_type(
    llm = llm,
    chain_type = "stuff",
    retriever = vectorstore_faiss.as_retriever(
        search_type = "similarity",
        search_kwargs = {"k":3} 
    ),
    return_source_documents = True,
    chain_type_kwargs = {'prompt': PROMPT}
    )
    answer = qa({'query':query})
    return answer['result']


if __name__=='__main__':

    faiss_index = FAISS.load_local("faiss_index",bedrock_embeddings,allow_dangerous_deserialization=True)
    query = "what is RAG token"
    llm = get_llama3_llm()
    # docs = data_ingestion()
    # vector_store_faiss = get_vector_store(docs)
    print(get_response_llm(llm,faiss_index,query))  