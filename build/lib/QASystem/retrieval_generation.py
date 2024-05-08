from langchain.chains import retrieval_qa
from langchain.vectorstores import faiss
from langchain.llms.bedrock import Bedrock
import boto3
from langchain.prompts import PromptTemplate
from QASystem.ingestion import get_vector_store

bedrock = boto3.client(service_name = 'bedrock-runtime')
def get_llama3_llm(service_name= 'bedrock-runtime'):
    llm = Bedrock(model_id = "meta.llama3-70b-instruct-v1:0" , client=bedrock,model_kwargs = {'max_tokens':512})
    return llm

prompt_template = """

Human : Use the following pieces of context to provide a concise answers to the question at the end but use
atleast summarize with 250 words with detailed explanation.If you don't know the answers just say don't know ,
don't try to make up the answer.
{context}
</context

Question : {question}

Assistant:"""

PROMPT = PromptTemplate(prompt_template,input_variables = ["context","question"])


def get_response_llm(LLm , vectorstore_faiss , query):
    qa = retrieval_qa.from_chain_type(
    LLm = LLm,
    chain_type = 'stuff',
    retriever = vectorstore_faiss.as_retriever(
        search_type = 'similarity',
        search_kwargs = {"k":3} 
    ),
    return_source_documents = True,
    chain_type_kwargs = {'prompt': PROMPT}
    )
    answer = qa{'query':query}
    return answer['result']


if __name__=='__main__':
    query = "what is RAG token"
    llm = get_llama3_llm()
    vector_store_faiss = get_vector_store()
    get_response_llm(llm, query,vector_store_faiss)