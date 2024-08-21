import os
from data.pinecone import search as search
from model.airesults import AIResults
from model.resource import Resource
from .multiQuery import *
from langchain_core.runnables import  RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

from config import config as config

llm_model_name = config.get_llm_model_config()['model_name']

def get_query(query:str)-> list[Resource]:
    resources, _ = search.similarity_search(query)
    return resources


def get_query_summary(query:str) -> str:
    prompt_template = """Write a summary of the following:
    "{text}"
    CONCISE SUMMARY:"""
    prompt = ChatPromptTemplate.from_template(prompt_template)
    resources, docs = search.similarity_search(query)

    if len(resources) == 0:
        return AIResults(text="No Documents Found",ResourceCollection=resources)

    llm_model = GoogleGenerativeAI(model=llm_model_name)

    stuff_chain = prompt | llm_model | StrOutputParser()

    return AIResults(text =  stuff_chain.invoke({'text' : docs}), ResourceCollection=resources) 

def get_qa_from_query(query:str, multiple_queries = True) -> str:
    if multiple_queries:
        queries = get_generated_queries(query)
        print("queries : ", queries)
        resources, docs = search.similarity_search(queries)
    else :
        resources, docs = search.similarity_search([query])
        
    print("docs : ", len(docs))

    if len(resources) == 0 :return AIResults(text="No Documents Found",ResourceCollection=resources)

    template = """
    Answer the question based on the context below. If you can't 
    answer the question, reply "I don't know".

    Context: {context}

    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)
    llm_model = GoogleGenerativeAI(model=llm_model_name)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    content = format_docs(docs)
    # print("content : ", content)
    rag_chain = (
    {"context": lambda x: content , "question": RunnablePassthrough()}
    | prompt
    | llm_model
    | StrOutputParser()
    )

    return AIResults(text=rag_chain.invoke(query),ResourceCollection=resources)

def get_qa_from_query_w_rerank(query:str, multiple_queries = True) -> str:
    if multiple_queries:
        queries = get_generated_queries(query)
        print("queries : ", queries)
        resources, docs = search.similarity_search_rerank(queries)
    else :
        resources, docs = search.similarity_search_rerank([query])  
    
    print("docs : ", len(docs))
    
    if len(resources) == 0 :return AIResults(text="No Documents Found",ResourceCollection=resources)

    template = """
    Answer the question based on the context below. If you can't 
    answer the question, reply "I don't know".

    Context: {context}

    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)
    llm_model = GoogleGenerativeAI(model=llm_model_name)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    content = format_docs(docs)

    rag_chain = (
    {"context": lambda x: content , "question": RunnablePassthrough()}
    | prompt
    | llm_model
    | StrOutputParser()
    )

    return AIResults(text=rag_chain.invoke(query),ResourceCollection=resources)

def get_llm_response(query:str) -> str:
    template = """
    Answer the question. If you can't 
    answer the question, reply "I don't know".
    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)
    llm_model = GoogleGenerativeAI(model=llm_model_name)

    rag_chain = (
    {"question": RunnablePassthrough()}
    | prompt
    | llm_model
    | StrOutputParser()
    )
    default_text = "This question is not related to the book !! This is the answer based on my knowledge :\n\n"
    return AIResults(text=default_text + rag_chain.invoke(query),ResourceCollection=[])