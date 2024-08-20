from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Dict, Any, Tuple
from config import config as config

llm_model_name = config.get_llm_model_config()['model_name']



class multiple_queries(BaseModel):  
    # setup: str = Field(description="Original query")
    query1: str  = Field(description="query 1")
    query2: str  = Field(description="query 2")
    query3: str  = Field(description="query 3")


def get_generated_queries(query, k_queries = 5):
    # RAG-Fusion: Related
    template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
    Generate multiple search queries related to: {question} \n
    Output (3 queries):"""
    prompt_rag_fusion = ChatPromptTemplate.from_template(template)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.8, top_p=0.5)
    structured_llm = llm.with_structured_output(multiple_queries)
    # generate_queries = (
    #     prompt_rag_fusion 
    #     | llm
    #     | StrOutputParser() 
    #     | (lambda x: x.split("\n"))
    # )
    generate_queries = (
        prompt_rag_fusion 
        | structured_llm
    )
    result = generate_queries.invoke({"question" : query})
    return [result.query1,result.query2,result.query3]
