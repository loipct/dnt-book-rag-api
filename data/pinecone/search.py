from model.resource import Resource
from .init import collection, retriever, cross_encoder_retriever
from langchain.docstore.document import Document
from typing import List, Optional, Union

def results_to_model(result:Document) -> Resource:
    return Resource(
                topic  = result.metadata["topic"],
                title = result.metadata["title"],
                principle   = result.metadata["principle"]
            )

def similarity_search(queries: List[str]) -> tuple[list[Resource], list[Document]]:
    docs = [retriever.invoke(subquery) for subquery in queries]
    docs = [doc for doc_sublist in docs for doc in doc_sublist]
    return docs
