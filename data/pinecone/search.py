from model.resource import Resource
from .init import collection, retriever
from langchain.docstore.document import Document
from typing import List, Optional, Union

def results_to_model(result:Document) -> Resource:
    return Resource(
                topic  = result.metadata["topic"],
                title = result.metadata["title"],
                principle   = result.metadata["principle"]
            )

def similarity_search(query:str) -> tuple[list[Resource], list[Document]]:
    docs = retriever.invoke(query)
    return [results_to_model(document) for document in docs], docs