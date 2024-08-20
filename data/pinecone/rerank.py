from langchain_core.retrievers import BaseRetriever
from sentence_transformers import CrossEncoder
from langchain.docstore.document import Document
from typing import List, Dict, Any, Tuple
from langchain_core.pydantic_v1 import BaseModel, Field

class CrossEncoderRetriever(BaseRetriever, BaseModel):
    vectorstore: Any = Field(description="Vector store for initial retrieval")
    cross_encoder: Any = Field(description="Cross-encoder model for reranking")
    k: int = Field(default=5, description="Number of documents to retrieve initially")
    rerank_top_k: int = Field(default=3, description="Number of documents to return after reranking")

    class Config:
        arbitrary_types_allowed = True

    def get_relevant_documents(self, query: str) -> List[Document]:
        # Initial retrieval
        initial_docs = self.vectorstore.similarity_search(query, k=self.k)
        
        # Prepare pairs for cross-encoder
        pairs = [[query, doc.page_content] for doc in initial_docs]
        
        # Get cross-encoder scores
        scores = self.cross_encoder.predict(pairs)
        
        # Sort documents by score
        scored_docs = sorted(zip(initial_docs, scores), key=lambda x: x[1], reverse=True)
        
        # Return top reranked documents
        return [doc for doc, _ in scored_docs[:self.rerank_top_k]]

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError("Async retrieval not implemented")