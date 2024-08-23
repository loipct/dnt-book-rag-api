from fastapi import APIRouter
from service import search as search 
from service import route as route 
from model.resource import Resource
from model.airesults import AIResults
from typing import Literal

router = APIRouter(prefix="/search")


@router.get("/{query}")
def get_search(query) -> list[Resource]:
    return search.get_query(query)


@router.get("/adaptive_query/{query}")
def get_adaptive_query(query, rerank_mode: bool = True, query_category = Literal["Auto", "Factual", "Analytical"]) -> AIResults:
    if route.routing_query(query):
        print("This question is related to the book !!")
        return search.get_adaptive_query(query, rerank_mode, query_category)
    return search.get_llm_response(query)
