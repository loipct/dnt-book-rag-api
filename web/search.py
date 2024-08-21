from fastapi import APIRouter
from service import search as search 
from service import route as route 
from model.resource import Resource
from model.airesults import AIResults

router = APIRouter(prefix="/search")


@router.get("/{query}")
def get_search(query) -> list[Resource]:
    return search.get_query(query)


@router.get("/summary/{query}")
def get_query_summary(query) -> AIResults:
    return search.get_query_summary(query)


@router.get("/qa/{query}")
def get_query_qa(query) -> AIResults:
    if route.routing_query(query):
        print("This question is not related to the book !!")
        return search.get_qa_from_query(query)
    return search.get_llm_response(query)

@router.get("/qa_w_rerank/{query}")
def get_query_qa_w_rerank(query) -> AIResults:
    if route.routing_query(query):
        print("This question is not related to the book !!")
        return search.get_qa_from_query_w_rerank(query)
    return search.get_llm_response(query)