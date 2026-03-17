from fastapi import APIRouter, Query
from schemas.graph import GraphResponse, NodeTypesResponse
from services.graph_service import get_graph_data, get_node_types_data
from typing import Optional

router = APIRouter()

@router.get("/graph", response_model=GraphResponse)
def get_graph(limit: int = Query(100), node_type: Optional[str] = Query(None), namespace: str = Query("Test")):
    data = get_graph_data(limit, node_type, namespace)
    return data

@router.get("/node_types", response_model=NodeTypesResponse)
def get_node_types():
    labels = get_node_types_data()
    return {"node_types": labels}
