from typing import Optional, List
from fastapi import APIRouter, Query, HTTPException
from schemas.graph import GraphResponse, NodeTypesResponse, StatsResponse
from services.graph_service import get_graph_data, get_node_types_data, get_node_by_id, get_database_stats_data

router = APIRouter()

@router.get("/graph", response_model=GraphResponse)
def get_graph(
    limit: int = Query(100), 
    node_types: Optional[List[str]] = Query(None, alias="node_type"), 
    namespace: str = Query("Test")
):
    # If node_types is passed as multiple query params like ?node_type=A&node_type=B
    data = get_graph_data(limit, node_types, namespace)
    return data

@router.get("/node_types", response_model=NodeTypesResponse)
def get_node_types():
    labels = get_node_types_data()
    return {"node_types": labels}

@router.get("/node/{node_id}", response_model=GraphResponse)
def search_node(node_id: str, namespace: str = Query("Test")):
    data = get_node_by_id(node_id, namespace)
    if not data:
        raise HTTPException(status_code=404, detail="Node not found")
    return data

@router.get("/stats", response_model=StatsResponse)
def get_stats():
    node_count, edge_count = get_database_stats_data()
    return {"total_nodes": node_count, "total_edges": edge_count}
