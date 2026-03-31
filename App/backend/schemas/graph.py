from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class Node(BaseModel):
    id: str
    labels: List[str]
    properties: Dict[str, Any]

class Link(BaseModel):
    source: str
    target: str
    type: str

class GraphResponse(BaseModel):
    nodes: List[Node]
    links: List[Link]

class NodeTypesResponse(BaseModel):
    node_types: List[str]

class BreakdownItem(BaseModel):
    type: Optional[str] = None
    labels: Optional[List[str]] = None
    count: int

class StatsResponse(BaseModel):
    total_nodes: int
    node_breakdown: List[BreakdownItem]
    total_edges: int
    edge_breakdown: List[BreakdownItem]
