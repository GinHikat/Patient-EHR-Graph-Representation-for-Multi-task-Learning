from pydantic import BaseModel
from typing import List, Dict, Any

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
