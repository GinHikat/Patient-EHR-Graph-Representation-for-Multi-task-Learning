from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel
from schemas.graph import GraphResponse, NodeTypesResponse, StatsResponse
from services.graph_service import (
    get_graph_data, 
    get_node_types_data, 
    get_node_by_id, 
    get_database_stats_data,
    get_cui_subgraph_data
)
import sys
import os

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from modules.extend.ner_engine import extract_entities_umls, extract_entities_llm, extract_entities_dl, extract_entities_ner

router = APIRouter()

class NlpAnalyzeRequest(BaseModel):
    text: str
    method: Optional[str] = "hybrid"
    threshold: Optional[float] = 0.5
    ner_model: Optional[str] = "phobert"
    ner_lang: Optional[str] = "vi"
    dl_model: Optional[str] = "auto"

@router.get("/graph", response_model=GraphResponse)
def get_graph(
    limit: int = Query(100), 
    node_types: Optional[List[str]] = Query(None, alias="node_type"), 
    edge_types: Optional[List[str]] = Query(None, alias="edge_type"),
    namespace: str = Query("Test")
):
    data = get_graph_data(limit, node_types, edge_types, namespace)
    return data

@router.get("/node_types", response_model=NodeTypesResponse)
def get_node_types():
    labels = get_node_types_data()
    return {"node_types": labels}

@router.get("/node/{node_id}", response_model=GraphResponse)
def search_node(
    node_id: str, 
    namespace: Optional[str] = Query(None),
    node_types: Optional[List[str]] = Query(None, alias="node_type")
):
    data = get_node_by_id(node_id, namespace, node_types)
    if not data:
        raise HTTPException(status_code=404, detail="Node not found")
    return data

@router.get("/stats", response_model=StatsResponse)
def get_stats():
    stats = get_database_stats_data()
    return stats

class RelationRequest(BaseModel):
    text: str
    head_span: List[int]
    target_span: List[int]
    head_term: str
    target_term: str

_re_engine = None

def get_re_engine():
    global _re_engine
    if _re_engine is None:
        from modules.extend.model.inference.inference_re import RelationExtractor
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "modules", "extend", "model", "statedict", "re", "english"))
        _re_engine = RelationExtractor(base_dir)
    return _re_engine

@router.post("/nlp/analyze")
def analyze_text(payload: NlpAnalyzeRequest):
    try:
        if payload.method == "llm":
            res = extract_entities_llm(payload.text)
        elif payload.method == "dl":
            res = extract_entities_dl(payload.text, threshold=payload.threshold, model_length=payload.dl_model)
        elif payload.method == "ner" or payload.method == "nere":
            # NERE uses the same underlying NER extraction first
            res = extract_entities_ner(payload.text, model_name=payload.ner_model, lang=payload.ner_lang)
        else:
            res = extract_entities_umls(payload.text)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"NLP analysis failed: {str(e)}")

@router.post("/nlp/relation")
def get_relation(payload: RelationRequest):
    try:
        re_engine = get_re_engine()
        pair = {
            "head_span": payload.head_span,
            "target_span": payload.target_span,
            "head_term": payload.head_term,
            "target_term": payload.target_term
        }
        results = re_engine.predict(payload.text, [pair])
        if results:
            return {"relation": results[0]["predicted_relation"]}
        return {"relation": "None"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Relation extraction failed: {str(e)}")

@router.get("/nlp/subgraph/{cui}", response_model=GraphResponse)
def get_cui_subgraph(
    cui: str,
    namespace: str = Query("Test"),
    rxnorm: Optional[str] = Query(None),
    snomed: Optional[str] = Query(None),
    mesh: Optional[str] = Query(None),
    drugbank: Optional[str] = Query(None),
    omim: Optional[str] = Query(None),
    icd9: Optional[str] = Query(None),
    icd10: Optional[str] = Query(None),
    loinc: Optional[str] = Query(None),
    pubchem: Optional[str] = Query(None),
    pubmed: Optional[str] = Query(None),
    hpo: Optional[str] = Query(None)
):
    try:
        data = get_cui_subgraph_data(
            cui=cui,
            namespace=namespace,
            rxnorm=rxnorm,
            snomed=snomed,
            mesh=mesh,
            drugbank=drugbank,
            omim=omim,
            icd9=icd9,
            icd10=icd10,
            loinc=loinc,
            pubchem=pubchem,
            pubmed=pubmed,
            hpo=hpo
        )
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch CUI subgraph: {str(e)}")

@router.get("/nlp/engine_status")
def get_engine_status():
    try:
        from modules.dataset_preprocessing.external.uml import is_engine_loaded
        return {"loaded": is_engine_loaded()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/nlp/start_engine")
def start_nlp_engine():
    try:
        from modules.dataset_preprocessing.external.uml import load_engine
        load_engine()
        return {"status": "success", "message": "Engine started successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start engine: {str(e)}")
