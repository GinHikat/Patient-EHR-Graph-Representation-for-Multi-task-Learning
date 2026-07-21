import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import TypedDict, List, Dict, Any
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# --- Setup Paths ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

var_dir = os.path.join(project_root, "VAR")
if var_dir not in sys.path:
    sys.path.append(var_dir)

# Import Utilities and SapBERT logic from VAR
try:
    from modules.utils import EntityExtractor, Utilities
    from modules.evaluation.test_sample_pipeline import load_dictionaries, get_best_row_lexical_sim
except ImportError as e:
    print(f"WARNING: Could not import VAR modules: {e}. Make sure you run this from the project root.")

# --- Define the State ---
class AgentState(TypedDict):
    text: str
    entities: List[Dict[str, Any]]
    feedback: str
    revision_count: int

# --- Pydantic Models (JSON Schema Match) ---
class EntityOutput(BaseModel):
    text: str = Field(description="The exact matched substring.")
    type: str = Field(description="MUST be one of: CHẨN_ĐOÁN, THUỐC, TÊN_XÉT_NGHIỆM, TRIỆU_CHỨNG, KẾT_QUẢ_XÉT_NGHIỆM")
    candidates: List[str] = Field(default_factory=list, description="Leave empty. To be filled by Ontology Mapper.")
    assertions: List[str] = Field(default_factory=list, description="Must be a list of exact strings: 'isNegated', 'isHistorical', 'isFamily'. Example: ['isNegated']. DO NOT output dicts.")
    position: List[int] = Field(description="[start, end] character offsets")

class RevisionOutput(BaseModel):
    entities: List[EntityOutput] = Field(description="The final corrected list of medical entities.")

# --- Global Models (Lazy Loaded) ---
_extractor = None
_ner_model = None
_sapbert_en = None
_utilities = None
_df_diag = _diag_embs = _df_drug = _drug_embs = _df_sym = _sym_embs = None

gpu_lock = threading.Lock()

def load_global_models():
    global _extractor, _ner_model, _sapbert_en, _utilities
    global _df_diag, _diag_embs, _df_drug, _drug_embs, _df_sym, _sym_embs
    if _extractor is None:
        print("    Loading global models and dictionaries (this will only happen once)...")
        _extractor = EntityExtractor(mode='ner + retrieval')
        _ner_model = _extractor._get_ner_instance(lang="vi")
        _sapbert_en = _extractor._get_sapbert_instance(lang="en")
        _utilities = Utilities()
        _df_diag, _diag_embs, _df_drug, _drug_embs, _df_sym, _sym_embs = load_dictionaries()

# --- NODES ---
def baseline_extractor(state: AgentState) -> AgentState:
    load_global_models()
    text = state["text"]
    
    # Run the Vietnamese NER model directly using the global model to prevent re-loading
    with gpu_lock:
        raw_entities = _ner_model.extract_entities(text)
    
    formatted_entities = []
    for ent in raw_entities:
        raw_type = ent.get("label", "")
        mapped_type = ""
        if raw_type in ["Disease", "Disease/Symptom", "Condition"]: mapped_type = "CHẨN_ĐOÁN"
        elif raw_type in ["Drug", "Medication", "Chemical"]: mapped_type = "THUỐC"
        elif raw_type in ["Procedure", "Test", "Procedure/Treatment"]: mapped_type = "TÊN_XÉT_NGHIỆM"
        
        if mapped_type:
            formatted_entities.append({
                "text": ent.get("term", ""),
                "type": mapped_type,
                "candidates": [],
                "assertions": [],
                "position": ent.get("offset", [0, 0])
            })
            
    return {
        "text": text,
        "entities": formatted_entities,
        "feedback": "",
        "revision_count": state.get("revision_count", 0)
    }

def format_entities_for_prompt(entities: List[Dict[str, Any]]) -> str:
    lines = []
    for i, ent in enumerate(entities):
        text = ent.get('text', '')
        t = ent.get('type', '')
        pos = ent.get('position', [])
        assertions = ent.get('assertions', [])
        
        line = f"[{i}] {t}: '{text}' (pos: {pos})"
        if assertions:
            line += f" [Assertions: {', '.join(assertions)}]"
        lines.append(line)
    return "\n".join(lines) if lines else "None"

def critic_agent(state: AgentState) -> AgentState:
    text = state["text"]
    entities = state["entities"]
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    system_prompt = """You are an expert Medical Entity Critic evaluating Vietnamese clinical text.
        Your job is to review the text and the current extracted entities based on STRICT guidelines.

        1. MUST ONLY use these 5 labels: CHẨN_ĐOÁN (Diagnosis), THUỐC (Drug), TÊN_XÉT_NGHIỆM (Procedure/Test), TRIỆU_CHỨNG (Symptom), KẾT_QUẢ_XÉT_NGHIỆM (Lab Result).
        2. NEGATED ENTITIES: Negated entities (e.g., following words like 'không', 'chưa') MUST still be extracted and saved. However, the entity's 'text' should NOT contain the negation word itself, and you MUST add 'isNegated' to its 'assertions' list. If the current extraction includes the negation word in the text (e.g., 'không ho'), tell the Reviser to fix the text to just 'ho', fix the 'position' offsets, and add the 'isNegated' assertion.
        3. ASSERTIONS: Check if `isNegated`, `isHistorical` (e.g. mentions of past history), or `isFamily` apply.
        4. MISSING ENTITIES: Check if any obvious drugs (like nitroglycerin), symptoms, or test results are missing from the text.
        5. DRUG BOUNDARIES: Check if THUỐC entities are missing their EXACT chemical dosage (e.g., 'aspirin 325mg'). However, DO NOT include conversational or temporal context like 'trong ngày', 'hàng ngày', 'sáng', 'chiều'. If the text is 'atenolol trong ngày', instruct the Reviser to reduce it to just 'atenolol'.
        6. LAB RESULTS SPLITTING: Check if a TÊN_XÉT_NGHIỆM has its result embedded inside it (e.g., "ecg bình thường", "nhịp tim 80"). If so, instruct the Reviser to SPLIT it into two separate entities: the test ("ecg" as TÊN_XÉT_NGHIỆM) and the result ("bình thường" as KẾT_QUẢ_XÉT_NGHIỆM). Also look for missed KẾT_QUẢ_XÉT_NGHIỆM nearby.
        7. GARBAGE FILTERING: Remove generic modifier words that are incorrectly extracted as standalone entities (e.g., "Nhẹ", "Nặng", "Bình thường", "Bất thường", "Cảm thấy", "Khó chịu", "Cảm giác"). These are not valid symptoms or diagnoses.

        If perfect, output exactly: APPROVED
        Otherwise, detail EXACTLY what the Reviser needs to fix (add/remove/modify, change assertions, fix position offsets)."""
    
    compact_entities = format_entities_for_prompt(entities)
    human_prompt = f"Original Text: {text}\n\nCurrent Entities:\n{compact_entities}"
    
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
    response = llm.invoke(messages)
    feedback = response.content.strip()
    
    return {"feedback": feedback}

def reviser_agent(state: AgentState) -> AgentState:
    text = state["text"]
    entities = state["entities"]
    feedback = state["feedback"]
    revision_count = state.get("revision_count", 0)
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    reviser_llm = llm.with_structured_output(RevisionOutput, method="function_calling")
    
    system_prompt = """You are an expert Medical Entity Reviser for Vietnamese clinical text.
You will be given the Original Text, Current Entities, and Critic Feedback.
Fix the entities list exactly as requested.
- If adding new entities, calculate the `position` as `[start_char_idx, end_char_idx]`.
- Enforce the 5 labels: CHẨN_ĐOÁN, THUỐC, TÊN_XÉT_NGHIỆM, TRIỆU_CHỨNG, KẾT_QUẢ_XÉT_NGHIỆM.
- Enforce assertions: MUST be a list of EXACT strings (e.g., ["isNegated"]). DO NOT output dictionaries. Valid strings: isNegated, isHistorical, isFamily. Leave as empty list [] if none apply, it's not required to have these assertions so whenever you are unsure, just ignore and leave an empty list [].
- For negated entities, ensure the 'text' does NOT include the negation word itself, adjust the 'position' accordingly, and add 'isNegated' to 'assertions'.
- Leave `candidates` empty ([])."""

    compact_entities = format_entities_for_prompt(entities)
    human_prompt = f"Original Text: {text}\n\nCurrent Entities:\n{compact_entities}\n\nCritic Feedback:\n{feedback}"
    
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
    result = reviser_llm.invoke(messages)
    
    return {
        "entities": [ent.model_dump() for ent in result.entities],
        "revision_count": revision_count + 1
    }

# --- Ontology Mapper Node (SapBERT Integration) ---
def ontology_mapper(state: AgentState) -> AgentState:
    load_global_models()
    global _sapbert_en, _df_diag, _diag_embs, _df_drug, _drug_embs, _df_sym, _sym_embs, _utilities
    
    entities = state["entities"]
        
    for ent in entities:
        mapped_type = ent["type"]
        display_term = ent["text"]
        candidates = []
        
        is_disease = mapped_type in ["CHẨN_ĐOÁN", "TRIỆU_CHỨNG"]
        
        # Dual-retrieval for Symptoms and Diagnoses
        if is_disease:
            with gpu_lock:
                en_term = _utilities.translate_vi2en(display_term.lower())
                emb = _sapbert_en.encode_text([en_term], show_progress=False)
            
            best_diag_sim = -1
            best_diag_id = None
            best_sym_sim = -1
            
            if emb.size > 0:
                # Compare to Diagnoses
                if not _df_diag.empty and _diag_embs.size > 0:
                    sims = cosine_similarity(emb, _diag_embs)[0]
                    top_3_idx = np.argsort(sims)[-3:][::-1]
                    best_hybrid_score = -1
                    for idx in top_3_idx:
                        sem_sim = sims[idx]
                        if sem_sim < 0.5: continue
                        lex_sim = get_best_row_lexical_sim(display_term, _df_diag.iloc[idx])
                        hybrid_score = sem_sim + lex_sim * 0.5
                        if hybrid_score > best_hybrid_score:
                            best_hybrid_score = hybrid_score
                            best_diag_sim = sem_sim
                            best_diag_id = str(_df_diag.iloc[idx]['id'])
                            
                # Compare to Symptoms
                if not _df_sym.empty and _sym_embs.size > 0:
                    sims = cosine_similarity(emb, _sym_embs)[0]
                    best_sym_sim = sims[np.argmax(sims)]
                    
            # Set the type dynamically based on score
            if best_diag_sim >= best_sym_sim and best_diag_id is not None and best_diag_sim >= 0.5:
                ent["type"] = "CHẨN_ĐOÁN"
                candidates = [best_diag_id]
            else:
                ent["type"] = "TRIỆU_CHỨNG"
                candidates = []
                
        # Retrieval for Drugs
        elif mapped_type == "THUỐC" and not _df_drug.empty and _drug_embs.size > 0:
            with gpu_lock:
                emb = _sapbert_en.encode_text([display_term.lower()], show_progress=False)
                
            if emb.size > 0:
                sims = cosine_similarity(emb, _drug_embs)[0]
                top_3_idx = np.argsort(sims)[-3:][::-1]
                best_hybrid_score = -1
                best_id = None
                best_sem_sim = -1
                for idx in top_3_idx:
                    sem_sim = sims[idx]
                    if sem_sim < 0.5: continue
                    lex_sim = get_best_row_lexical_sim(display_term, _df_drug.iloc[idx])
                    hybrid_score = sem_sim + lex_sim * 0.5
                    if hybrid_score > best_hybrid_score:
                        best_hybrid_score = hybrid_score
                        best_sem_sim = sem_sim
                        best_id = str(_df_drug.iloc[idx]['rxcui'])
                if best_id and best_sem_sim >= 0.6:
                    candidates = [best_id]
                    
        ent["candidates"] = candidates
        
    return {"entities": entities}

# --- ROUTING ---
def router(state: AgentState) -> str:
    feedback = state["feedback"]
    revision_count = state["revision_count"]
    
    if "APPROVED" in feedback.upper() or revision_count >= 3:
        return "ontology_mapper"
    else:
        return "reviser_agent"

# --- GRAPH COMPILATION ---
workflow = StateGraph(AgentState)

workflow.add_node("baseline_extractor", baseline_extractor)
workflow.add_node("critic_agent", critic_agent)
workflow.add_node("reviser_agent", reviser_agent)
workflow.add_node("ontology_mapper", ontology_mapper)

workflow.set_entry_point("baseline_extractor")
workflow.add_edge("baseline_extractor", "critic_agent")
workflow.add_conditional_edges(
    "critic_agent",
    router,
    {
        "reviser_agent": "reviser_agent",
        "ontology_mapper": "ontology_mapper"
    }
)
workflow.add_edge("reviser_agent", "critic_agent")
workflow.add_edge("ontology_mapper", END)

app = workflow.compile()

def run_pipeline(samples: int = None):
    # Pre-load models in the main thread to avoid race conditions in the worker threads
    load_global_models()
    
    test_dir = Path(var_dir) / "data" / "var" / "test"
    output_dir = Path(var_dir) / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    test_files = list(test_dir.glob("*.txt"))
    test_files.sort(key=lambda x: int(x.stem) if x.stem.isdigit() else x.stem)
    
    if samples:
        test_files = test_files[:samples]
        
    print(f"Loading {len(test_files)} test files to process from {test_dir}...")
    
    langfuse_handler = None
    callbacks = []
    if os.getenv("LANGFUSE_SECRET_KEY") and os.getenv("LANGFUSE_PUBLIC_KEY"):
        try:
            from langfuse.callback import CallbackHandler
            langfuse_handler = CallbackHandler()
            callbacks.append(langfuse_handler)
        except ImportError:
            pass
    config = {"callbacks": callbacks} if callbacks else {}

    def process_file(file_path):
        out_json_path = output_dir / f"{file_path.stem}.json"
        if out_json_path.exists():
            return
            
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        initial_state = {
            "text": text,
            "entities": [],
            "feedback": "",
            "revision_count": 0
        }
        
        final_state = app.invoke(initial_state, config=config)
        
        with open(out_json_path, 'w', encoding='utf-8') as f:
            json.dump(final_state["entities"], f, ensure_ascii=False, indent=4)

    # Run in parallel with 5 workers
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(process_file, fp): fp for fp in test_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Test Files"):
            try:
                future.result()
            except Exception as e:
                fp = futures[future]
                print(f"\nError processing {fp.stem}: {e}")

    if langfuse_handler is not None:
        langfuse_handler.flush()

# --- MAIN EXECUTION FOR TESTING ---
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY is not set.")
        sys.exit(1)
        
    parser = argparse.ArgumentParser(description="Run LangGraph Agent Pipeline")
    parser.add_argument("--samples", type=int, default=None, help="Number of files to process")
    args = parser.parse_args()
    
    run_pipeline(samples=args.samples)
