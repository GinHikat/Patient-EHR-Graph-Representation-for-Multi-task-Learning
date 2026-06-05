import React, { useState, useEffect, useRef } from "react";
import axios from "axios";
import ForceGraph2D from "react-force-graph-2d";
import {
  Brain,
  Play,
  Sparkles,
  Link2,
  FileText,
  AlertCircle,
  Database,
  Activity,
  Trash2,
  Plus,
  Save,
  Network,
  X
} from "lucide-react";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000/api";

const SAMPLE_NOTES = [
  {
    title: "Discharge Summary: Cardiovascular",
    text: "Patient is a 67-year-old male with a history of HTN, osteoarthritis, and type 2 DM. Presented to the ED with acute onset of SOB and crushing chest pain. Was diagnosed with unstable angina. Prescribed Metoprolol 25mg PO BID and Aspirin 81mg daily. Echocardiogram showed LVEF of 45%. Scheduled for outpatient PTCA next week."
  },
  {
    title: "Radiology Report: Chest",
    text: "EXAMINATION:  LIVER OR GALLBLADDER US (SINGLE ORGAN). \n\nTECHNIQUE:  Grey scale and color Doppler ultrasound images of the abdomen were	obtained. \n\nHISTORY: History of COPD and worsening dyspnea. Rule out pneumonia.\n\nFINDINGS: The lungs are hyperinflated. No focal consolidation, pneumothorax, or large pleural effusion. Calcified mediastinal lymph nodes and mild cardiomegaly are noted. Degenerative changes of the thoracic spine. EKG shows sinus tachycardia."
  },
  {
    title: "Admitting Note: Renal/Endocrine",
    text: "Pt with history of ESRD on hemodialysis, CKD stage 5. Presented with generalized fatigue and weakness. Lab results: potassium 6.2 mmol/L (critical high), creatinine 5.8 mg/dL. EKG showed peaked T-waves. Treated with IV calcium gluconate and insulin/dextrose in the ER. Admitted for urgent dialysis."
  }
];

// Predefined palette for Concept categories and Database Node Types
const TYPE_COLORS = {
  Patient: "#3b82f6",
  Admission: "#ec4899",
  Disease: "#ff2a5f",
  Diagnosis: "#ffb000",
  Phenotype: "#ec4899",
  BodyParts: "#06b6d4",
  "Body Parts": "#06b6d4",
  Drugs: "#10b981",
  Drug: "#10b981",
  Chemicals: "#34d399",
  Chemical: "#34d399",
  Procedures: "#a855f7",
  Procedure: "#a855f7",
  Labs: "#6366f1",
  Lab: "#6366f1",
  Devices: "#94a3b8",
  Device: "#94a3b8",
  Concept: "#a855f7",
  Outpatient: "#fb923c"
};

function NlpSandbox() {
  const [inputText, setInputText] = useState(SAMPLE_NOTES[0].text);
  const [method, setMethod] = useState("hybrid");
  const [loading, setLoading] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);

  // Engine loading states
  const [engineLoaded, setEngineLoaded] = useState(false);
  const [engineChecking, setEngineChecking] = useState(true);
  const [engineLoading, setEngineLoading] = useState(false);

  // Tokenization and editing states
  const [selectedToken, setSelectedToken] = useState(null);
  const [selectedEntity, setSelectedEntity] = useState(null);

  // Form states for manual tag editing
  const [editCui, setEditCui] = useState("");
  const [editCanonicalName, setEditCanonicalName] = useState("");
  const [editCategory, setEditCategory] = useState("Disease");
  const [editIcd10, setEditIcd10] = useState("");
  const [editRxnorm, setEditRxnorm] = useState("");
  const [editSnomed, setEditSnomed] = useState("");
  const [editLoinc, setEditLoinc] = useState("");
  const [editMesh, setEditMesh] = useState("");
  const [editOmim, setEditOmim] = useState("");
  const [editIcd9, setEditIcd9] = useState("");
  const [editDrugbank, setEditDrugbank] = useState("");
  const [editHpo, setEditHpo] = useState("");
  const [editPubchem, setEditPubchem] = useState("");
  const [editPubmed, setEditPubmed] = useState("");
  const [editDetailClass, setEditDetailClass] = useState("");

  // Track manually added vocab databases for the current token session
  const [manuallyAddedDbs, setManuallyAddedDbs] = useState([]);

  // Subgraph visualization states
  const [activeMiddleTab, setActiveMiddleTab] = useState("reader");
  const [subgraphCui, setSubgraphCui] = useState("");
  const [subgraphData, setSubgraphData] = useState({ nodes: [], links: [] });
  const [subgraphLoading, setSubgraphLoading] = useState(false);

  const graphContainerRef = useRef(null);
  const [graphDimensions, setGraphDimensions] = useState({ width: 500, height: 400 });

  // Check engine status on mount
  useEffect(() => {
    const checkEngineStatus = async () => {
      try {
        const response = await axios.get(`${API_BASE_URL}/nlp/engine_status`);
        setEngineLoaded(response.data.loaded);
      } catch (err) {
        console.error("Failed to check NLP engine status", err);
      } finally {
        setEngineChecking(false);
      }
    };
    checkEngineStatus();
  }, []);

  const startEngine = async () => {
    setEngineLoading(true);
    try {
      await axios.post(`${API_BASE_URL}/nlp/start_engine`);
      setEngineLoaded(true);
    } catch (err) {
      console.error("Failed to start NLP engine", err);
      alert("Failed to start Clinical NLP engine: " + (err.response?.data?.detail || err.message));
    } finally {
      setEngineLoading(false);
    }
  };

  // Handle graph resizing inside columns
  useEffect(() => {
    if (graphContainerRef.current) {
      const resizeObserver = new ResizeObserver((entries) => {
        for (let entry of entries) {
          setGraphDimensions({
            width: entry.contentRect.width || 500,
            height: 400
          });
        }
      });
      resizeObserver.observe(graphContainerRef.current);
      return () => resizeObserver.disconnect();
    }
  }, [activeMiddleTab]);

  const runAnalysis = async () => {
    setLoading(true);
    setSelectedToken(null);
    setSelectedEntity(null);
    try {
      const response = await axios.post(`${API_BASE_URL}/nlp/analyze`, {
        text: inputText,
        method: method
      });
      setAnalysisResult(response.data);
    } catch (err) {
      console.error("NLP analysis failed", err);
      alert("Failed to analyze clinical text: " + (err.response?.data?.detail || err.message));
    } finally {
      setLoading(false);
    }
  };

  // Helper to segment note text into words, punctuation, and whitespaces
  const tokenizeText = (text) => {
    if (!text) return [];
    const regex = /(\w+|[^\w\s]|\s+)/g;
    let match;
    const tokens = [];
    while ((match = regex.exec(text)) !== null) {
      const textVal = match[0];
      tokens.push({
        text: textVal,
        start: match.index,
        end: regex.lastIndex,
        isWord: /\w+/.test(textVal)
      });
    }
    return tokens;
  };

  // Maps token offset bounds to recognized clinical entity bounds
  const getEntityForToken = (token, entities) => {
    if (!token || !token.isWord) return null;
    if (!entities) return null;
    return entities.find(ent => token.start >= ent.start && token.end <= ent.end);
  };

  const handleTokenClick = (token) => {
    setSelectedToken(token);
    const entity = getEntityForToken(token, analysisResult?.entities);
    if (entity) {
      setSelectedEntity(entity);
      setEditCui(entity.cui || "");
      setEditCanonicalName(entity.canonical_name || "");
      setEditCategory(entity.category || "Disease");
      setEditDetailClass(entity.type || "Unknown");
      setEditIcd10(entity.codes?.icd10 || "");
      setEditRxnorm(entity.codes?.rxnorm || "");
      setEditSnomed(entity.codes?.snomed || "");
      setEditLoinc(entity.codes?.loinc || "");
      setEditMesh(entity.codes?.mesh || "");
      setEditOmim(entity.codes?.omim || "");
      setEditIcd9(entity.codes?.icd9 || "");
      setEditDrugbank(entity.codes?.drugbank || "");
      setEditHpo(entity.codes?.hpo || "");
      setEditPubchem(entity.codes?.pubchem || "");
      setEditPubmed(entity.codes?.pubmed || "");
    } else {
      setSelectedEntity(null);
      // Clean target CUI to only alphanumeric for custom creation
      const cleanWord = token.text.toUpperCase().replace(/[^A-Z0-9]/g, "");
      setEditCui(`CUI_MANUAL_${cleanWord}`);
      setEditCanonicalName(token.text);
      setEditCategory("Disease");
      setEditDetailClass("User Defined");
      setEditIcd10("");
      setEditRxnorm("");
      setEditSnomed("");
      setEditLoinc("");
      setEditMesh("");
      setEditOmim("");
      setEditIcd9("");
      setEditDrugbank("");
      setEditHpo("");
      setEditPubchem("");
      setEditPubmed("");
    }
    setManuallyAddedDbs([]);
  };

  // Manual Annotation: Create New Tag
  const handleCreateTag = () => {
    if (!selectedToken || !analysisResult) return;
    const newEntity = {
      start: selectedToken.start,
      end: selectedToken.end,
      text: selectedToken.text,
      canonical_name: editCanonicalName || selectedToken.text,
      cui: editCui || `CUI_MANUAL_${selectedToken.text.toUpperCase()}`,
      similarity: 1.0,
      type: editDetailClass || "Manual Annotation",
      category: editCategory,
      codes: {
        icd10: editIcd10,
        rxnorm: editRxnorm,
        snomed: editSnomed,
        loinc: editLoinc,
        mesh: editMesh,
        omim: editOmim,
        icd9: editIcd9,
        drugbank: editDrugbank,
        hpo: editHpo,
        pubchem: editPubchem,
        pubmed: editPubmed
      }
    };
    const updatedEntities = [...(analysisResult.entities || []), newEntity];
    updatedEntities.sort((a, b) => a.start - b.start);
    setAnalysisResult({ ...analysisResult, entities: updatedEntities });
    setSelectedEntity(newEntity);
  };

  // Manual Annotation: Update Tag Details
  const handleUpdateTag = () => {
    if (!selectedEntity || !analysisResult) return;
    const updatedEntities = analysisResult.entities.map(ent => {
      if (ent.start === selectedEntity.start && ent.end === selectedEntity.end) {
        return {
          ...ent,
          canonical_name: editCanonicalName,
          cui: editCui,
          category: editCategory,
          type: editDetailClass,
          codes: {
            icd10: editIcd10,
            rxnorm: editRxnorm,
            snomed: editSnomed,
            loinc: editLoinc,
            mesh: editMesh,
            omim: editOmim,
            icd9: editIcd9,
            drugbank: editDrugbank,
            hpo: editHpo,
            pubchem: editPubchem,
            pubmed: editPubmed
          }
        };
      }
      return ent;
    });
    setAnalysisResult({ ...analysisResult, entities: updatedEntities });
    setSelectedEntity({
      ...selectedEntity,
      canonical_name: editCanonicalName,
      cui: editCui,
      category: editCategory,
      type: editDetailClass,
      codes: {
        icd10: editIcd10,
        rxnorm: editRxnorm,
        snomed: editSnomed,
        loinc: editLoinc,
        mesh: editMesh,
        omim: editOmim,
        icd9: editIcd9,
        drugbank: editDrugbank,
        hpo: editHpo,
        pubchem: editPubchem,
        pubmed: editPubmed
      }
    });
  };

  // Manual Annotation: Delete Tag
  const handleRemoveTag = () => {
    if (!selectedEntity || !analysisResult) return;
    const updatedEntities = analysisResult.entities.filter(
      ent => !(ent.start === selectedEntity.start && ent.end === selectedEntity.end)
    );
    setAnalysisResult({ ...analysisResult, entities: updatedEntities });
    setSelectedEntity(null);
    setSelectedToken(null);
  };

  // Fetch CUI subgraph neighborhood from Neo4j
  const handleMapSubgraph = async (cui, codes = null) => {
    if (!cui) return;
    setSubgraphLoading(true);
    setSubgraphCui(cui);
    setActiveMiddleTab("subgraph");
    try {
      const params = new URLSearchParams();
      const targetCodes = codes || {
        rxnorm: editRxnorm,
        snomed: editSnomed,
        mesh: editMesh,
        drugbank: editDrugbank,
        omim: editOmim,
        icd9: editIcd9,
        icd10: editIcd10,
        loinc: editLoinc,
        pubchem: editPubchem,
        pubmed: editPubmed,
        hpo: editHpo
      };
      
      Object.entries(targetCodes).forEach(([key, val]) => {
        if (val) params.append(key, val);
      });

      const url = `${API_BASE_URL}/nlp/subgraph/${cui}?${params.toString()}`;
      const response = await axios.get(url);
      setSubgraphData(response.data);
    } catch (err) {
      console.error("Subgraph query failed", err);
      alert("Failed to fetch CUI subgraph from Neo4j database.");
    } finally {
      setSubgraphLoading(false);
    }
  };

  const renderTokenizedText = () => {
    if (!analysisResult) {
      return (
        <div className="no-selection-card py-5">
          <Activity className="pulse-icon text-muted" size={40} />
          <p>Click "Run Pipeline" to analyze and extract clinical concepts.</p>
        </div>
      );
    }

    const { original_text, entities } = analysisResult;
    const sortedEntities = [...(entities || [])].sort((a, b) => a.start - b.start);
    
    const elements = [];
    let lastIndex = 0;

    // Helper to tokenize unannotated plain text into clickable word spans
    const renderPlainSpans = (textSegment, startOffset) => {
      const tokens = tokenizeText(textSegment);
      return tokens.map((token, subIdx) => {
        const globalStart = startOffset + token.start;
        const globalEnd = startOffset + token.end;
        
        if (/^\s+$/.test(token.text)) {
          return <span key={`space-${globalStart}`}>{token.text}</span>;
        }

        const isSelected = selectedToken && selectedToken.start === globalStart && selectedToken.end === globalEnd;
        let tokenClass = "nlp-token";
        if (token.isWord) {
          tokenClass += " word";
        }
        if (isSelected) {
          tokenClass += " selected";
        }

        return (
          <span
            key={`word-${globalStart}`}
            className={tokenClass}
            onClick={() => handleTokenClick({
              text: token.text,
              start: globalStart,
              end: globalEnd,
              isWord: token.isWord
            })}
          >
            {token.text}
          </span>
        );
      });
    };

    sortedEntities.forEach((entity, idx) => {
      // Add leading plain text (rendered word by word)
      if (entity.start > lastIndex) {
        const textSegment = original_text.substring(lastIndex, entity.start);
        elements.push(
          <React.Fragment key={`plain-${lastIndex}`}>
            {renderPlainSpans(textSegment, lastIndex)}
          </React.Fragment>
        );
      }

      // Add grouped entity as a single highlighted tag
      const isSelected = selectedEntity && selectedEntity.start === entity.start && selectedEntity.end === entity.end;
      const cleanCat = entity.category.toLowerCase().replace(/\s+/g, "");
      let categoryClass = `nlp-span-${cleanCat}`;
      
      let tokenClass = `nlp-token entity ${categoryClass}`;
      if (isSelected) {
        tokenClass += " selected";
      }

      elements.push(
        <span
          key={`entity-${entity.start}`}
          className={tokenClass}
          onClick={() => {
            setSelectedEntity(entity);
            setSelectedToken({
              text: entity.text,
              start: entity.start,
              end: entity.end,
              isWord: true
            });
            // Sync form states for editing
            setEditCui(entity.cui || "");
            setEditCanonicalName(entity.canonical_name || "");
            setEditCategory(entity.category || "Disease");
            setEditDetailClass(entity.type || "Unknown");
            setEditIcd10(entity.codes?.icd10 || "");
            setEditRxnorm(entity.codes?.rxnorm || "");
            setEditSnomed(entity.codes?.snomed || "");
            setEditLoinc(entity.codes?.loinc || "");
            setEditMesh(entity.codes?.mesh || "");
            setEditOmim(entity.codes?.omim || "");
            setEditIcd9(entity.codes?.icd9 || "");
            setEditDrugbank(entity.codes?.drugbank || "");
            setEditHpo(entity.codes?.hpo || "");
            setEditPubchem(entity.codes?.pubchem || "");
            setEditPubmed(entity.codes?.pubmed || "");
            setManuallyAddedDbs([]);
          }}
        >
          {entity.text}
          <span className="entity-badge-sup ml-1" style={{ fontSize: "0.65rem", textTransform: "uppercase", opacity: 0.85 }}>
            {entity.category.substring(0, 3)}
          </span>
        </span>
      );

      lastIndex = entity.end;
    });

    // Add trailing plain text
    if (lastIndex < original_text.length) {
      const textSegment = original_text.substring(lastIndex);
      elements.push(
        <React.Fragment key={`plain-${lastIndex}`}>
          {renderPlainSpans(textSegment, lastIndex)}
        </React.Fragment>
      );
    }

    return (
      <div className="nlp-highlighted-content" style={{ lineHeight: "2.9rem" }}>
        {elements}
      </div>
    );
  };

  const renderSubgraphGraph = () => {
    if (subgraphLoading) {
      return (
        <div className="flex flex-col items-center justify-center h-[350px]">
          <div className="spinner mb-4"></div>
          <p className="text-muted">Querying Neo4j database for CUI "{subgraphCui}"...</p>
        </div>
      );
    }

    if (!subgraphData || !subgraphData.nodes || subgraphData.nodes.length === 0) {
      return (
        <div className="no-selection-card py-5">
          <Database className="text-muted mb-3" size={40} />
          <p>No Neo4j nodes mapped for CUI "{subgraphCui}".</p>
          <p className="text-xs text-muted mt-2">The concept is not connected in the active graph database.</p>
        </div>
      );
    }

    return (
      <div 
        ref={graphContainerRef}
        className="subgraph-frame"
        style={{ height: "400px" }}
      >
        <div className="absolute top-2 left-2 z-10 bg-black/60 p-2 rounded text-xs border border-white/10">
          <div className="font-semibold text-cyan">CUI Neighborhood</div>
          <div className="text-muted">Nodes: {subgraphData.nodes.length} | Edges: {subgraphData.links.length}</div>
        </div>
        <ForceGraph2D
          graphData={subgraphData}
          width={graphDimensions.width}
          height={graphDimensions.height}
          nodeRelSize={6.5}
          nodeLabel={(node) => `
            <div style="background: rgba(0,0,0,0.9); padding: 8px; border-radius: 4px; color: white; font-family: sans-serif; font-size: 11px;">
              <div style="color: #60a5fa; font-weight: bold; margin-bottom: 2px;">${node.labels.filter(l => l !== "Test").join(', ')}</div>
              <div>${node.properties?.name || node.properties?.title || node.id}</div>
            </div>
          `}
          nodeCanvasObject={(node, ctx, globalScale) => {
            const label = node.labels.find(l => TYPE_COLORS[l]) || node.labels[0] || "Concept";
            const color = TYPE_COLORS[label] || "#94a3b8";
            const radius = 6.5;
            
            ctx.save();
            ctx.beginPath();
            ctx.arc(node.x, node.y, radius, 0, 2 * Math.PI, false);
            ctx.fillStyle = color;
            ctx.fill();
            
            ctx.strokeStyle = "rgba(255, 255, 255, 0.9)";
            ctx.lineWidth = 1.2 / globalScale;
            ctx.stroke();
            ctx.restore();
          }}
          linkColor={() => "rgba(255, 255, 255, 0.2)"}
          linkWidth={1.5}
          linkDirectionalArrowLength={2.5}
          cooldownTicks={80}
          backgroundColor="rgba(0,0,0,0)"
        />
      </div>
    );
  };

  const DB_OPTIONS = [
    { key: "icd10", label: "ICD-10", placeholder: "ICD-10 code" },
    { key: "rxnorm", label: "RxNorm", placeholder: "RxNorm code" },
    { key: "snomed", label: "SNOMED", placeholder: "SNOMED code" },
    { key: "loinc", label: "LOINC", placeholder: "LOINC code" },
    { key: "mesh", label: "MeSH", placeholder: "MeSH code" },
    { key: "omim", label: "OMIM", placeholder: "OMIM code" },
    { key: "icd9", label: "ICD-9", placeholder: "ICD-9 code" },
    { key: "drugbank", label: "DrugBank", placeholder: "DrugBank code" },
    { key: "hpo", label: "HPO", placeholder: "HPO code" },
    { key: "pubchem", label: "PubChem", placeholder: "PubChem code" },
    { key: "pubmed", label: "PubMed", placeholder: "PubMed code" }
  ];

  const getDbValue = (dbKey) => {
    switch (dbKey) {
      case "icd10": return editIcd10;
      case "rxnorm": return editRxnorm;
      case "snomed": return editSnomed;
      case "loinc": return editLoinc;
      case "mesh": return editMesh;
      case "omim": return editOmim;
      case "icd9": return editIcd9;
      case "drugbank": return editDrugbank;
      case "hpo": return editHpo;
      case "pubchem": return editPubchem;
      case "pubmed": return editPubmed;
      default: return "";
    }
  };

  const setDbValue = (dbKey, val) => {
    switch (dbKey) {
      case "icd10": setEditIcd10(val); break;
      case "rxnorm": setEditRxnorm(val); break;
      case "snomed": setEditSnomed(val); break;
      case "loinc": setEditLoinc(val); break;
      case "mesh": setEditMesh(val); break;
      case "omim": setEditOmim(val); break;
      case "icd9": setEditIcd9(val); break;
      case "drugbank": setEditDrugbank(val); break;
      case "hpo": setEditHpo(val); break;
      case "pubchem": setEditPubchem(val); break;
      case "pubmed": setEditPubmed(val); break;
      default: break;
    }
  };

  const isDbVisible = (dbKey) => {
    if (getDbValue(dbKey)) return true;
    return manuallyAddedDbs.includes(dbKey);
  };

  const visibleDbs = DB_OPTIONS.filter(db => isDbVisible(db.key));
  const hiddenDbs = DB_OPTIONS.filter(db => !isDbVisible(db.key));

  if (engineChecking) {
    return (
      <div className="flex flex-col items-center justify-center h-full min-h-[500px] w-full text-white">
        <Sparkles size={32} className="text-cyan spin mb-4" />
        <p className="text-muted text-sm font-semibold tracking-wider uppercase">Checking NLP Engine Status...</p>
      </div>
    );
  }

  if (!engineLoaded) {
    return (
      <div className="nlp-engine-start-container">
        <div className="nlp-engine-card glass-panel animate-fade-in">
          <div className="absolute -top-10 -right-10 w-40 h-40 bg-cyan/10 rounded-full blur-3xl"></div>
          <div className="absolute -bottom-10 -left-10 w-40 h-40 bg-rose/10 rounded-full blur-3xl"></div>
          
          <div className="flex justify-center mb-2">
            <div className={`nlp-engine-icon-wrapper ${engineLoading ? 'pulsing' : ''}`}>
              <Brain size={56} className={`text-cyan ${engineLoading ? 'spin-slow' : ''}`} />
            </div>
          </div>
          
          <h2>Clinical NLP Engine</h2>
          
          <p className="nlp-engine-description">
            The clinical NLP engine utilizes <span className="nlp-engine-highlight">QuickUMLS</span> and the <span className="nlp-engine-highlight">UMLS Metathesaurus</span> to extract, map, and resolve biomedical concepts from unstructured clinical notes.
            <span className="nlp-engine-status-sub">
              Initializing the engine loads the full dictionaries (ICD-10, SNOMED CT, RxNorm) and entity indexes into memory. This process may take up to 60 seconds.
            </span>
          </p>

          <div className="flex flex-col gap-4 items-center w-full mt-2">
            <button
              className={`search-btn w-full max-w-[250px] flex items-center justify-center gap-2 text-[15px] py-3 shadow-lg ${
                engineLoading ? 'opacity-70 cursor-not-allowed' : ''
              }`}
              onClick={startEngine}
              disabled={engineLoading}
              style={{ height: '50px' }}
            >
              {engineLoading ? (
                <>
                  <Sparkles size={18} className="spin" />
                  <span>Starting Engine...</span>
                </>
              ) : (
                <>
                  <Play size={18} />
                  <span>Start Engine</span>
                </>
              )}
            </button>
            
            <div className="nlp-engine-status-row">
              <span className={`status-indicator ${engineLoading ? 'initializing' : 'stopped'}`}></span>
              <span>Status: {engineLoading ? 'Initializing Dictionaries...' : 'Offline / Unloaded'}</span>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="nlp-sandbox-layout">
      {/* LEFT COLUMN: Note Input and Selection */}
      <div className="nlp-column glass-panel">
        <div className="panel-header">
          <Brain className="panel-icon text-cyan" />
          <h2>Clinical Text Processor</h2>
        </div>

        <div className="sample-notes-bar">
          <span className="section-label">Select Sample Case:</span>
          <div className="sample-buttons">
            {SAMPLE_NOTES.map((sample, idx) => (
              <button
                key={idx}
                className="sample-btn"
                onClick={() => {
                  setInputText(sample.text);
                  setAnalysisResult(null);
                  setSelectedToken(null);
                  setSelectedEntity(null);
                }}
              >
                Case {idx + 1}
              </button>
            ))}
          </div>
        </div>

        <div className="nlp-input-wrapper">
          <label className="section-label">Unstructured Clinical Note</label>
          <textarea
            className="nlp-textarea custom-input"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            placeholder="Type or paste medical note, clinical chart, or discharge summary here..."
          />
        </div>

        <div className="nlp-settings-row">
          <div className="setting-control">
            <span className="section-label">Method</span>
            <select
              className="custom-select"
              value={method}
              onChange={(e) => setMethod(e.target.value)}
            >
              <option value="hybrid">Hybrid Engine (QuickUMLS + LLM)</option>
              <option value="local">UMLS Dictionary Matcher (Local)</option>
              <option value="llm">Zero-Shot Medical LLM (Cloud)</option>
            </select>
          </div>
          
          <button
            className="search-btn run-nlp-btn"
            onClick={runAnalysis}
            disabled={loading || !inputText.trim()}
          >
            {loading ? (
              <>
                <Sparkles size={16} className="spin mr-2" />
                Parsing...
              </>
            ) : (
              <>
                <Play size={16} className="mr-2" />
                Run Pipeline
              </>
            )}
          </button>
        </div>
      </div>

      {/* MIDDLE COLUMN: Interactive Highlighted Note & Subgraph Visualizer */}
      <div className="nlp-column glass-panel">
        <div className="middle-panel-tabs">
          <button
            className={`tab-btn flex items-center gap-1 transition-all ${
              activeMiddleTab === "reader" ? "active" : ""
            }`}
            onClick={() => setActiveMiddleTab("reader")}
          >
            <FileText size={16} />
            Annotated Note
          </button>
          <button
            className={`tab-btn flex items-center gap-1 transition-all ${
              activeMiddleTab === "subgraph" ? "active" : ""
            }`}
            onClick={() => {
              if (selectedEntity) {
                handleMapSubgraph(selectedEntity.cui, selectedEntity.codes);
              } else {
                setActiveMiddleTab("subgraph");
              }
            }}
          >
            <Network size={16} />
            Subgraph Explorer
          </button>
        </div>

        {activeMiddleTab === "reader" ? (
          <>
            <div className="nlp-reader-box overflow-y-auto max-h-[380px]">
              {renderTokenizedText()}
            </div>
            {analysisResult && (
              <div className="nlp-quick-metrics">
                <div className="quick-metric">
                  <span className="metric-label">Sentences</span>
                  <span className="metric-val">{analysisResult.sentences.length}</span>
                </div>
                <div className="quick-metric">
                  <span className="metric-label">Clinical Entities</span>
                  <span className="metric-val">{analysisResult.entities.length}</span>
                </div>
              </div>
            )}
          </>
        ) : (
          <div className="nlp-reader-box flex flex-col justify-between h-full">
            {renderSubgraphGraph()}
          </div>
        )}
      </div>

      {/* RIGHT COLUMN: Annotation & Mapping Editor */}
      <div className="nlp-column glass-panel flex flex-col h-full overflow-hidden">
        <div className="panel-header">
          <Link2 className="panel-icon text-rose" />
          <h2>Clinical Tag Editor</h2>
        </div>

        <div className="entity-details-view flex-1 min-height-0">
          {selectedToken ? (
            <div className="entity-card animate-fade-in flex flex-col gap-3">
              <div className="entity-card-header-hud">
                <span className="entity-card-title">{selectedToken.text}</span>
                <span className="entity-card-subtitle">Offsets: {selectedToken.start}-{selectedToken.end}</span>
              </div>

              {/* Tag Editor Form */}
              <div className="entity-card-form">
                <div className="setting-control w-full">
                  <label className="section-label text-xs">Concept Class</label>
                  <select
                    className="custom-select w-full mt-1"
                    value={editCategory}
                    onChange={(e) => setEditCategory(e.target.value)}
                  >
                    <option value="Disease">Disease</option>
                    <option value="Diagnosis">Diagnosis</option>
                    <option value="Phenotype">Phenotype</option>
                    <option value="Body Parts">Body Parts</option>
                    <option value="Drugs">Drugs</option>
                    <option value="Chemicals">Chemicals</option>
                    <option value="Procedures">Procedures</option>
                    <option value="Labs">Labs</option>
                    <option value="Devices">Devices</option>
                  </select>
                </div>

                <div className="setting-control w-full">
                  <label className="section-label text-xs">Detail Class (UMLS Original)</label>
                  <input
                    type="text"
                    className="custom-input w-full mt-1 bg-white/5 border-white/10 text-muted cursor-not-allowed opacity-80"
                    value={editDetailClass}
                    readOnly
                    disabled
                  />
                </div>

                <div className="setting-control w-full">
                  <label className="section-label text-xs">Canonical Term</label>
                  <input
                    type="text"
                    className="custom-input w-full mt-1"
                    value={editCanonicalName}
                    onChange={(e) => setEditCanonicalName(e.target.value)}
                  />
                </div>

                <div className="setting-control w-full">
                  <label className="section-label text-xs">UMLS CUI</label>
                  <input
                    type="text"
                    className="custom-input w-full mt-1"
                    value={editCui}
                    onChange={(e) => setEditCui(e.target.value)}
                  />
                </div>

                <div className="border-t border-white/5 pt-3">
                  <span className="text-xs text-muted font-semibold block mb-2">Vocab Mappings</span>
                  {visibleDbs.length > 0 ? (
                    <div className="vocab-grid">
                      {visibleDbs.map(db => (
                        <div key={db.key} className="vocab-field">
                          <label className="text-[10px] text-muted">{db.label}</label>
                          <input
                            type="text"
                            className="custom-input text-xs w-full"
                            value={getDbValue(db.key)}
                            onChange={(e) => setDbValue(db.key, e.target.value)}
                            placeholder={db.placeholder}
                          />
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-xs text-muted italic my-1">No vocab mappings found.</div>
                  )}

                  {hiddenDbs.length > 0 && (
                    <div className="mt-2 pt-1 border-t border-white/5">
                      <select
                        className="custom-select text-xs w-full cursor-pointer bg-cyan/10 border-cyan/20 text-cyan hover:bg-cyan/20"
                        value=""
                        onChange={(e) => {
                          const val = e.target.value;
                          if (val) {
                            setManuallyAddedDbs([...manuallyAddedDbs, val]);
                          }
                        }}
                      >
                        <option value="" disabled>+ Add ID from other databases</option>
                        {hiddenDbs.map(db => (
                          <option key={db.key} value={db.key}>{db.label}</option>
                        ))}
                      </select>
                    </div>
                  )}
                </div>
              </div>

              {/* Action Buttons */}
              <div className="flex flex-col gap-3 mt-auto pt-4 border-t border-white/10">
                {selectedEntity ? (
                  <>
                    <div className="flex gap-2">
                      <button
                        className="search-btn flex-1 flex justify-center items-center gap-1 text-white font-semibold"
                        onClick={handleUpdateTag}
                      >
                        <Save size={14} />
                        Update Tag
                      </button>
                      <button
                        className="btn-outline-rose flex-1 flex justify-center items-center gap-1"
                        onClick={handleRemoveTag}
                      >
                        <Trash2 size={14} />
                        Remove Tag
                      </button>
                    </div>

                    <button
                      className="btn-outline-cyan w-full flex justify-center items-center gap-1 mt-1"
                      onClick={() => handleMapSubgraph(selectedEntity.cui, selectedEntity.codes)}
                    >
                      <Network size={14} />
                      View Neo4j Subgraph
                    </button>
                  </>
                ) : (
                  <button
                    className="search-btn w-full flex justify-center items-center gap-1 bg-teal/80 text-black font-semibold"
                    onClick={handleCreateTag}
                  >
                    <Plus size={14} />
                    Tag as Clinical Concept
                  </button>
                )}
              </div>
            </div>
          ) : (
            <div className="no-selection-card py-5">
              <Activity className="pulse-icon text-muted" size={40} />
              <p>Click on any word in the reader to edit annotations or map its Neo4j subgraph.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default NlpSandbox;
