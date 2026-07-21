document.addEventListener("DOMContentLoaded", () => {
  // Stage Explorer Data (4 Core MIMIC-IV Stages)
  const stageData = {
    1: {
      badge: "STAGE 01",
      title: "MIMIC-IV Preprocessing & Ontological Standardizing",
      body: "Parses raw relational EHR tables from MIMIC-IV. Applies medical filtering, handles missing clinical values, and aligns ICD diagnosis codes with external medical ontologies."
    },
    2: {
      badge: "STAGE 02",
      title: "Neo4j Heterogeneous Knowledge Graph Construction",
      body: "Constructs Neo4j Graph DB schema connecting patient admissions, transfers, lab events, and prescriptions linked to external disease-disease and drug-drug interaction nodes."
    },
    3: {
      badge: "STAGE 03",
      title: "GAT Representation Learning & Semantic Embeddings",
      body: "Leverages Graph Attention Networks (GAT) to compute rich semantic entity embeddings across patient admission nodes before temporal sequence alignment."
    },
    4: {
      badge: "STAGE 04",
      title: "Downstream Multi-Task Temporal Sequence Modeling",
      body: "Aligns event vectors into dense patient timeline sequences (patient_timelines.pt) and trains deep RNN/Transformer models for mortality, readmission, and prescription prediction."
    }
  };

  const stepCards = document.querySelectorAll(".p-step-card");
  const stageBadge = document.getElementById("stage-badge");
  const stageTitle = document.getElementById("stage-title");
  const stageBody = document.getElementById("stage-body");

  stepCards.forEach(card => {
    card.addEventListener("click", () => {
      stepCards.forEach(c => c.classList.remove("active"));
      card.classList.add("active");

      const stageNum = card.getAttribute("data-stage");
      const data = stageData[stageNum];

      if (data) {
        stageBadge.textContent = data.badge;
        stageTitle.textContent = data.title;
        stageBody.textContent = data.body;
      }
    });
  });

  // Lightbox Zoom functionality for Graph Visualizations & Tables
  const lightbox = document.getElementById("lightbox");
  const lightboxImg = document.getElementById("lightbox-img");
  const lightboxClose = document.getElementById("lightbox-close");

  // Image Zooming
  const zoomableBoxes = document.querySelectorAll(".vis-img-box, .image-wrapper");
  zoomableBoxes.forEach(box => {
    box.addEventListener("click", (e) => {
      e.stopPropagation();
      const img = box.querySelector("img");
      if (img && lightbox && lightboxImg) {
        lightboxImg.style.display = "block";
        const oldTable = lightbox.querySelector(".lightbox-table-container");
        if (oldTable) oldTable.remove();

        lightboxImg.src = img.src;
        lightboxImg.alt = img.alt || "Expanded View";
        lightbox.style.display = "flex";
      }
    });
  });

  // Full Expanded SOTA Table HTML for Lightbox Zoom
  const fullSotaTableHTML = `
    <h3 style="font-size: 1.15rem; margin-bottom: 1rem; color: var(--text-main);">
      Comprehensive SOTA Benchmark & Citation Comparison Table
    </h3>
    <div class="table-responsive">
      <table class="sota-table">
        <thead>
          <tr>
            <th>Model & Citation</th>
            <th>Approach / Architecture</th>
            <th>Mortality AUROC</th>
            <th>Mortality AUPR</th>
            <th>Readmission AUROC</th>
            <th>Readmission AUPR</th>
            <th>Drug Rec AUROC</th>
            <th>Drug Rec AUPR</th>
            <th>Diag Prog AUROC</th>
            <th>Diag Prog AUPR</th>
          </tr>
        </thead>
        <tbody>
          <tr class="highlight-row">
            <td><strong>This Work (Patient EHR Graph - LSTM)</strong></td>
            <td>Temporal sequence learning on GAT-enriched EHR graph embeddings</td>
            <td><span class="badge-score">0.99</span></td>
            <td><span class="badge-score">0.79</span></td>
            <td>0.89</td>
            <td><span class="badge-score">0.79</span></td>
            <td>0.77</td>
            <td>0.50</td>
            <td>0.87</td>
            <td><span class="badge-score">0.20</span></td>
          </tr>
          <tr>
            <td>Daphne et al. (2025) <a href="https://doi.org/10.3390/diagnostics15060756" target="_blank" class="footer-link">[Diagnostics '25]</a></td>
            <td>GNN on Graph of similar patient Note embeddings</td>
            <td>0.93</td>
            <td>0.65</td>
            <td><span class="badge-score">0.95</span></td>
            <td>0.75</td>
            <td>—</td>
            <td>—</td>
            <td>—</td>
            <td>—</td>
          </tr>
          <tr>
            <td>Deng et al. (2022) <a href="https://doi.org/10.3389/fmed.2022.933037" target="_blank" class="footer-link">[Frontiers '22]</a></td>
            <td>RNN-variants on 24-step 24h sequence</td>
            <td>0.87</td>
            <td>—</td>
            <td>0.64</td>
            <td>—</td>
            <td>—</td>
            <td>—</td>
            <td>—</td>
            <td>—</td>
          </tr>
          <tr>
            <td>Chan et al. (2024) [MulT-EHR] <a href="https://arxiv.org" target="_blank" class="footer-link">[arXiv '24]</a></td>
            <td>GNN on EHR Graph representation</td>
            <td>0.71</td>
            <td>0.05</td>
            <td>0.69</td>
            <td>0.70</td>
            <td><span class="badge-score">0.98</span></td>
            <td>0.70</td>
            <td>—</td>
            <td>—</td>
          </tr>
          <tr>
            <td>Jiang et al. (2023) [GraphCare] <a href="https://arxiv.org/abs/2305.12788" target="_blank" class="footer-link">[ICLR '24]</a></td>
            <td>GNN on EHR graph and entity sequences</td>
            <td>0.73</td>
            <td>0.07</td>
            <td>0.82</td>
            <td>—</td>
            <td>0.95</td>
            <td><span class="badge-score">0.77</span></td>
            <td>—</td>
            <td>—</td>
          </tr>
          <tr>
            <td>Gupta et al. (2022) <a href="https://proceedings.mlr.press/v193/gupta22a.html" target="_blank" class="footer-link">[ML4H '22]</a></td>
            <td>ML on Tabular data & RNN on entity sequences</td>
            <td>0.87</td>
            <td>0.55</td>
            <td>0.77</td>
            <td>0.55</td>
            <td>—</td>
            <td>—</td>
            <td>—</td>
            <td>—</td>
          </tr>
          <tr>
            <td>Rohr et al. (2024) [COP-IV] <a href="https://aclanthology.org/2024.clinicalnlp-1.1" target="_blank" class="footer-link">[ClinicalNLP '24]</a></td>
            <td>BERT-based Classification from Discharge Notes</td>
            <td>—</td>
            <td>—</td>
            <td>—</td>
            <td>—</td>
            <td>—</td>
            <td>—</td>
            <td>0.84</td>
            <td>0.16</td>
          </tr>
          <tr>
            <td>Chen et al. (2025) [CrossRep] <a href="https://arxiv.org" target="_blank" class="footer-link">[arXiv '25]</a></td>
            <td>RNN-variants on entity sequences</td>
            <td>0.86</td>
            <td>0.33</td>
            <td>0.74</td>
            <td>0.27</td>
            <td>—</td>
            <td>—</td>
            <td>—</td>
            <td>—</td>
          </tr>
          <tr>
            <td>Bui et al. (2024) <a href="https://arxiv.org/abs/2401.15290" target="_blank" class="footer-link">[arXiv '24]</a></td>
            <td>XGBoost on Tabular data & Transformer on entity sequences</td>
            <td>0.87</td>
            <td>0.52</td>
            <td>—</td>
            <td>—</td>
            <td>—</td>
            <td>—</td>
            <td>—</td>
            <td>—</td>
          </tr>
          <tr>
            <td>Kumar et al. (2026) [UMGCA] <a href="#" class="footer-link">[UMGCA '26]</a></td>
            <td>Multi-Modal GNN on structured data, imaging & notes</td>
            <td>—</td>
            <td>—</td>
            <td>—</td>
            <td>—</td>
            <td>—</td>
            <td>—</td>
            <td><span class="badge-score">0.94</span></td>
            <td>—</td>
          </tr>
        </tbody>
      </table>
    </div>
  `;

  // Table Card Zooming into Lightbox Modal
  const sotaTableCard = document.getElementById("sota-table-card");
  if (sotaTableCard) {
    sotaTableCard.addEventListener("click", (e) => {
      if (lightbox) {
        lightboxImg.style.display = "none";
        
        let tableContainer = lightbox.querySelector(".lightbox-table-container");
        if (!tableContainer) {
          tableContainer = document.createElement("div");
          tableContainer.className = "lightbox-table-container";
          tableContainer.style.background = "var(--bg-card)";
          tableContainer.style.border = "1px solid var(--border-color)";
          tableContainer.style.borderRadius = "8px";
          tableContainer.style.padding = "1.25rem";
          tableContainer.style.maxWidth = "94vw";
          tableContainer.style.maxHeight = "85vh";
          tableContainer.style.overflow = "auto";
          tableContainer.style.boxShadow = "0 20px 50px rgba(0,0,0,0.8)";
          lightbox.appendChild(tableContainer);
        }
        
        tableContainer.innerHTML = fullSotaTableHTML;
        lightbox.style.display = "flex";
      }
    });
  }

  // Close lightbox on close button click
  if (lightboxClose) {
    lightboxClose.addEventListener("click", () => {
      lightbox.style.display = "none";
    });
  }

  // Close lightbox when clicking outside
  if (lightbox) {
    lightbox.addEventListener("click", (e) => {
      if (e.target === lightbox || e.target === lightboxClose) {
        lightbox.style.display = "none";
      }
    });
  }

  // Close lightbox on Escape key
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && lightbox && lightbox.style.display === "flex") {
      lightbox.style.display = "none";
    }
  });
});
