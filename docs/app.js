document.addEventListener("DOMContentLoaded", () => {
  // Stage Explorer Data
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
});
