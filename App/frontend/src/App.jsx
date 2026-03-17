import React, { useState } from "react";
import GraphViewer from "./components/GraphViewer";
import "./index.css";

function App() {
  const [activeTab, setActiveTab] = useState("graph");

  return (
    <div className="app-container">
      <header className="app-header glass-panel">
        <div className="logo">
          <h1>Neo4j Explorer</h1>
        </div>
        <nav className="tabs">
          <button
            className={`custom-button ${activeTab === "graph" ? "active" : ""}`}
            onClick={() => setActiveTab("graph")}
          >
            Graph Visualization
          </button>
          <button
            className={`custom-button ${activeTab === "data" ? "active" : ""}`}
            onClick={() => setActiveTab("data")}
          >
            Settings
          </button>
        </nav>
      </header>

      <main className="app-content">
        {activeTab === "graph" ? (
          <GraphViewer />
        ) : (
          <div className="coming-soon glass-panel">
            <h2>Settings View</h2>
            <p>Configure advanced connection settings here.</p>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
