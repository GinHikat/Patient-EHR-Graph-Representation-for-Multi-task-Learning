import React, { useState, useEffect } from "react";
import GraphViewer from "./components/GraphViewer";
import axios from "axios";
import { Info, Database } from "lucide-react";
import "./index.css";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000/api";

function App() {
  const [activeTab, setActiveTab] = useState("graph");
  const [showStats, setShowStats] = useState(false);
  const [dbStats, setDbStats] = useState({
    total_nodes: 0,
    total_edges: 0,
    node_breakdown: [],
    edge_breakdown: [],
  });
  const [externalFilter, setExternalFilter] = useState(null);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const response = await axios.get(`${API_BASE_URL}/stats`);
        setDbStats(response.data);
      } catch (err) {
        console.error("Failed to fetch DB stats", err);
      }
    };
    fetchStats();
  }, []);

  const handleTypeClick = (type, category = "node") => {
    setExternalFilter({ type, category });
    setActiveTab("graph");
    setShowStats(false);
  };

  return (
    <div className="app-container">
      <header className="app-header glass-panel">
        <div className="header-left">
          <div className="logo">
            <h1>Patient EHR Explorer</h1>
          </div>
          <button
            className={`stats-toggle-btn ${showStats ? "active" : ""}`}
            onClick={() => setShowStats(!showStats)}
            title="Database Statistics"
          >
            <Database size={18} />
          </button>
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

      {showStats && (
        <div className="database-stats-pane glass-panel animate-fade-in">
          <div className="stats-summary">
            <div className="header-stat">
              <span className="stat-label">Total Nodes</span>
              <span className="stat-value">
                {dbStats.total_nodes.toLocaleString()}
              </span>
            </div>
            <div className="header-stat">
              <span className="stat-label">Total Edges</span>
              <span className="stat-value">
                {dbStats.total_edges.toLocaleString()}
              </span>
            </div>
          </div>

          <div className="stats-breakdown">
            <div className="breakdown-section">
              <h3>Node Types</h3>
              <div className="breakdown-list">
                {dbStats.node_breakdown.map((item) => (
                  <div
                    key={item.type}
                    className="breakdown-item"
                    onClick={() => handleTypeClick(item.type, "node")}
                  >
                    <span className="breakdown-type">{item.type}</span>
                    <span className="breakdown-count">
                      {item.count.toLocaleString()}
                    </span>
                  </div>
                ))}
              </div>
            </div>
            <div className="breakdown-section">
              <h3>Relationship Types</h3>
              <div className="breakdown-list">
                {dbStats.edge_breakdown.map((item) => (
                  <div
                    key={item.type}
                    className="breakdown-item"
                    onClick={() => handleTypeClick(item.type, "edge")}
                  >
                    <span className="breakdown-type">{item.type}</span>
                    <span className="breakdown-count">
                      {item.count.toLocaleString()}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      <main className="app-content">
        {activeTab === "graph" ? (
          <GraphViewer
            externalFilter={externalFilter}
            onFilterUsed={() => setExternalFilter(null)}
          />
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
