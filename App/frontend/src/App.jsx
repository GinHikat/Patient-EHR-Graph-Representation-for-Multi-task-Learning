import React, { useState, useEffect } from "react";
import GraphViewer from "./components/GraphViewer";
import axios from "axios";
import { Info, Database } from "lucide-react";
import "./index.css";

const API_BASE_URL = "http://localhost:8000/api";

function App() {
  const [activeTab, setActiveTab] = useState("graph");
  const [showStats, setShowStats] = useState(false);
  const [dbStats, setDbStats] = useState({ total_nodes: 0, total_edges: 0 });

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

  return (
    <div className="app-container">
      <header className="app-header glass-panel">
        <div className="header-left">
          <div className="logo">
            <h1>Neo4j Explorer</h1>
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
      )}

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
