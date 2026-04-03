import React, { useState, useEffect, useMemo } from "react";
import GraphViewer from "./components/GraphViewer";
import axios from "axios";
import {
  Info,
  Database,
  ChevronDown,
  ChevronRight,
  Sun,
  Moon,
} from "lucide-react";
import "./index.css";

const HierarchyItem = ({ item, onTypeClick, depth = 0 }) => {
  const [isOpen, setIsOpen] = useState(depth < 2); // Auto-expand top levels

  if (item.count === 0 && !item.children) return null;

  return (
    <div className="hierarchy-item-container">
      <div
        className={`breakdown-item hierarchy-level-${depth}`}
        onClick={() =>
          item.children ? setIsOpen(!isOpen) : onTypeClick(item.name)
        }
      >
        <div className="hierarchy-label-group">
          {item.children && (
            <span className="hierarchy-toggle">
              {isOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
            </span>
          )}
          <span className="breakdown-type">{item.name}</span>
        </div>
        <span className="breakdown-count">{item.count.toLocaleString()}</span>
      </div>
      {isOpen && item.children && (
        <div className="hierarchy-children">
          {item.children.map((child, idx) => (
            <HierarchyItem
              key={`${child.name}-${idx}`}
              item={child}
              onTypeClick={onTypeClick}
              depth={depth + 1}
            />
          ))}
        </div>
      )}
    </div>
  );
};

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
  const [theme, setTheme] = useState(localStorage.getItem("theme") || "dark");

  // Handle theme persistence
  useEffect(() => {
    document.body.className = theme === "light" ? "light-theme" : "";
    localStorage.setItem("theme", theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme(theme === "dark" ? "light" : "dark");
  };

  // Automatically detect hierarchy based on label frequency and mutual exclusion
  const detectedHierarchy = useMemo(() => {
    if (!dbStats.node_breakdown || dbStats.node_breakdown.length === 0)
      return [];

    // Prioritize structural labels to ensure they stay at the top of the hierarchy
    const PRIORITY_LABELS = [
      "Test",
      "External",
      "MIMIC",
      "Admission",
      "Patient",
      "ICU",
      "Disease",
      "Drug",
      "Diagnosis",
      "Procedure",
      "Stay",
      "Lab",
      "Item",
    ];

    // 1. Calculate global frequency for each label across all node sets
    const labelCounts = {};
    dbStats.node_breakdown.forEach((entry) => {
      entry.labels?.forEach((label) => {
        labelCounts[label] = (labelCounts[label] || 0) + entry.count;
      });
    });

    // 2. For each entry, sort its labels by priority and then frequency
    const sortedEntries = dbStats.node_breakdown.map((entry) => ({
      ...entry,
      sortedLabels: [...(entry.labels || [])].sort((a, b) => {
        const aIdx = PRIORITY_LABELS.indexOf(a);
        const bIdx = PRIORITY_LABELS.indexOf(b);
        if (aIdx !== -1 && bIdx !== -1) return aIdx - bIdx;
        if (aIdx !== -1) return -1;
        if (bIdx !== -1) return 1;
        return labelCounts[b] - labelCounts[a];
      }),
    }));

    // 3. Build a Trie structure with accumulated counts
    const root = { children: {}, count: 0 };
    sortedEntries.forEach((entry) => {
      let current = root;
      current.count += entry.count;
      entry.sortedLabels.forEach((label) => {
        if (!current.children[label]) {
          current.children[label] = { name: label, children: {}, count: 0 };
        }
        current = current.children[label];
        current.count += entry.count; // Crucial for summation
      });
    });

    // 4. Convert Trie back to the expected recursive array structure
    const convertNode = (node) => {
      const children = Object.values(node.children)
        .map(convertNode)
        .sort((a, b) => b.count - a.count); // Sort siblings by count
      return {
        name: node.name,
        count: node.count,
        children: children.length > 0 ? children : null,
      };
    };

    return Object.values(root.children).map(convertNode);
  }, [dbStats.node_breakdown]);

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

          <button
            className={`theme-toggle-btn ${theme === "light" ? "active" : ""}`}
            onClick={toggleTheme}
            title={
              theme === "light" ? "Switch to Dark Mode" : "Switch to Light Mode"
            }
          >
            {theme === "light" ? <Moon size={18} /> : <Sun size={18} />}
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
              <h3>Node Types Hierarchy</h3>
              <div className="breakdown-list hierarchy-list">
                {detectedHierarchy.map((group) => (
                  <HierarchyItem
                    key={group.name}
                    item={group}
                    nodeBreakdown={dbStats.node_breakdown}
                    onTypeClick={(type) => handleTypeClick(type, "node")}
                  />
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
            theme={theme}
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
