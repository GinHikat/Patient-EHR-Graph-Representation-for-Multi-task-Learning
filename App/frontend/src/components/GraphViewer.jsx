import React, { useState, useEffect, useCallback, useRef } from "react";
import ForceGraph2D from "react-force-graph-2d";
import axios from "axios";
import { X } from "lucide-react";

const API_BASE_URL = "http://localhost:8000/api";

// Generate colors based on string
const getColor = (str) => {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = str.charCodeAt(i) + ((hash << 5) - hash);
  }
  const c = (hash & 0x00ffffff).toString(16).toUpperCase();
  return "#" + "00000".substring(0, 6 - c.length) + c;
};

const GraphViewer = () => {
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });
  const [nodeTypes, setNodeTypes] = useState(["All"]);
  const [selectedType, setSelectedType] = useState("All");
  const [nodeLimit, setNodeLimit] = useState(100);
  const [loading, setLoading] = useState(false);
  const [selectedNode, setSelectedNode] = useState(null);
  const [highlightNodes, setHighlightNodes] = useState(new Set());
  const [highlightLinks, setHighlightLinks] = useState(new Set());
  const [hoverNode, setHoverNode] = useState(null);

  const fgRef = useRef();

  // Load node types on mount
  useEffect(() => {
    const fetchNodeTypes = async () => {
      try {
        const response = await axios.get(`${API_BASE_URL}/node_types`);
        setNodeTypes(["All", ...response.data.node_types]);
      } catch (err) {
        console.error("Failed to fetch node types", err);
      }
    };
    fetchNodeTypes();
  }, []);

  // Fetch graph data when filters change
  useEffect(() => {
    const fetchGraph = async () => {
      setLoading(true);
      try {
        const response = await axios.get(`${API_BASE_URL}/graph`, {
          params: {
            limit: nodeLimit,
            node_type: selectedType === "All" ? "" : selectedType,
          },
        });
        setGraphData(response.data);
      } catch (err) {
        console.error("Failed to fetch graph data", err);
      } finally {
        setLoading(false);
      }
    };

    // Add a slight debounce
    const timer = setTimeout(() => {
      fetchGraph();
    }, 300);

    return () => clearTimeout(timer);
  }, [nodeLimit, selectedType]);

  const handleNodeClick = useCallback(
    (node) => {
      setSelectedNode(node);

      // Highlight connected nodes and links
      const connectedNodes = new Set();
      const connectedLinks = new Set();

      connectedNodes.add(node);
      graphData.links.forEach((link) => {
        if (link.source.id === node.id || link.source === node.id) {
          connectedLinks.add(link);
          connectedNodes.add(link.target);
        }
        if (link.target.id === node.id || link.target === node.id) {
          connectedLinks.add(link);
          connectedNodes.add(link.source);
        }
      });

      setHighlightNodes(connectedNodes);
      setHighlightLinks(connectedLinks);

      // Aim at node
      const distance = 40;
      const distRatio = 1 + distance / Math.hypot(node.x, node.y, node.z || 0);
      if (fgRef.current) {
        fgRef.current.centerAt(node.x, node.y, 1000);
        fgRef.current.zoom(5, 2000);
      }
    },
    [graphData],
  );

  const handleNodeHover = useCallback(
    (node) => {
      setHoverNode(node || null);

      // Remove if clicked node is present, logic could be merged
      if (!selectedNode) {
        const connectedNodes = new Set();
        const connectedLinks = new Set();

        if (node) {
          connectedNodes.add(node);
          graphData.links.forEach((link) => {
            if (link.source.id === node.id) {
              connectedLinks.add(link);
              connectedNodes.add(link.target);
            }
            if (link.target.id === node.id) {
              connectedLinks.add(link);
              connectedNodes.add(link.source);
            }
          });
        }

        setHighlightNodes(connectedNodes);
        setHighlightLinks(connectedLinks);
      }
    },
    [graphData, selectedNode],
  );

  const resetSelection = () => {
    setSelectedNode(null);
    setHighlightNodes(new Set());
    setHighlightLinks(new Set());
    if (fgRef.current) {
      fgRef.current.zoomToFit(400);
    }
  };

  return (
    <div className="graph-container">
      {loading && (
        <div className="loading-overlay">
          <div className="spinner"></div>
        </div>
      )}

      <div className="controls-panel glass-panel">
        <div className="control-group">
          <label>Filter by Node Type</label>
          <select
            className="custom-select"
            value={selectedType}
            onChange={(e) => setSelectedType(e.target.value)}
          >
            {nodeTypes.map((type) => (
              <option key={type} value={type}>
                {type}
              </option>
            ))}
          </select>
        </div>

        <div className="control-group">
          <label>
            <span>Node Limit</span>
            <span style={{ color: "var(--accent-color)" }}>{nodeLimit}</span>
          </label>
          <input
            type="range"
            className="custom-slider"
            min="10"
            max="2000"
            step="50"
            value={nodeLimit}
            onChange={(e) => setNodeLimit(parseInt(e.target.value))}
          />
        </div>

        {/* Dynamic Legend based on visible nodes */}
        <div className="legend">
          <label
            style={{ fontSize: "0.75rem", marginBottom: "8px", opacity: 0.8 }}
          >
            All Types
          </label>
          {nodeTypes
            .filter((type) => type !== "All" && type !== "Test")
            .map((label) => (
              <div key={label} className="legend-item">
                <div
                  className="legend-color"
                  style={{ backgroundColor: getColor(label) }}
                ></div>
                <span>{label}</span>
              </div>
            ))}
        </div>
      </div>

      <ForceGraph2D
        ref={fgRef}
        graphData={graphData}
        nodeLabel={(node) => `
          <div style="background: rgba(0,0,0,0.8); padding: 8px; border-radius: 4px; font-family: Inter, sans-serif;">
             <div style="color: #60a5fa; font-size: 10px; margin-bottom: 2px;">${node.labels?.filter((l) => l !== "Test")[0] || "Node"}</div>
             <div style="font-weight: bold; color: white;">${node.properties?.name || node.properties?.title || node.id}</div>
          </div>
        `}
        nodeColor={(node) => {
          if (highlightNodes.size > 0 && !highlightNodes.has(node)) {
            return "rgba(255, 255, 255, 0.1)";
          }
          return getColor(node.labels?.[0] || "Unknown");
        }}
        nodeRelSize={6}
        linkWidth={(link) => (highlightLinks.has(link) ? 3 : 1)}
        linkColor={(link) =>
          highlightLinks.has(link)
            ? "rgba(59, 130, 246, 0.8)"
            : "rgba(255, 255, 255, 0.2)"
        }
        linkDirectionalParticles={(link) => (highlightLinks.has(link) ? 4 : 0)}
        linkDirectionalParticleWidth={4}
        linkLabel="type"
        linkCanvasObjectMode={() => "after"}
        linkCanvasObject={(link, ctx) => {
          const start = link.source;
          const end = link.target;
          if (typeof start !== "object" || typeof end !== "object") return;
          const textPos = Object.assign(
            ...["x", "y"].map((c) => ({
              [c]: start[c] + (end[c] - start[c]) / 2,
            })),
          );
          const relLink = { x: end.x - start.x, y: end.y - start.y };
          let textAngle = Math.atan2(relLink.y, relLink.x);
          if (textAngle > Math.PI / 2) textAngle = -(Math.PI - textAngle);
          if (textAngle < -Math.PI / 2) textAngle = -(-Math.PI - textAngle);
          ctx.font = `4px Inter, sans-serif`;
          ctx.save();
          ctx.translate(textPos.x, textPos.y);
          ctx.rotate(textAngle);
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillStyle = highlightLinks.has(link)
            ? "rgba(96, 165, 250, 1)"
            : "rgba(255, 255, 255, 0.6)";
          ctx.fillText(link.type, 0, -2);
          ctx.restore();
        }}
        onNodeClick={handleNodeClick}
        onNodeHover={handleNodeHover}
        onBackgroundClick={resetSelection}
        cooldownTicks={100}
        backgroundColor="rgba(0,0,0,0)" // Transparent to see gradient background
      />

      {selectedNode && (
        <div className="node-info-panel glass-panel">
          <div className="node-info-header">
            <div>
              <h2>
                {selectedNode.properties?.name ||
                  selectedNode.properties?.title ||
                  "Node Info"}
              </h2>
              <div className="node-labels">
                {selectedNode.labels
                  ?.filter((l) => l !== "Test")
                  .map((label, idx) => (
                    <span
                      key={idx}
                      className="node-label"
                      style={{ borderLeft: `3px solid ${getColor(label)}` }}
                    >
                      {label}
                    </span>
                  ))}
                {selectedNode.labels?.filter((l) => l !== "Test").length ===
                  0 && (
                  <span
                    className="node-label"
                    style={{ borderLeft: "3px solid #ccc" }}
                  >
                    Node
                  </span>
                )}
              </div>
            </div>
            <button className="close-btn" onClick={resetSelection}>
              <X size={20} />
            </button>
          </div>

          <div className="property-list">
            {Object.entries(selectedNode.properties || {}).map(
              ([key, value]) => {
                if (key === "name" || key === "title") return null; // Already shown
                return (
                  <div key={key} className="property-item">
                    <span className="prop-key">{key}</span>
                    <span className="prop-value">
                      {typeof value === "object"
                        ? JSON.stringify(value)
                        : String(value)}
                    </span>
                  </div>
                );
              },
            )}
            {Object.keys(selectedNode.properties || {}).length === 0 && (
              <span className="prop-value" style={{ opacity: 0.5 }}>
                No additional properties
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default GraphViewer;
