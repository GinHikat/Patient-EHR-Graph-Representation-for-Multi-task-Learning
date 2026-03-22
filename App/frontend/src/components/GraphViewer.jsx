import React, { useState, useEffect, useCallback, useRef } from "react";
import ForceGraph2D from "react-force-graph-2d";
import axios from "axios";
import { X } from "lucide-react";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000/api";

// Predefined high-contrast palette for common EHR and CTD node types
const TYPE_COLORS = {
  Patient: "#3b82f6", // Blue
  Condition: "#f43f5e", // Rose/Red (more vibrant)
  Disease: "#f43f5e", // Rose/Red
  Drug: "#fbbf24", // Amber (brighter)
  Observation: "#10b981", // Green
  Procedure: "#8b5cf6", // Violet
  Encounter: "#ec4899", // Pink
  LabResult: "#06b6d4", // Cyan
  Medication: "#f97316", // Orange
  Allergy: "#ef4444", // Red
  Immunization: "#6366f1", // Indigo
  Gene: "#fb7185", // Rose
  Pathway: "#2dd4bf", // Teal
  Chemical: "#facc15", // Yellow
};

// Generate high-contrast colors based on string
const getColor = (str) => {
  if (TYPE_COLORS[str]) return TYPE_COLORS[str];

  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = str.charCodeAt(i) + ((hash << 5) - hash);
  }

  // Use HSL for consistent vibrancy and contrast against dark background
  // Hue: 0-360, Saturation: 80% (vibrant), Lightness: 60% (bright but not washed out)
  const h = Math.abs(hash) % 360;
  return `hsl(${h}, 85%, 65%)`;
};

const GraphViewer = () => {
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });
  const [nodeTypes, setNodeTypes] = useState(["All"]);
  const [selectedTypes, setSelectedTypes] = useState(["All"]);
  const [nodeLimit, setNodeLimit] = useState(100);
  const [loading, setLoading] = useState(false);
  const [selectedNode, setSelectedNode] = useState(null);
  const [highlightNodes, setHighlightNodes] = useState(new Set());
  const [highlightLinks, setHighlightLinks] = useState(new Set());
  const [hoverNode, setHoverNode] = useState(null);
  const [searchId, setSearchId] = useState("");
  const [searchLoading, setSearchLoading] = useState(false);

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
        const params = new URLSearchParams();
        params.append("limit", nodeLimit);
        if (!selectedTypes.includes("All")) {
          selectedTypes.forEach((type) => params.append("node_type", type));
        }
        const response = await axios.get(`${API_BASE_URL}/graph`, { params });
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
  }, [nodeLimit, selectedTypes]);

  const handleSearchNode = async (e) => {
    e.preventDefault();
    if (!searchId.trim()) return;

    setSearchLoading(true);
    try {
      const response = await axios.get(
        `${API_BASE_URL}/node/${searchId.trim()}`,
      );
      if (response.data && response.data.nodes.length > 0) {
        const newData = response.data;
        setGraphData(newData);

        const targetNodeId = searchId.trim().toLowerCase();
        const firstNode =
          newData.nodes.find(
            (n) =>
              n.id.toLowerCase() === targetNodeId ||
              n.properties.id?.toString().toLowerCase() === targetNodeId ||
              n.properties.title?.toLowerCase() === targetNodeId ||
              n.properties.Title?.toLowerCase() === targetNodeId ||
              n.properties.name?.toLowerCase() === targetNodeId,
          ) || newData.nodes[0];

        setSelectedNode(firstNode);

        // Extract highlights from new data
        const connectedNodes = new Set();
        const connectedLinks = new Set();
        connectedNodes.add(firstNode);
        newData.links.forEach((link) => {
          if (
            link.source === firstNode.id ||
            (link.source && link.source.id === firstNode.id)
          ) {
            connectedLinks.add(link);
            const target = newData.nodes.find(
              (n) => n.id === (link.target.id || link.target),
            );
            if (target) connectedNodes.add(target);
          }
          if (
            link.target === firstNode.id ||
            (link.target && link.target.id === firstNode.id)
          ) {
            connectedLinks.add(link);
            const source = newData.nodes.find(
              (n) => n.id === (link.source.id || link.source),
            );
            if (source) connectedNodes.add(source);
          }
        });
        setHighlightNodes(connectedNodes);
        setHighlightLinks(connectedLinks);

        // Centering will happen after simulation starts
        setTimeout(() => {
          if (fgRef.current) {
            fgRef.current.zoomToFit(1000, 100);
          }
        }, 500);
      }
    } catch (err) {
      console.error("Failed to find node", err);
      alert("Node not found");
    } finally {
      setSearchLoading(false);
    }
  };

  const toggleType = (type) => {
    if (type === "All") {
      setSelectedTypes(["All"]);
    } else {
      setSelectedTypes((prev) => {
        const newTypes = prev.filter((t) => t !== "All");
        if (newTypes.includes(type)) {
          const updated = newTypes.filter((t) => t !== type);
          return updated.length === 0 ? ["All"] : updated;
        } else {
          return [...newTypes, type];
        }
      });
    }
  };

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
          <div className="search-pane">
            <label>Search by ID / Name</label>
            <form onSubmit={handleSearchNode} className="search-form">
              <input
                type="text"
                className="custom-input"
                placeholder="Enter ID/Name..."
                value={searchId}
                onChange={(e) => setSearchId(e.target.value)}
              />
              <button
                type="submit"
                className="search-btn"
                disabled={searchLoading}
              >
                {searchLoading ? "..." : "Find"}
              </button>
            </form>
          </div>
        </div>

        <div className="control-group">
          <label>Filter by Node Type</label>
          <div className="type-badges">
            {nodeTypes.map((type) => (
              <button
                key={type}
                className={`type-badge ${selectedTypes.includes(type) ? "active" : ""}`}
                onClick={() => toggleType(type)}
              >
                {type}
              </button>
            ))}
          </div>
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
          <div style="background: rgba(0,0,0,0.9); padding: 12px; border-radius: 8px; font-family: Inter, sans-serif; box-shadow: 0 4px 20px rgba(0,0,0,0.5); min-width: 140px; border: 1px solid rgba(255,255,255,0.1);">
             <div style="color: #60a5fa; font-size: 12px; font-weight: 600; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.5px; opacity: 0.8;">${node.labels?.filter((l) => l !== "Test")[0] || "Node"}</div>
             <div style="font-weight: 700; color: white; font-size: 16px; line-height: 1.4;">${node.properties?.name || node.properties?.title || node.properties?.Title || node.id}</div>
          </div>
        `}
        nodeRelSize={7}
        nodeCanvasObject={(node, ctx, globalScale) => {
          const label = node.labels?.[0] || "Unknown";
          const isHighlighted =
            highlightNodes.size === 0 || highlightNodes.has(node);
          const isHovered = hoverNode === node;
          const color = getColor(label);
          const fontSize = 12 / globalScale;
          const radius = 6;

          // Dim nodes that are not highlighted
          ctx.globalAlpha = isHighlighted ? 1 : 0.15;

          // Draw node shadow/glow if highlighted or hovered
          if (isHighlighted || isHovered) {
            ctx.beginPath();
            ctx.arc(node.x, node.y, radius + 1, 0, 2 * Math.PI, false);
            ctx.fillStyle = isHovered
              ? "rgba(255, 255, 255, 0.3)"
              : "rgba(0, 0, 0, 0.3)";
            ctx.fill();
          }

          // Draw main circle
          ctx.beginPath();
          ctx.arc(node.x, node.y, radius, 0, 2 * Math.PI, false);
          ctx.fillStyle = color;
          ctx.fill();

          // Draw white border for high contrast
          ctx.strokeStyle = "white";
          ctx.lineWidth = (isHovered ? 2 : 1) / globalScale;
          ctx.stroke();

          // Reset alpha for following operations
          ctx.globalAlpha = 1;
        }}
        nodePointerAreaPaint={(node, color, ctx) => {
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(node.x, node.y, 7, 0, 2 * Math.PI, false);
          ctx.fill();
        }}
        linkWidth={(link) => (highlightLinks.has(link) ? 3 : 1)}
        linkColor={(link) =>
          highlightLinks.has(link)
            ? "rgba(59, 130, 246, 0.8)"
            : "rgba(255, 255, 255, 0.2)"
        }
        linkDirectionalParticles={(link) => (highlightLinks.has(link) ? 4 : 0)}
        linkDirectionalParticleWidth={4}
        linkLabel={(link) => `
          <div style="background: rgba(0,0,0,0.9); padding: 8px 12px; border-radius: 4px; font-family: Inter, sans-serif; color: white; border: 1px solid rgba(255,255,255,0.1);">
            <div style="color: #60a5fa; font-size: 10px; font-weight: 600; text-transform: uppercase;">Relationship</div>
            <div style="font-weight: 700; font-size: 14px;">${link.type}</div>
          </div>
        `}
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
          ctx.font = `6px Inter, sans-serif`;
          ctx.save();
          ctx.translate(textPos.x, textPos.y);
          ctx.rotate(textAngle);
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillStyle = highlightLinks.has(link)
            ? "rgba(96, 165, 250, 1)"
            : "rgba(255, 255, 255, 0.4)";
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
                  selectedNode.properties?.Title ||
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
