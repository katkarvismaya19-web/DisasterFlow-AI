import { useEffect, useMemo, useState } from "react";
import "./App.css";
import RiskMap from "./components/RiskMap";

const API = "http://127.0.0.1:8010";
const REFRESH_INTERVAL = 30000;

function formatNumber(value, decimals = 1) {
  const number = Number(value);

  if (!Number.isFinite(number)) {
    return "0";
  }

  return number.toFixed(decimals);
}

function probabilityPercent(value) {
  const number = Number(value);

  if (!Number.isFinite(number)) {
    return "0.000%";
  }

  return `${(number * 100).toFixed(3)}%`;
}

function getForecast(node) {
  const value =
    node.forecast_flood_probability ??
    node.stgnn_forecast ??
    node.forecast_probability ??
    node.forecast_probability_value ??
    null;

  if (value === null || value === undefined) {
    return null;
  }

  const number = Number(value);

  if (!Number.isFinite(number)) {
    return null;
  }

  return number;
}

function getResourceStatus(node) {
  const required = Number(node.resources_required) || 0;
  const allocated = Number(node.allocated_resources) || 0;

  const difference = allocated - required;

  if (difference > 0) {
    return {
      type: "surplus",
      label: `SURPLUS ${difference}`,
      shortLabel: "SURPLUS",
      value: difference,
    };
  }

  if (difference < 0) {
    return {
      type: "shortage",
      label: `SHORTAGE ${Math.abs(difference)}`,
      shortLabel: "SHORTAGE",
      value: Math.abs(difference),
    };
  }

  return {
    type: "covered",
    label: "FULLY COVERED",
    shortLabel: "COVERED",
    value: 0,
  };
}

function getRiskClass(risk) {
  return String(risk || "LOW").toLowerCase();
}

function getPriorityClass(priority) {
  return String(priority || "LOW").toLowerCase();
}

function getAction(node) {
  if (node.recommended_action) {
    return node.recommended_action;
  }

  const risk = String(node.risk_level || "").toUpperCase();
  const priority = String(node.emergency_priority || "").toUpperCase();

  if (risk === "CRITICAL" || priority === "CRITICAL") {
    return "DEPLOY EMERGENCY RESOURCES — immediate response readiness";
  }

  if (risk === "HIGH" || priority === "HIGH") {
    return "ELEVATED MONITORING — prepare emergency response";
  }

  if (risk === "MEDIUM" || priority === "MEDIUM") {
    return "ACTIVE MONITORING — maintain response readiness";
  }

  return "ROUTINE MONITORING — maintain disaster readiness";
}

function App() {
  const [prediction, setPrediction] = useState(null);
  const [priorities, setPriorities] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [lastUpdated, setLastUpdated] = useState(null);

  async function loadData() {
    try {
      setError("");

      const [predictionResponse, priorityResponse] =
        await Promise.all([
          fetch(`${API}/prediction`),
          fetch(`${API}/priorities`),
        ]);

      if (!predictionResponse.ok) {
        throw new Error(
          `Prediction API failed: ${predictionResponse.status}`
        );
      }

      if (!priorityResponse.ok) {
        throw new Error(
          `Priorities API failed: ${priorityResponse.status}`
        );
      }

      const predictionData = await predictionResponse.json();
      const priorityData = await priorityResponse.json();

      setPrediction(predictionData);

      const sortedPriorities = Array.isArray(priorityData)
        ? [...priorityData].sort(
            (a, b) =>
              Number(a.emergency_rank || 999999) -
              Number(b.emergency_rank || 999999)
          )
        : [];

      setPriorities(sortedPriorities);
      setLastUpdated(new Date());
    } catch (err) {
      console.error("DisasterFlow API error:", err);
      setError(err.message || "Unable to connect to DisasterFlow AI.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadData();

    const interval = setInterval(loadData, REFRESH_INTERVAL);

    return () => clearInterval(interval);
  }, []);

  const totalNodes = Number(prediction?.total_nodes) || 0;

  const criticalCount = Number(
    prediction?.risk_distribution?.CRITICAL || 0
  );

  const highCount = Number(
    prediction?.risk_distribution?.HIGH || 0
  );

  const mediumCount = Number(
    prediction?.risk_distribution?.MEDIUM || 0
  );

  const lowCount = Number(
    prediction?.risk_distribution?.LOW || 0
  );

  const shortageLocations = useMemo(() => {
    return priorities.filter((node) => {
      const required = Number(node.resources_required) || 0;
      const allocated = Number(node.allocated_resources) || 0;

      return allocated < required;
    }).length;
  }, [priorities]);

  const topPriorities = priorities.slice(0, 10);

  if (loading) {
    return (
      <div className="loading-screen">
        <div className="loading-logo">DF</div>

        <h1>DisasterFlow AI</h1>

        <p>
          Connecting to AI disaster intelligence engine...
        </p>

        <div className="loading-bar">
          <div className="loading-progress"></div>
        </div>
      </div>
    );
  }

  if (error || !prediction) {
    return (
      <div className="error-screen">
        <div className="error-card">
          <div className="error-icon">⚠️</div>

          <h1>Backend Connection Error</h1>

          <p>
            {error || "Unable to load DisasterFlow AI data."}
          </p>

          <div className="error-command">
            FastAPI should be running on
            <strong> http://127.0.0.1:8010</strong>
          </div>

          <button onClick={loadData}>
            Retry Connection
          </button>
        </div>
      </div>
    );
  }

  const coverageRequired =
    Number(prediction.total_resources_required) || 0;

  const coverageAllocated =
    Number(prediction.total_resources_allocated) || 0;

  const coveragePercent =
    coverageRequired > 0
      ? Math.min(
          100,
          (coverageAllocated / coverageRequired) * 100
        )
      : 100;

  return (
    <div className="app">

      {/* =====================================================
          HEADER
      ===================================================== */}

      <header className="hero-header">

        <div className="hero-inner">

          <div className="brand-row">

            <div className="brand-logo">
              DF
            </div>

            <div className="brand-text">
              <div className="brand-name">
                DisasterFlow AI
              </div>

              <div className="brand-subtitle">
                AI-Powered Spatiotemporal Disaster Intelligence
              </div>
            </div>

          </div>

          <div className="header-right">

            <div className="location-pill">
              <span>📍</span>
              Mumbai, Maharashtra
            </div>

            <div className="online-pill">
              <span className="online-dot"></span>
              SYSTEM ONLINE
            </div>

          </div>

        </div>

      </header>


      {/* =====================================================
          MAIN DASHBOARD
      ===================================================== */}

      <main className="dashboard">

        <section className="intro-section">

          <div className="eyebrow">
            AI EMERGENCY DECISION SUPPORT
          </div>

          <h1>
            Mumbai Disaster Intelligence Dashboard
          </h1>

          <p className="intro-description">
            Real-time spatial risk monitoring, ST-GNN flood
            forecasting and intelligent emergency resource
            allocation.
          </p>

          <div className="technology-row">

            <span className="technology-chip">
              🧠 ST-GNN Forecasting
            </span>

            <span className="technology-chip">
              🗺️ {totalNodes} Spatial Nodes
            </span>

            <span className="technology-chip">
              ⚡ AI Priority Engine
            </span>

            <span className="technology-chip">
              📦 Resource Optimization
            </span>

          </div>

        </section>


        {/* =====================================================
            ST-GNN COVERAGE
        ===================================================== */}

        <section className="forecast-banner">

          <div className="forecast-left">

            <div className="forecast-icon">
              🧠
            </div>

            <div>

              <div className="forecast-title">
                ST-GNN FORECAST COVERAGE
              </div>

              <div className="forecast-subtitle">
                Spatiotemporal flood prediction availability
              </div>

            </div>

          </div>

          <div className="forecast-value">
            {formatNumber(
              prediction.stgnn_forecast_coverage_percent,
              1
            )}
            %
          </div>

          <div className="forecast-matched">

            <strong>
              {prediction.stgnn_forecast_nodes || 0}
            </strong>

            <span>
              spatial nodes matched
            </span>

          </div>

          <div className="live-indicator">
            <span></span>
            LIVE
          </div>

        </section>


        {/* =====================================================
            KPI CARDS
        ===================================================== */}

        <section className="kpi-grid">

          <div className="kpi-card">

            <div className="kpi-icon blue">
              🗺️
            </div>

            <div className="kpi-label">
              Total Spatial Nodes
            </div>

            <div className="kpi-value">
              {totalNodes}
            </div>

            <div className="kpi-description">
              Mumbai monitoring grid
            </div>

          </div>


          <div className="kpi-card">

            <div className="kpi-icon cyan">
              🌊
            </div>

            <div className="kpi-label">
              Average Flood Probability
            </div>

            <div className="kpi-value">
              {probabilityPercent(
                prediction.average_flood_probability
              )}
            </div>

            <div className="kpi-description">
              Current risk signal
            </div>

          </div>


          <div className="kpi-card">

            <div className="kpi-icon orange">
              ⚠️
            </div>

            <div className="kpi-label">
              Maximum Flood Probability
            </div>

            <div className="kpi-value">
              {probabilityPercent(
                prediction.maximum_flood_probability
              )}
            </div>

            <div className="kpi-description">
              Highest susceptibility
            </div>

          </div>


          <div className="kpi-card">

            <div className="kpi-icon purple">
              📦
            </div>

            <div className="kpi-label">
              Resources Required
            </div>

            <div className="kpi-value">
              {formatNumber(
                prediction.total_resources_required,
                0
              )}
            </div>

            <div className="kpi-description">
              Total emergency demand
            </div>

          </div>


          <div className="kpi-card">

            <div className="kpi-icon red">
              🚑
            </div>

            <div className="kpi-label">
              Resources Allocated
            </div>

            <div className="kpi-value">
              {formatNumber(
                prediction.total_resources_allocated,
                0
              )}
            </div>

            <div className="kpi-description">
              Current deployment
            </div>

          </div>

        </section>


        {/* =====================================================
            RESOURCE OPTIMIZATION
        ===================================================== */}

        <section className="resource-coverage panel">

          <div className="section-heading">

            <div>
              <div className="eyebrow">
                RESOURCE OPTIMIZATION
              </div>

              <h2>
                🚑 Emergency Resource Coverage
              </h2>

              <p>
                System-wide emergency resource allocation status
              </p>
            </div>

            <div className="coverage-status">
              <span className="online-dot"></span>
              ALLOCATION ACTIVE
            </div>

          </div>

          <div className="coverage-main">

            <div className="coverage-number">
              {formatNumber(coveragePercent, 1)}%
            </div>

            <div className="coverage-label">
              Resource coverage
            </div>

            <div className="coverage-track">
              <div
                className="coverage-fill"
                style={{
                  width: `${coveragePercent}%`,
                }}
              ></div>
            </div>

          </div>

          <div className="coverage-stats">

            <div>
              <span>Required</span>
              <strong>{coverageRequired}</strong>
            </div>

            <div>
              <span>Allocated</span>
              <strong>{coverageAllocated}</strong>
            </div>

            <div>
              <span>Coverage</span>
              <strong>
                {formatNumber(coveragePercent, 1)}%
              </strong>
            </div>

            <div>
              <span>Shortage Locations</span>
              <strong>{shortageLocations}</strong>
            </div>

          </div>

        </section>


        {/* =====================================================
            MAP
        ===================================================== */}

        <section className="panel map-panel">

          <div className="section-heading">

            <div>

              <div className="eyebrow">
                GEOSPATIAL INTELLIGENCE
              </div>

              <h2>
                Mumbai Flood Risk Map
              </h2>

              <p>
                {totalNodes} spatial nodes • AI-generated
                disaster risk intelligence
              </p>

            </div>

            <div className="map-live">
              <span></span>
              LIVE DATA
            </div>

          </div>

          <RiskMap nodes={priorities} />

        </section>


        {/* =====================================================
            EMERGENCY PRIORITIES
        ===================================================== */}

        <section className="panel emergency-panel">

          <div className="section-heading priority-heading">

            <div>

              <div className="eyebrow">
                AI DECISION ENGINE
              </div>

              <h2>
                Emergency Priorities
              </h2>

              <p>
                Locations ranked by combined flood,
                environmental and resource-risk signals
              </p>

            </div>

            <div className="priority-summary">

              <div className="summary-item critical">
                <strong>{criticalCount}</strong>
                <span>Critical</span>
              </div>

              <div className="summary-item high">
                <strong>{highCount}</strong>
                <span>High</span>
              </div>

              <div className="summary-item medium">
                <strong>{mediumCount}</strong>
                <span>Medium</span>
              </div>

              <div className="summary-item low">
                <strong>{lowCount}</strong>
                <span>Low</span>
              </div>

            </div>

          </div>


          <div className="priority-list">

            {topPriorities.map((node, index) => {

              const rank =
                Number(node.emergency_rank) ||
                index + 1;

              const required =
                Number(node.resources_required) || 0;

              const allocated =
                Number(node.allocated_resources) || 0;

              const aiScore =
                Number(node.emergency_priority_score) || 0;

              const forecast =
                getForecast(node);

              const resourceStatus =
                getResourceStatus(node);

              const risk =
                node.risk_level || "LOW";

              const emergencyPriority =
                node.emergency_priority || "LOW";

              return (
                <article
                  className="priority-card"
                  key={`${node.grid_id || node.place_name || "node"}-${rank}`}
                >

                  {/* CARD TOP */}

                  <div className="priority-top">

                    <div className="rank-badge">
                      #{rank}
                    </div>

                    <div className="priority-location">

                      <h3>
                        {node.place_name ||
                          `Mumbai Grid Node ${rank}`}
                      </h3>

                      <div className="location-line">
                        <span>📍</span>
                        <strong>
                          Mumbai, Maharashtra
                        </strong>
                      </div>

                      <div className="coordinates">
                        {formatNumber(node.latitude, 4)}
                        <span> , </span>
                        {formatNumber(node.longitude, 4)}

                        {node.region !== undefined &&
                          node.region !== null && (
                            <span className="region-tag">
                              Region {node.region}
                            </span>
                          )}
                      </div>

                    </div>

                    <div
                      className={`risk-badge ${getRiskClass(
                        risk
                      )}`}
                    >
                      {risk}
                    </div>

                  </div>


                  {/* SIGNAL GRID */}

                  <div className="signal-grid">

                    <div className="signal-card">

                      <div className="signal-icon">
                        🌊
                      </div>

                      <div>
                        <span>
                          FLOOD RISK
                        </span>

                        <strong>
                          {probabilityPercent(
                            node.flood_probability
                          )}
                        </strong>
                      </div>

                    </div>


                    <div className="signal-card forecast">

                      <div className="signal-icon">
                        🤖
                      </div>

                      <div>
                        <span>
                          ST-GNN FORECAST
                        </span>

                        <strong>
                          {forecast !== null
                            ? probabilityPercent(forecast)
                            : "N/A"}
                        </strong>

                        <small>
                          Spatiotemporal prediction
                        </small>
                      </div>

                    </div>


                    <div className="signal-card">

                      <div className="signal-icon">
                        📦
                      </div>

                      <div>
                        <span>
                          REQUIRED
                        </span>

                        <strong>
                          {formatNumber(required, 0)}
                        </strong>

                        <small>
                          Emergency resources
                        </small>
                      </div>

                    </div>


                    <div className="signal-card">

                      <div className="signal-icon">
                        🚑
                      </div>

                      <div>
                        <span>
                          ALLOCATED
                        </span>

                        <strong>
                          {formatNumber(allocated, 0)}
                        </strong>

                        <small>
                          Current deployment
                        </small>
                      </div>

                    </div>

                  </div>


                  {/* RESOURCE STATUS */}

                  <div className="resource-status-row">

                    <div className="resource-status-label">
                      <span>⚠️</span>
                      Resource Status
                    </div>

                    <div
                      className={`resource-status ${resourceStatus.type}`}
                    >
                      {resourceStatus.type === "surplus" && "✓ "}
                      {resourceStatus.type === "shortage" && "⚠ "}
                      {resourceStatus.type === "covered" && "✓ "}

                      {resourceStatus.label}
                    </div>

                  </div>


                  {/* AI SCORE */}

                  <div className="ai-score-row">

                    <div>

                      <span className="score-label">
                        🧠 AI PRIORITY SCORE
                      </span>

                      <strong className="score-value">
                        {formatNumber(aiScore, 1)}
                      </strong>

                    </div>

                    <div
                      className={`emergency-priority ${getPriorityClass(
                        emergencyPriority
                      )}`}
                    >
                      {emergencyPriority}
                    </div>

                  </div>


                  {/* ACTION */}

                  <div className="recommended-action">

                    <div className="action-title">
                      🎯 RECOMMENDED ACTION
                    </div>

                    <div className="action-text">
                      {getAction(node)}
                    </div>

                  </div>

                </article>
              );
            })}

          </div>

        </section>


        {/* =====================================================
            RESOURCE EXPLANATION
        ===================================================== */}

        <section className="explanation-grid">

          <div className="explanation-card">

            <div className="explanation-icon">
              📦
            </div>

            <div>

              <h3>
                Resource Demand
              </h3>

              <p>
                AI estimates the emergency resources
                required at each spatial node based on
                disaster risk and resource-demand signals.
              </p>

              <strong>
                Demand = Required resources
              </strong>

            </div>

          </div>


          <div className="explanation-card">

            <div className="explanation-icon">
              🚑
            </div>

            <div>

              <h3>
                Resource Allocation
              </h3>

              <p>
                DisasterFlow compares required resources
                against current allocation to identify
                underserved or over-covered locations.
              </p>

              <strong>
                Allocated = Current deployment
              </strong>

            </div>

          </div>


          <div className="explanation-card">

            <div className="explanation-icon">
              🧠
            </div>

            <div>

              <h3>
                AI Priority Engine
              </h3>

              <p>
                ST-GNN forecasts are combined with flood,
                rainfall, water exposure and resource-gap
                signals to rank emergency locations.
              </p>

              <strong>
                Multi-signal emergency scoring
              </strong>

            </div>

          </div>

        </section>


        {/* =====================================================
            RISK DISTRIBUTION
        ===================================================== */}

        <section className="bottom-grid">

          <div className="panel">

            <div className="section-heading">

              <div>

                <div className="eyebrow">
                  RISK ANALYTICS
                </div>

                <h2>
                  Risk Distribution
                </h2>

              </div>

            </div>

            <div className="risk-distribution">

              {[
                {
                  label: "CRITICAL",
                  count: criticalCount,
                  className: "critical",
                },
                {
                  label: "HIGH",
                  count: highCount,
                  className: "high",
                },
                {
                  label: "MEDIUM",
                  count: mediumCount,
                  className: "medium",
                },
                {
                  label: "LOW",
                  count: lowCount,
                  className: "low",
                },
              ].map((item) => {

                const percentage =
                  totalNodes > 0
                    ? (item.count / totalNodes) * 100
                    : 0;

                return (
                  <div
                    className="distribution-item"
                    key={item.label}
                  >

                    <div className="distribution-header">

                      <span>
                        <i
                          className={`distribution-dot ${item.className}`}
                        ></i>

                        {item.label}
                      </span>

                      <strong>
                        {item.count}
                      </strong>

                    </div>

                    <div className="distribution-track">

                      <div
                        className={`distribution-fill ${item.className}`}
                        style={{
                          width: `${percentage}%`,
                        }}
                      ></div>

                    </div>

                    <div className="distribution-percent">
                      {formatNumber(percentage, 1)}%
                      of monitored nodes
                    </div>

                  </div>
                );
              })}

            </div>

          </div>


          <div className="panel insight-panel">

            <div className="section-heading">

              <div>

                <div className="eyebrow">
                  AI SYSTEM INSIGHT
                </div>

                <h2>
                  Emergency Intelligence
                </h2>

              </div>

            </div>

            <div className="insight-content">

              <div className="insight-icon">
                🧠
              </div>

              <p>
                DisasterFlow continuously evaluates
                spatial flood susceptibility, ST-GNN
                forecast signals and emergency resource
                availability to identify locations requiring
                attention.
              </p>

            </div>

            <div className="insight-stats">

              <div>
                <strong>{criticalCount}</strong>
                <span>Critical</span>
              </div>

              <div>
                <strong>{highCount}</strong>
                <span>High Priority</span>
              </div>

              <div>
                <strong>{shortageLocations}</strong>
                <span>Shortage</span>
              </div>

            </div>

          </div>

        </section>


        {/* =====================================================
            FOOTER
        ===================================================== */}

        <footer className="footer">

          <div className="footer-brand">
            <div className="footer-logo">
              DF
            </div>

            <div>
              <strong>
                DisasterFlow AI
              </strong>

              <span>
                AI-Powered Spatiotemporal Disaster
                Intelligence & Emergency Resource Allocation
              </span>
            </div>
          </div>

          <div className="footer-tech">
            <span>FastAPI</span>
            <span>ST-GNN</span>
            <span>GIS</span>
            <span>AI Priority Engine</span>
            <span>{totalNodes} Mumbai Nodes</span>
          </div>

          <div className="footer-update">
            Last updated:{" "}
            {lastUpdated
              ? lastUpdated.toLocaleTimeString()
              : "—"}
          </div>

        </footer>

      </main>
    </div>
  );
}

export default App;