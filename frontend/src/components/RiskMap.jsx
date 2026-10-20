import {
  MapContainer,
  TileLayer,
  CircleMarker,
  Popup,
  Tooltip
} from "react-leaflet";

import "leaflet/dist/leaflet.css";

function getRiskColor(risk) {
  switch (String(risk).toUpperCase()) {
    case "CRITICAL":
      return "#dc2626";
    case "HIGH":
      return "#f97316";
    case "MEDIUM":
      return "#eab308";
    case "LOW":
      return "#22c55e";
    default:
      return "#64748b";
  }
}

function getAllocationStatus(demand, allocated) {
  const d = Number(demand) || 0;
  const a = Number(allocated) || 0;

  if (a >= d) {
    return "ADEQUATELY ALLOCATED";
  }

  if (a >= d * 0.5) {
    return "PARTIALLY ALLOCATED";
  }

  return "UNDER-ALLOCATED";
}

function getPrioritySize(priority) {
  const p = Number(priority) || 0;

  if (p >= 0.9) return 12;
  if (p >= 0.75) return 10;
  if (p >= 0.5) return 8;
  return 6;
}

function RiskMap({ nodes = [] }) {
  const validNodes = nodes.filter(
    (node) =>
      Number.isFinite(Number(node.latitude)) &&
      Number.isFinite(Number(node.longitude))
  );

  const center = [19.076, 72.8777];

  return (
    <div className="risk-map-wrapper">

      <MapContainer
        center={center}
        zoom={10}
        scrollWheelZoom={true}
        style={{
          height: "550px",
          width: "100%",
          borderRadius: "14px"
        }}
      >

        <TileLayer
          attribution="&copy; OpenStreetMap contributors"
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        {validNodes.map((node) => {

          const probability =
            Number(node.flood_probability) || 0;

          const demand =
            Number(node.resources_required) || 0;

          const allocated =
            Number(node.allocated_resources) || 0;

          const priority =
            Number(node.operational_priority) || 0;

          const risk =
            String(node.risk_level || "UNKNOWN").toUpperCase();

          const color =
            getRiskColor(risk);

          const radius =
            getPrioritySize(priority);

          return (
            <CircleMarker
              key={node.grid_id}
              center={[
                Number(node.latitude),
                Number(node.longitude)
              ]}
              radius={radius}
              pathOptions={{
                color: color,
                fillColor: color,
                fillOpacity: 0.78,
                weight: 2
              }}
            >

              <Tooltip
                direction="top"
                offset={[0, -8]}
              >
                <strong>
                  {node.grid_id}
                </strong>
                <br />
                {risk} • Priority #{node.priority_rank}
              </Tooltip>

              <Popup>

                <div className="node-popup">

                  <div className="popup-header">

                    <h3>
                      {node.grid_id}
                    </h3>

                    <span
                      className={`risk-badge ${risk.toLowerCase()}`}
                    >
                      {risk}
                    </span>

                  </div>

                  <div className="popup-section">

                    <p>
                      <strong>Flood Probability</strong>
                      <span>
                        {(probability * 100).toFixed(3)}%
                      </span>
                    </p>

                    <p>
                      <strong>Priority Rank</strong>
                      <span>
                        #{node.priority_rank}
                      </span>
                    </p>

                    <p>
                      <strong>Operational Priority</strong>
                      <span>
                        {priority.toFixed(3)}
                      </span>
                    </p>

                  </div>

                  <div className="popup-section">

                    <p>
                      <strong>Resource Demand</strong>
                      <span>
                        {demand}
                      </span>
                    </p>

                    <p>
                      <strong>Resources Allocated</strong>
                      <span>
                        {allocated}
                      </span>
                    </p>

                    <p>
                      <strong>Allocation Status</strong>
                      <span>
                        {getAllocationStatus(
                          demand,
                          allocated
                        )}
                      </span>
                    </p>

                  </div>

                  <div className="popup-section">

                    <p>
                      <strong>Elevation</strong>
                      <span>
                        {Number(
                          node.elevation_m || 0
                        ).toFixed(2)} m
                      </span>
                    </p>

                    <p>
                      <strong>Slope</strong>
                      <span>
                        {Number(
                          node.slope_degrees || 0
                        ).toFixed(2)}°
                      </span>
                    </p>

                    <p>
                      <strong>Distance to Water</strong>
                      <span>
                        {Number(
                          node.distance_to_water_m || 0
                        ).toFixed(2)} m
                      </span>
                    </p>

                  </div>

                  <div className="popup-location">

                    <span>
                      Latitude:{" "}
                      {Number(
                        node.latitude
                      ).toFixed(5)}
                    </span>

                    <span>
                      Longitude:{" "}
                      {Number(
                        node.longitude
                      ).toFixed(5)}
                    </span>

                  </div>

                </div>

              </Popup>

            </CircleMarker>
          );
        })}

      </MapContainer>

      <div className="map-legend">

        <strong>Flood Risk</strong>

        <div className="legend-item">
          <span
            className="legend-dot low"
          />
          LOW
        </div>

        <div className="legend-item">
          <span
            className="legend-dot medium"
          />
          MEDIUM
        </div>

        <div className="legend-item">
          <span
            className="legend-dot high"
          />
          HIGH
        </div>

        <div className="legend-item">
          <span
            className="legend-dot critical"
          />
          CRITICAL
        </div>

        <div className="legend-note">
          Marker size = operational priority
        </div>

      </div>

    </div>
  );
}

export default RiskMap;
