import os
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

print("=" * 70)
print("DISASTERFLOW AI — PHASE 5A")
print("Spatial Graph Construction")
print("=" * 70)

# ============================================================
# PATHS
# ============================================================

INPUT_FILE = "data/processed/disasterflow_final_node_priorities.csv"
OUTPUT_DIR = "data/processed"

NODE_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "spatial_graph_nodes.csv"
)

EDGE_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "spatial_graph_edges.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 1. LOAD SPATIAL DATA
# ============================================================

print("\n[1/7] Loading spatial node data...")

df = pd.read_csv(INPUT_FILE)

print("Rows:", len(df))
print("Columns:", len(df.columns))

required = [
    "grid_id",
    "latitude",
    "longitude",
    "elevation_m",
    "slope_degrees",
    "distance_to_water_m"
]

missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )

# ============================================================
# 2. VALIDATE NODES
# ============================================================

print("\n[2/7] Validating spatial nodes...")

df = df.drop_duplicates("grid_id").reset_index(drop=True)

print("Unique nodes:", df["grid_id"].nunique())

if df["grid_id"].nunique() != 140:
    raise ValueError(
        f"Expected 140 nodes, found {df['grid_id'].nunique()}"
    )

print("✓ All 140 spatial nodes available")

# ============================================================
# 3. CREATE NODE INDEX
# ============================================================

print("\n[3/7] Creating node index...")

df["node_index"] = np.arange(len(df))

node_columns = [
    "node_index",
    "grid_id",
    "latitude",
    "longitude",
    "elevation_m",
    "slope_degrees",
    "distance_to_water_m"
]

nodes = df[node_columns].copy()

print("Node index range:",
      nodes["node_index"].min(),
      "to",
      nodes["node_index"].max())

# ============================================================
# 4. BUILD KNN GRAPH
# ============================================================

print("\n[4/7] Building spatial KNN graph...")

coordinates = nodes[
    ["latitude", "longitude"]
].values

# 6 nearest neighbours
K = 6

knn = NearestNeighbors(
    n_neighbors=K + 1,
    metric="haversine"
)

# Convert degrees to radians
coords_rad = np.radians(coordinates)

knn.fit(coords_rad)

distances, indices = knn.kneighbors(coords_rad)

print("K:", K)

# ============================================================
# 5. CREATE EDGES
# ============================================================

print("\n[5/7] Creating graph edges...")

edges = []

EARTH_RADIUS_KM = 6371.0

for i in range(len(nodes)):

    for j_pos in range(1, K + 1):

        j = indices[i, j_pos]

        distance_km = (
            distances[i, j_pos]
            * EARTH_RADIUS_KM
        )

        edges.append({
            "source": int(i),
            "target": int(j),
            "source_grid_id": nodes.iloc[i]["grid_id"],
            "target_grid_id": nodes.iloc[j]["grid_id"],
            "distance_km": float(distance_km)
        })

edges = pd.DataFrame(edges)

print("Directed edges:", len(edges))

# ============================================================
# 6. ADD SYMMETRIC EDGES
# ============================================================

print("\n[6/7] Making graph bidirectional...")

reverse_edges = edges.rename(
    columns={
        "source": "target",
        "target": "source",
        "source_grid_id": "target_grid_id",
        "target_grid_id": "source_grid_id"
    }
)

edges = pd.concat(
    [edges, reverse_edges],
    ignore_index=True
)

# Remove duplicate edges
edges = edges.drop_duplicates(
    subset=["source", "target"]
).reset_index(drop=True)

print("Final directed edges:", len(edges))

print(
    "Average edge distance:",
    round(edges["distance_km"].mean(), 3),
    "km"
)

print(
    "Maximum edge distance:",
    round(edges["distance_km"].max(), 3),
    "km"
)

# ============================================================
# 7. SAVE
# ============================================================

print("\n[7/7] Saving graph...")

nodes.to_csv(
    NODE_OUTPUT,
    index=False
)

edges.to_csv(
    EDGE_OUTPUT,
    index=False
)

print("\n" + "=" * 70)
print("PHASE 5A COMPLETE")
print("=" * 70)

print("\nNode file:")
print(os.path.abspath(NODE_OUTPUT))

print("\nEdge file:")
print(os.path.abspath(EDGE_OUTPUT))

print("\nNodes:", len(nodes))
print("Edges:", len(edges))

print("\nSample nodes:")
print(nodes.head().to_string(index=False))

print("\nSample edges:")
print(edges.head(10).to_string(index=False))

print("\n✓ Spatial graph created successfully.")
print("=" * 70)