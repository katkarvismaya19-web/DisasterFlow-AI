import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# ============================================================
# DISASTERFLOW AI — PHASE 5B
# Spatiotemporal GNN Dataset Construction
# MEMORY-SAFE VERSION
# ============================================================

print("=" * 70)
print("DISASTERFLOW AI — PHASE 5B")
print("Spatiotemporal GNN Dataset Construction")
print("=" * 70)

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

TRAIN_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "ml_train.csv"
)

VAL_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "ml_validation.csv"
)

TEST_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "ml_test.csv"
)

GRAPH_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "spatial_graph_edges.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "gnn"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

NUM_NODES = 140

SEQUENCE_LENGTH = 24
FORECAST_HORIZON = 6

TARGET = "future_flood_event_6h"

FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "wind_speed_10m",
    "rainfall_1h",
    "rainfall_3h",
    "rainfall_6h",
    "rainfall_12h",
    "rainfall_24h",
    "rainfall_72h",
    "rainfall_max_6h",
    "rainfall_max_24h",
    "rainfall_baseline",
    "rainfall_std",
    "rainfall_anomaly",
    "rainfall_3h_previous",
    "rainfall_change_3h",
    "rain_persistence_6h",
    "humidity_stress",
    "wind_stress",
    "rain_humidity_interaction",
    "elevation_m",
    "slope_degrees",
    "distance_to_water_m",
    "flood_probability",
    "elevation_risk",
    "water_risk",
    "dynamic_rainfall_risk",
    "dynamic_flood_signal",
    "environmental_stress",
    "resource_demand",
    "latitude_static",
    "longitude_static",
    "hour",
    "day_of_year",
    "month",
    "monsoon"
]

ID_COLUMNS = [
    "grid_id",
    "time"
]

REQUIRED_COLUMNS = (
    ID_COLUMNS +
    FEATURES +
    [TARGET]
)

# ============================================================
# STEP 1 — LOAD DATA
# ============================================================

print("\n[1/8] Loading ML datasets...")

train = pd.read_csv(TRAIN_FILE)
validation = pd.read_csv(VAL_FILE)
test = pd.read_csv(TEST_FILE)

print("Train rows:", len(train))
print("Validation rows:", len(validation))
print("Test rows:", len(test))

print("Train nodes:", train.grid_id.nunique())
print("Validation nodes:", validation.grid_id.nunique())
print("Test nodes:", test.grid_id.nunique())

# ============================================================
# STEP 2 — VALIDATE COLUMNS
# ============================================================

print("\n[2/8] Validating columns...")

for dataset_name, dataset in [
    ("train", train),
    ("validation", validation),
    ("test", test)
]:

    missing = [
        col
        for col in REQUIRED_COLUMNS
        if col not in dataset.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in {dataset_name}: {missing}"
        )

print("✓ All required columns available")
print("Features:", len(FEATURES))

# ============================================================
# STEP 3 — LOAD GRAPH
# ============================================================

print("\n[3/8] Loading spatial graph...")

edges = pd.read_csv(GRAPH_FILE)

edge_index = edges[
    ["source", "target"]
].to_numpy(dtype=np.int64).T

print("Edges:", len(edges))
print("Edge index shape:", edge_index.shape)

print("✓ Spatial graph loaded")

# ============================================================
# STEP 4 — COMBINE DATA
# ============================================================

print("\n[4/8] Combining temporal datasets...")

train["split"] = "train"
validation["split"] = "validation"
test["split"] = "test"

df = pd.concat(
    [train, validation, test],
    ignore_index=True
)

del train
del validation
del test

df["time"] = pd.to_datetime(df["time"])

df = df.sort_values(
    ["time", "grid_id"]
).reset_index(drop=True)

print("Combined rows:", len(df))
print("Nodes:", df.grid_id.nunique())
print(
    "Time range:",
    df.time.min(),
    "to",
    df.time.max()
)

node_ids = sorted(
    df["grid_id"].unique()
)

if len(node_ids) != NUM_NODES:
    raise ValueError(
        f"Expected {NUM_NODES} nodes, "
        f"found {len(node_ids)}"
    )

node_to_index = {
    node_id: i
    for i, node_id in enumerate(node_ids)
}

df["node_index"] = df["grid_id"].map(
    node_to_index
)

print(
    f"✓ {NUM_NODES} spatial nodes confirmed"
)

# ============================================================
# STEP 5 — STANDARDIZE FEATURES
# ============================================================

print("\n[5/8] Standardizing GNN features...")

scaler = StandardScaler()

train_mask = (
    df["split"] == "train"
)

scaler.fit(
    df.loc[
        train_mask,
        FEATURES
    ]
)

df[FEATURES] = scaler.transform(
    df[FEATURES]
)

scaler_file = os.path.join(
    OUTPUT_DIR,
    "gnn_feature_scaler.joblib"
)

joblib.dump(
    scaler,
    scaler_file
)

print("✓ Standardization complete")
print("✓ Scaler fitted only on training data")

# ============================================================
# STEP 6 — BUILD HOURLY NODE MATRICES
# ============================================================

print("\n[6/8] Building hourly node matrices...")

times = pd.DatetimeIndex(
    sorted(df["time"].unique())
)

print(
    "Unique timestamps:",
    len(times)
)

# IMPORTANT:
#
# The flood-target dataset ends at
# 2025-12-30 23:00 because the final
# 24 hours were removed when creating
# future flood targets.
#
# Therefore the correct number is:
#
# 17520 hours
#
# NOT 17544.

EXPECTED_HOURS = 17520

if len(times) != EXPECTED_HOURS:

    raise ValueError(
        f"Expected {EXPECTED_HOURS} timestamps, "
        f"found {len(times)}"
    )

# ------------------------------------------------------------
# Check hourly continuity
# ------------------------------------------------------------

time_diffs = times.to_series().diff().dropna()

expected_delta = pd.Timedelta(
    hours=1
)

bad_intervals = (
    time_diffs != expected_delta
)

if bad_intervals.sum() > 0:

    print(
        "⚠ Non-hourly intervals detected:"
    )

    print(
        time_diffs[bad_intervals]
        .head(10)
    )

    raise ValueError(
        "Timeline is not continuous hourly."
    )

print(
    f"✓ Complete hourly timeline confirmed "
    f"({len(times)} hours)"
)

# ------------------------------------------------------------
# Allocate hourly matrices
# ------------------------------------------------------------

num_features = len(FEATURES)

feature_array = np.zeros(
    (
        len(times),
        NUM_NODES,
        num_features
    ),
    dtype=np.float32
)

target_array = np.zeros(
    (
        len(times),
        NUM_NODES
    ),
    dtype=np.int8
)

split_array = np.empty(
    len(times),
    dtype="U12"
)

print(
    "Feature tensor:",
    feature_array.shape
)

print(
    "Feature memory:",
    round(
        feature_array.nbytes /
        (1024 ** 2),
        2
    ),
    "MB"
)

# ------------------------------------------------------------
# Efficient population
# ------------------------------------------------------------

print(
    "\nCreating hourly lookup..."
)

for i, timestamp in enumerate(times):

    hour_df = df[
        df["time"] == timestamp
    ].sort_values(
        "node_index"
    )

    if len(hour_df) != NUM_NODES:

        raise ValueError(
            f"{timestamp} has "
            f"{len(hour_df)} nodes; "
            f"expected {NUM_NODES}"
        )

    feature_array[i] = (
        hour_df[FEATURES]
        .to_numpy(
            dtype=np.float32
        )
    )

    target_array[i] = (
        hour_df[TARGET]
        .to_numpy(
            dtype=np.int8
        )
    )

    split_array[i] = (
        hour_df["split"].iloc[0]
    )

    if i % 1000 == 0:

        print(
            f"  Processed "
            f"{i}/{len(times)} timestamps"
        )

print("✓ Hourly matrices created")

# ============================================================
# STEP 7 — CREATE MEMORY-SAFE SEQUENCES
# ============================================================

print("\n[7/8] Creating temporal sequences...")

sequence_count = (
    len(times)
    - SEQUENCE_LENGTH
    - FORECAST_HORIZON
    + 1
)

print(
    "Potential sequences:",
    sequence_count
)

X_shape = (
    sequence_count,
    SEQUENCE_LENGTH,
    NUM_NODES,
    num_features
)

Y_shape = (
    sequence_count,
    NUM_NODES
)

print(
    "X shape:",
    X_shape
)

print(
    "Y shape:",
    Y_shape
)

# ------------------------------------------------------------
# Disk-backed memory maps
# ------------------------------------------------------------

X_file = os.path.join(
    OUTPUT_DIR,
    "X_sequences.dat"
)

Y_file = os.path.join(
    OUTPUT_DIR,
    "Y_targets.dat"
)

X = np.memmap(
    X_file,
    dtype=np.float32,
    mode="w+",
    shape=X_shape
)

Y = np.memmap(
    Y_file,
    dtype=np.int8,
    mode="w+",
    shape=Y_shape
)

sequence_times = np.empty(
    sequence_count,
    dtype="datetime64[ns]"
)

sequence_splits = np.empty(
    sequence_count,
    dtype="U12"
)

print(
    "\n✓ Disk-backed sequence arrays created"
)

print(
    "X disk size:",
    round(
        np.prod(X_shape) *
        4 /
        (1024 ** 3),
        2
    ),
    "GB"
)

# ------------------------------------------------------------
# Build sequences
# ------------------------------------------------------------

for i in range(sequence_count):

    input_start = i

    input_end = (
        i +
        SEQUENCE_LENGTH
    )

    target_index = (
        input_end +
        FORECAST_HORIZON -
        1
    )

    X[i] = feature_array[
        input_start:input_end
    ]

    Y[i] = target_array[
        target_index
    ]

    sequence_times[i] = (
        times[target_index]
    )

    sequence_splits[i] = (
        split_array[target_index]
    )

    if i % 1000 == 0:

        print(
            f"  Created "
            f"{i}/{sequence_count} sequences"
        )

# Flush disk-backed arrays

X.flush()
Y.flush()

print(
    "✓ Sequence construction complete"
)

# ------------------------------------------------------------
# Save sequence metadata
# ------------------------------------------------------------

np.save(
    os.path.join(
        OUTPUT_DIR,
        "sequence_times.npy"
    ),
    sequence_times
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "sequence_splits.npy"
    ),
    sequence_splits
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "edge_index.npy"
    ),
    edge_index
)

# ============================================================
# STEP 8 — SAVE METADATA + FINAL VALIDATION
# ============================================================

print("\n[8/8] Saving GNN metadata...")

metadata = {
    "num_nodes": NUM_NODES,
    "num_features": num_features,
    "sequence_length": SEQUENCE_LENGTH,
    "forecast_horizon": FORECAST_HORIZON,
    "target": TARGET,
    "num_edges": int(edge_index.shape[1]),
    "num_timestamps": len(times),
    "num_sequences": sequence_count,
    "X_shape": list(X_shape),
    "Y_shape": list(Y_shape),
    "storage": "numpy_memmap"
}

pd.Series(metadata).to_json(
    os.path.join(
        OUTPUT_DIR,
        "gnn_metadata.json"
    ),
    indent=2
)

# ------------------------------------------------------------
# Final validation
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PHASE 5B COMPLETE")
print("=" * 70)

print("\nDataset:")
print("X shape:", X_shape)
print("Y shape:", Y_shape)

print("\nNodes:", NUM_NODES)
print("Features:", num_features)
print(
    "Sequence length:",
    SEQUENCE_LENGTH
)
print(
    "Forecast horizon:",
    FORECAST_HORIZON
)
print(
    "Edges:",
    edge_index.shape[1]
)

print("\nSequence split distribution:")

unique, counts = np.unique(
    sequence_splits,
    return_counts=True
)

for name, count in zip(
    unique,
    counts
):

    print(
        f"{name}: {count}"
    )

print("\nTarget distribution:")

unique, counts = np.unique(
    Y,
    return_counts=True
)

for value, count in zip(
    unique,
    counts
):

    print(
        f"{value}: {count}"
    )

print("\nSaved files:")

print(X_file)
print(Y_file)

print(
    os.path.join(
        OUTPUT_DIR,
        "sequence_times.npy"
    )
)

print(
    os.path.join(
        OUTPUT_DIR,
        "sequence_splits.npy"
    )
)

print(
    os.path.join(
        OUTPUT_DIR,
        "edge_index.npy"
    )
)

print(scaler_file)

print(
    os.path.join(
        OUTPUT_DIR,
        "gnn_metadata.json"
    )
)

print(
    "\n✓ Phase 5B finished successfully."
)

print("=" * 70)