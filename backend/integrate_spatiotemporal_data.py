# ============================================================
# DISASTERFLOW AI — PHASE 2C
# SPATIAL + SPATIOTEMPORAL WEATHER INTEGRATION
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np


print("=" * 70)
print("DISASTERFLOW AI — PHASE 2C")
print("Spatial + Spatiotemporal Weather Integration")
print("=" * 70)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

SPATIAL_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "disasterflow_final_node_priorities.csv"
)

WEATHER_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "mumbai_spatiotemporal_features.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "disasterflow_spatiotemporal_nodes.csv"
)


# ============================================================
# CHECK INPUT FILES
# ============================================================

print("\n[1/9] Checking input files...")

if not SPATIAL_FILE.exists():
    raise FileNotFoundError(
        f"Spatial dataset not found:\n{SPATIAL_FILE}"
    )

if not WEATHER_FILE.exists():
    raise FileNotFoundError(
        f"Weather feature dataset not found:\n{WEATHER_FILE}"
    )

print("✓ Spatial dataset found")
print("✓ Weather feature dataset found")


# ============================================================
# LOAD SPATIAL DATA
# ============================================================

print("\n[2/9] Loading spatial node data...")

spatial = pd.read_csv(SPATIAL_FILE)

print("Spatial rows:", len(spatial))
print("Spatial nodes:", spatial["grid_id"].nunique())

required_spatial = [
    "grid_id",
    "latitude",
    "longitude",
    "elevation_m",
    "slope_degrees",
    "distance_to_water_m",
    "flood_probability",
    "risk_level",
    "elevation_risk",
    "water_risk",
    "resource_demand",
    "resources_required",
    "region_id",
    "allocated_resources",
    "operational_priority",
    "priority_rank",
]

missing_spatial = [
    c for c in required_spatial
    if c not in spatial.columns
]

if missing_spatial:
    raise ValueError(
        f"Missing spatial columns: {missing_spatial}"
    )

# Ensure one row per spatial node
if spatial["grid_id"].duplicated().any():

    raise ValueError(
        "Spatial dataset contains duplicate grid_id values."
    )

print("✓ Spatial dataset validated")


# ============================================================
# LOAD WEATHER FEATURES
# ============================================================

print("\n[3/9] Loading spatiotemporal weather data...")

weather = pd.read_csv(
    WEATHER_FILE,
    parse_dates=["time"]
)

print("Weather rows:", len(weather))
print("Weather nodes:", weather["grid_id"].nunique())

print(
    "Weather time range:",
    weather["time"].min(),
    "to",
    weather["time"].max()
)


# ============================================================
# WEATHER VALIDATION
# ============================================================

required_weather = [
    "grid_id",
    "latitude",
    "longitude",
    "time",

    # Base weather
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "wind_speed_10m",

    # Rainfall accumulation
    "rainfall_1h",
    "rainfall_3h",
    "rainfall_6h",
    "rainfall_12h",
    "rainfall_24h",
    "rainfall_72h",

    # Rainfall intensity
    "rainfall_max_6h",
    "rainfall_max_24h",

    # Rainfall anomaly
    "rainfall_baseline",
    "rainfall_std",
    "rainfall_anomaly",

    # Weather stress
    "humidity_stress",
    "wind_stress",

    # Cyclical temporal features
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "month_sin",
    "month_cos",

    # Metadata / seasonal features
    "year",
    "monsoon",

    # Rainfall event indicators
    "heavy_rain_flag",
    "extreme_rain_flag",

    # Additional spatiotemporal features
    "rainfall_3h_previous",
    "rainfall_change_3h",
    "rain_persistence_6h",
    "rain_humidity_interaction",]
# ============================================================
# WEATHER NODE COMPLETENESS
# ============================================================

print("\n[4/9] Checking weather-node completeness...")

weather_counts = (
    weather
    .groupby("grid_id")
    .size()
)

expected_hours = 17544

incomplete_weather = weather_counts[
    weather_counts != expected_hours
]

if len(incomplete_weather) > 0:

    print(
        "\n⚠ Incomplete weather nodes detected:"
    )

    print(incomplete_weather)

    raise ValueError(
        "\nWeather dataset contains incomplete nodes. "
        "Phase 2C stopped to prevent corrupted integration."
    )

print(
    f"✓ All {len(weather_counts)} weather nodes contain "
    f"{expected_hours} hourly records."
)


# ============================================================
# PREPARE NODE COORDINATES
# ============================================================

print("\n[5/9] Building nearest weather-node mapping...")


spatial_coords = (
    spatial[
        [
            "grid_id",
            "latitude",
            "longitude"
        ]
    ]
    .drop_duplicates("grid_id")
    .reset_index(drop=True)
)

weather_coords = (
    weather[
        [
            "grid_id",
            "latitude",
            "longitude"
        ]
    ]
    .drop_duplicates("grid_id")
    .reset_index(drop=True)
)


# ============================================================
# HAVERSINE DISTANCE
# ============================================================

def haversine_km(
    lat1,
    lon1,
    lat2,
    lon2
):

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)

    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        +
        np.cos(lat1)
        *
        np.cos(lat2)
        *
        np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arcsin(
        np.sqrt(a)
    )

    earth_radius = 6371.0

    return earth_radius * c


# ============================================================
# FIND NEAREST WEATHER NODE
# ============================================================

mapping_rows = []

for _, s in spatial_coords.iterrows():

    distances = haversine_km(
        s["latitude"],
        s["longitude"],
        weather_coords["latitude"].values,
        weather_coords["longitude"].values
    )

    nearest_index = np.argmin(distances)

    nearest = weather_coords.iloc[
        nearest_index
    ]

    distance = distances[
        nearest_index
    ]

    mapping_rows.append(
        {
            "spatial_grid_id":
                s["grid_id"],

            "weather_grid_id":
                nearest["grid_id"],

            "weather_distance_km":
                float(distance)
        }
    )


mapping = pd.DataFrame(
    mapping_rows
)


print(
    "Spatial nodes mapped:",
    mapping["spatial_grid_id"].nunique()
)

print(
    "Weather nodes used:",
    mapping["weather_grid_id"].nunique()
)

print(
    "Maximum mapping distance:",
    round(
        mapping["weather_distance_km"].max(),
        3
    ),
    "km"
)

print(
    "Average mapping distance:",
    round(
        mapping["weather_distance_km"].mean(),
        3
    ),
    "km"
)


# ============================================================
# VERIFY EVERY SPATIAL NODE HAS A MAPPING
# ============================================================

if (
    mapping["spatial_grid_id"].nunique()
    !=
    spatial["grid_id"].nunique()
):

    raise RuntimeError(
        "Not every spatial node received a weather mapping."
    )

print(
    "✓ All 140 spatial nodes mapped successfully"
)


# ============================================================
# BUILD WEATHER TIME-SERIES INDEX
# ============================================================

print(
    "\n[6/9] Applying weather mapping..."
)


# Rename weather node ID so it cannot collide
# with the final spatial node ID.

weather_for_merge = weather.rename(
    columns={
        "grid_id":
            "weather_grid_id"
    }
)


# Merge mapping onto complete weather data.
#
# Each spatial node receives the complete
# time-series of its nearest weather node.

integrated = mapping.merge(
    weather_for_merge,
    on="weather_grid_id",
    how="left",
    validate="many_to_many"
)


print(
    "Integrated weather rows:",
    len(integrated)
)

print(
    "Expected rows:",
    spatial["grid_id"].nunique()
    *
    expected_hours
)


# ============================================================
# VERIFY WEATHER INTEGRATION
# ============================================================

expected_rows = (
    spatial["grid_id"].nunique()
    *
    expected_hours
)

if len(integrated) != expected_rows:

    raise RuntimeError(
        f"\nIncorrect integrated row count.\n"
        f"Expected: {expected_rows}\n"
        f"Got: {len(integrated)}"
    )


# Rename spatial ID to final grid_id

integrated = integrated.rename(
    columns={
        "spatial_grid_id":
            "grid_id"
    }
)


# ============================================================
# REMOVE WEATHER NODE ID
# ============================================================

# Keep weather_grid_id temporarily because it is useful
# for validation and traceability.

print(
    "Spatial nodes represented:",
    integrated["grid_id"].nunique()
)


# ============================================================
# MERGE STATIC SPATIAL FEATURES
# ============================================================

print(
    "\n[7/9] Combining static spatial features..."
)


static_columns = [
    "grid_id",
    "latitude",
    "longitude",
    "elevation_m",
    "slope_degrees",
    "distance_to_water_m",
    "flood_probability",
    "risk_level",
    "elevation_risk",
    "water_risk",
    "resource_demand",
    "resources_required",
    "region_id",
    "allocated_resources",
    "operational_priority",
    "priority_rank",
]


static = spatial[
    static_columns
].copy()


# Rename coordinate columns to avoid collision
static = static.rename(
    columns={
        "latitude":
            "latitude_static",

        "longitude":
            "longitude_static"
    }
)


integrated = integrated.merge(
    static,
    on="grid_id",
    how="left",
    validate="many_to_one"
)


# ============================================================
# SPATIAL-WEATHER INTERACTION FEATURES
# ============================================================

print(
    "\nCreating spatial-weather interaction features..."
)


# ------------------------------------------------------------
# Elevation × rainfall
# ------------------------------------------------------------

integrated[
    "elevation_rainfall_risk"
] = (
    integrated["rainfall_24h"]
    *
    (1.0 / (
        integrated["elevation_m"]
        + 1.0
    ))
)


# ------------------------------------------------------------
# Water proximity × rainfall
# ------------------------------------------------------------

integrated[
    "water_rainfall_risk"
] = (
    integrated["rainfall_24h"]
    /
    (
        integrated["distance_to_water_m"]
        + 1.0
    )
)


# ------------------------------------------------------------
# Environmental stress
# ------------------------------------------------------------

integrated[
    "environmental_stress"
] = (
    0.5
    *
    integrated["humidity_stress"]
    +
    0.5
    *
    integrated["wind_stress"]
)


# ------------------------------------------------------------
# Dynamic rainfall risk
# ------------------------------------------------------------

integrated[
    "dynamic_rainfall_risk"
] = (
    0.35
    *
    integrated["rainfall_24h"]
    /
    max(
        integrated["rainfall_24h"].max(),
        1
    )
    +
    0.25
    *
    integrated["rainfall_72h"]
    /
    max(
        integrated["rainfall_72h"].max(),
        1
    )
    +
    0.20
    *
    integrated["rainfall_anomaly"].clip(
        lower=0
    )
    /
    max(
        integrated["rainfall_anomaly"].clip(
            lower=0
        ).max(),
        1
    )
    +
    0.20
    *
    integrated["rain_persistence_6h"]
)


# ------------------------------------------------------------
# Dynamic flood signal
# ------------------------------------------------------------

integrated[
    "dynamic_flood_signal"
] = (
    0.50
    *
    integrated["dynamic_rainfall_risk"]
    +
    0.25
    *
    integrated["flood_probability"]
    +
    0.15
    *
    integrated["water_risk"]
    +
    0.10
    *
    integrated["elevation_risk"]
)


# ------------------------------------------------------------
# Dynamic resource demand
# ------------------------------------------------------------

integrated[
    "dynamic_resource_demand"
] = (
    integrated["resource_demand"]
    *
    (
        1
        +
        integrated["dynamic_flood_signal"]
    )
)


# ------------------------------------------------------------
# Resource gap
# ------------------------------------------------------------

integrated[
    "resource_gap"
] = (
    integrated["dynamic_resource_demand"]
    -
    integrated["allocated_resources"]
)


# ------------------------------------------------------------
# Resource shortage flag
# ------------------------------------------------------------

integrated[
    "resource_shortage_flag"
] = (
    integrated["resource_gap"] > 0
).astype(int)


# ============================================================
# CLEANUP
# ============================================================

print(
    "\n[8/9] Cleaning integrated dataset..."
)


integrated = integrated.replace(
    [np.inf, -np.inf],
    np.nan
)


numeric_columns = integrated.select_dtypes(
    include=[np.number]
).columns


for col in numeric_columns:

    integrated[col] = (
        integrated[col]
        .fillna(0)
    )


# ============================================================
# SORT FINAL DATASET
# ============================================================

integrated = integrated.sort_values(
    [
        "grid_id",
        "time"
    ]
).reset_index(
    drop=True
)


# ============================================================
# FINAL VALIDATION BEFORE SAVE
# ============================================================

print(
    "\n[9/9] Final validation..."
)


final_nodes = (
    integrated["grid_id"]
    .nunique()
)

final_rows = len(
    integrated
)

missing_values = (
    integrated.isnull()
    .sum()
    .sum()
)

duplicate_rows = (
    integrated
    .duplicated(
        subset=[
            "grid_id",
            "time"
        ]
    )
    .sum()
)


print(
    "Final rows:",
    final_rows
)

print(
    "Expected rows:",
    expected_rows
)

print(
    "Final nodes:",
    final_nodes
)

print(
    "Expected nodes:",
    spatial["grid_id"].nunique()
)

print(
    "Missing values:",
    missing_values
)

print(
    "Duplicate node-time records:",
    duplicate_rows
)


# ============================================================
# HARD VALIDATION
# ============================================================

if final_rows != expected_rows:

    raise RuntimeError(
        "FINAL VALIDATION FAILED: incorrect row count."
    )


if final_nodes != spatial["grid_id"].nunique():

    raise RuntimeError(
        "FINAL VALIDATION FAILED: spatial nodes were lost."
    )


if missing_values != 0:

    raise RuntimeError(
        "FINAL VALIDATION FAILED: missing values remain."
    )


if duplicate_rows != 0:

    raise RuntimeError(
        "FINAL VALIDATION FAILED: duplicate node-time records."
    )


# Every node must have exactly 17,544 rows

rows_per_node = (
    integrated
    .groupby("grid_id")
    .size()
)


bad_nodes = rows_per_node[
    rows_per_node != expected_hours
]


if len(bad_nodes) > 0:

    print(
        "\nBad nodes:"
    )

    print(
        bad_nodes
    )

    raise RuntimeError(
        "FINAL VALIDATION FAILED: one or more nodes "
        "do not contain the complete time series."
    )


# ============================================================
# SAVE
# ============================================================

print(
    "\nSaving integrated dataset..."
)


integrated.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FILE SIZE
# ============================================================

file_size_mb = (
    OUTPUT_FILE.stat().st_size
    /
    (1024 * 1024)
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PHASE 2C COMPLETE")
print("=" * 70)

print(
    "\nOutput:"
)

print(
    OUTPUT_FILE
)

print(
    "\nRows:",
    final_rows
)

print(
    "Nodes:",
    final_nodes
)

print(
    "Hours per node:",
    expected_hours
)

print(
    "\nTime range:"
)

print(
    integrated["time"].min(),
    "to",
    integrated["time"].max()
)

print(
    "\nWeather nodes used:",
    integrated["weather_grid_id"].nunique()
)

print(
    "Maximum weather mapping distance:",
    round(
        mapping["weather_distance_km"].max(),
        3
    ),
    "km"
)

print(
    "Average weather mapping distance:",
    round(
        mapping["weather_distance_km"].mean(),
        3
    ),
    "km"
)

print(
    "\nMissing values:",
    missing_values
)

print(
    "Duplicate node-time records:",
    duplicate_rows
)

print(
    "\nFile size:",
    f"{file_size_mb:.2f} MB"
)

print(
    "\n✓ ALL 140 SPATIAL NODES SUCCESSFULLY INTEGRATED"
)

print(
    "✓ COMPLETE 2024–2025 HOURLY TIME SERIES"
)

print(
    "✓ PHASE 2C FINISHED SUCCESSFULLY"
)

print("=" * 70)