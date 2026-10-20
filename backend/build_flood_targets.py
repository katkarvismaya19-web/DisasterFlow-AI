# ============================================================
# DISASTERFLOW AI — PHASE 3
# FLOOD TARGET ENGINEERING
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np


print("=" * 70)
print("DISASTERFLOW AI — PHASE 3")
print("Flood Target Engineering")
print("=" * 70)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "disasterflow_spatiotemporal_nodes.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "disasterflow_flood_targets.csv"
)


# ============================================================
# CHECK INPUT
# ============================================================

print("\n[1/8] Checking input dataset...")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"\nInput file not found:\n{INPUT_FILE}"
    )

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print("✓ Integrated dataset found")


# ============================================================
# LOAD DATA
# ============================================================

print("\n[2/8] Loading integrated dataset...")

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=["time"]
)

print("Rows:", len(df))
print("Nodes:", df["grid_id"].nunique())

print(
    "Time range:",
    df["time"].min(),
    "to",
    df["time"].max()
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "grid_id",
    "time",
    "precipitation",
    "rainfall_1h",
    "rainfall_3h",
    "rainfall_6h",
    "rainfall_12h",
    "rainfall_24h",
    "rainfall_72h",
    "rainfall_max_6h",
    "rainfall_max_24h",
    "rainfall_anomaly",
    "relative_humidity_2m",
    "distance_to_water_m",
    "elevation_m",
    "slope_degrees",
    "flood_probability"
]

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:

    raise ValueError(
        "\nMissing required columns:\n"
        + "\n".join(missing_columns)
    )

print("✓ Required columns available")


# ============================================================
# SORT
# ============================================================

print("\n[3/8] Sorting node-time data...")

df = (
    df.sort_values(
        ["grid_id", "time"]
    )
    .reset_index(drop=True)
)

print("✓ Data sorted")


# ============================================================
# NUMERIC CONVERSION
# ============================================================

print("\n[4/8] Preparing flood predictors...")

numeric_columns = [
    "precipitation",
    "rainfall_1h",
    "rainfall_3h",
    "rainfall_6h",
    "rainfall_12h",
    "rainfall_24h",
    "rainfall_72h",
    "rainfall_max_6h",
    "rainfall_max_24h",
    "rainfall_anomaly",
    "relative_humidity_2m",
    "distance_to_water_m",
    "elevation_m",
    "slope_degrees",
    "flood_probability"
]

for col in numeric_columns:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

    df[col] = df[col].fillna(0)


# ============================================================
# TEMPORAL RAINFALL SIGNALS
# ============================================================

print("Creating temporal rainfall signals...")


# Previous 3-hour accumulated rainfall

df["rainfall_3h_lag"] = (
    df.groupby("grid_id")["rainfall_3h"]
      .shift(3)
      .fillna(0)
)


# Previous 6-hour accumulated rainfall

df["rainfall_6h_lag"] = (
    df.groupby("grid_id")["rainfall_6h"]
      .shift(6)
      .fillna(0)
)


# Previous 24-hour accumulated rainfall

df["rainfall_24h_lag"] = (
    df.groupby("grid_id")["rainfall_24h"]
      .shift(24)
      .fillna(0)
)


# Rainfall change

df["rainfall_3h_change"] = (
    df["rainfall_3h"]
    - df["rainfall_3h_lag"]
)


# ============================================================
# RAINFALL INTENSITY SIGNAL
# ============================================================

print("Creating rainfall intensity score...")


df["rainfall_intensity_score"] = (
    0.30 * np.clip(
        df["rainfall_1h"] / 20.0,
        0,
        1
    )
    +
    0.25 * np.clip(
        df["rainfall_3h"] / 50.0,
        0,
        1
    )
    +
    0.20 * np.clip(
        df["rainfall_6h"] / 75.0,
        0,
        1
    )
    +
    0.15 * np.clip(
        df["rainfall_24h"] / 150.0,
        0,
        1
    )
    +
    0.10 * np.clip(
        df["rainfall_72h"] / 300.0,
        0,
        1
    )
)


# ============================================================
# WATER PROXIMITY SCORE
# ============================================================

print("Creating water proximity score...")


# Closer to water = higher flood vulnerability

df["water_proximity_score"] = (
    np.exp(
        -df["distance_to_water_m"] / 1500.0
    )
)

df["water_proximity_score"] = np.clip(
    df["water_proximity_score"],
    0,
    1
)


# ============================================================
# ELEVATION VULNERABILITY
# ============================================================

print("Creating elevation vulnerability score...")


elevation_min = df["elevation_m"].min()
elevation_max = df["elevation_m"].max()

if elevation_max > elevation_min:

    elevation_norm = (
        (df["elevation_m"] - elevation_min)
        /
        (elevation_max - elevation_min)
    )

else:

    elevation_norm = pd.Series(
        0.5,
        index=df.index
    )


# Lower elevation = higher vulnerability

df["low_elevation_score"] = (
    1 - elevation_norm
)

df["low_elevation_score"] = np.clip(
    df["low_elevation_score"],
    0,
    1
)


# ============================================================
# SLOPE SCORE
# ============================================================

print("Creating slope vulnerability score...")


# Lower slope generally allows water to accumulate more easily.

slope_norm = np.clip(
    df["slope_degrees"] / 10.0,
    0,
    1
)

df["low_slope_score"] = (
    1 - slope_norm
)


# ============================================================
# HUMIDITY SCORE
# ============================================================

df["humidity_score"] = (
    df["relative_humidity_2m"]
    / 100.0
)

df["humidity_score"] = np.clip(
    df["humidity_score"],
    0,
    1
)


# ============================================================
# RAINFALL ANOMALY SCORE
# ============================================================

df["rainfall_anomaly_score"] = np.clip(
    (df["rainfall_anomaly"] + 2.0) / 4.0,
    0,
    1
)


# ============================================================
# FLOOD RISK SCORE
# ============================================================

print("\n[5/8] Building flood target score...")


df["flood_risk_score"] = (

    0.40
    * df["rainfall_intensity_score"]

    +

    0.20
    * df["water_proximity_score"]

    +

    0.15
    * df["low_elevation_score"]

    +

    0.10
    * df["low_slope_score"]

    +

    0.10
    * df["humidity_score"]

    +

    0.05
    * df["rainfall_anomaly_score"]
)


# ============================================================
# INCLUDE EXISTING FLOOD PROBABILITY
# ============================================================

print("Integrating existing flood probability...")


df["flood_target_score"] = (

    0.70 * df["flood_risk_score"]

    +

    0.30
    * np.clip(
        df["flood_probability"],
        0,
        1
    )
)


df["flood_target_score"] = np.clip(
    df["flood_target_score"],
    0,
    1
)


# ============================================================
# FLOOD TARGET CLASS
# ============================================================

print("\n[6/8] Creating flood target labels...")


def classify_flood_risk(score):

    if score < 0.25:
        return "LOW"

    elif score < 0.50:
        return "MODERATE"

    elif score < 0.75:
        return "HIGH"

    else:
        return "EXTREME"


df["flood_target"] = (
    df["flood_target_score"]
    .apply(classify_flood_risk)
)


# ============================================================
# BINARY FLOOD TARGET
# ============================================================

df["flood_event"] = (
    df["flood_target_score"] >= 0.50
).astype(int)


# ============================================================
# EXTREME FLOOD EVENT
# ============================================================

df["extreme_flood_event"] = (
    df["flood_target_score"] >= 0.75
).astype(int)


# ============================================================
# FUTURE FLOOD TARGET
# ============================================================

print("Creating future flood prediction targets...")


# This is important for ML:
# Predict flood risk 6 hours into the future.

df["future_flood_score_6h"] = (
    df.groupby("grid_id")["flood_target_score"]
      .shift(-6)
)


df["future_flood_event_6h"] = (
    df["future_flood_score_6h"]
    >= 0.50
).astype(int)


# 12-hour prediction target

df["future_flood_score_12h"] = (
    df.groupby("grid_id")["flood_target_score"]
      .shift(-12)
)


df["future_flood_event_12h"] = (
    df["future_flood_score_12h"]
    >= 0.50
).astype(int)


# 24-hour prediction target

df["future_flood_score_24h"] = (
    df.groupby("grid_id")["flood_target_score"]
      .shift(-24)
)


df["future_flood_event_24h"] = (
    df["future_flood_score_24h"]
    >= 0.50
).astype(int)


# ============================================================
# REMOVE LAST FUTURE-UNKNOWN HOURS
# ============================================================

print("\nRemoving rows without future target...")


df = df[
    df["future_flood_score_24h"].notna()
].copy()


# ============================================================
# FINAL CLEANUP
# ============================================================

print("\n[7/8] Cleaning target dataset...")


df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

numeric_columns = df.select_dtypes(
    include=[np.number]
).columns

for col in numeric_columns:

    df[col] = df[col].fillna(0)


# ============================================================
# VALIDATION
# ============================================================

print("\nValidating flood targets...")


print(
    "Rows:",
    len(df)
)

print(
    "Nodes:",
    df["grid_id"].nunique()
)

print(
    "Missing values:",
    df.isnull().sum().sum()
)

print(
    "Duplicate node-time:",
    df.duplicated(
        ["grid_id", "time"]
    ).sum()
)


print("\nFlood target distribution:")

print(
    df["flood_target"]
    .value_counts()
)


print("\nFlood event distribution:")

print(
    df["flood_event"]
    .value_counts()
)


print("\nFuture 6h flood events:")

print(
    df["future_flood_event_6h"]
    .value_counts()
)


print("\nFuture 12h flood events:")

print(
    df["future_flood_event_12h"]
    .value_counts()
)


print("\nFuture 24h flood events:")

print(
    df["future_flood_event_24h"]
    .value_counts()
)


# ============================================================
# SAVE
# ============================================================

print("\n[8/8] Saving flood target dataset...")

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PHASE 3 COMPLETE")
print("=" * 70)

print("\nOutput:")
print(OUTPUT_FILE)

print("\nRows:", len(df))

print(
    "Nodes:",
    df["grid_id"].nunique()
)

print(
    "Time range:",
    df["time"].min(),
    "to",
    df["time"].max()
)

print(
    "Missing values:",
    df.isnull().sum().sum()
)

print(
    "Duplicate node-time:",
    df.duplicated(
        ["grid_id", "time"]
    ).sum()
)

print("\nFlood target columns:")

target_columns = [
    "flood_risk_score",
    "flood_target_score",
    "flood_target",
    "flood_event",
    "extreme_flood_event",
    "future_flood_score_6h",
    "future_flood_event_6h",
    "future_flood_score_12h",
    "future_flood_event_12h",
    "future_flood_score_24h",
    "future_flood_event_24h"
]

for col in target_columns:
    print("  ✓", col)


size_mb = (
    OUTPUT_FILE.stat().st_size
    / (1024 * 1024)
)

print(
    f"\nFile size: {size_mb:.2f} MB"
)

print("\n✓ Flood prediction targets created successfully.")
print("✓ 6-hour future target created.")
print("✓ 12-hour future target created.")
print("✓ 24-hour future target created.")
print("✓ Phase 3 finished successfully.")

print("=" * 70)