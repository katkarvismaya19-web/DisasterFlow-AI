# ============================================================
# DISASTERFLOW AI — PHASE 4A
# ML DATASET PREPARATION
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np


print("=" * 70)
print("DISASTERFLOW AI — PHASE 4A")
print("ML Dataset Preparation")
print("=" * 70)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "disasterflow_flood_targets.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TRAIN_FILE = OUTPUT_DIR / "ml_train.csv"
VAL_FILE = OUTPUT_DIR / "ml_validation.csv"
TEST_FILE = OUTPUT_DIR / "ml_test.csv"


# ============================================================
# CHECK INPUT
# ============================================================

print("\n[1/8] Checking input dataset...")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"\nFlood target dataset not found:\n{INPUT_FILE}"
    )

print("✓ Flood target dataset found")


# ============================================================
# LOAD DATA
# ============================================================

print("\n[2/8] Loading flood target dataset...")

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=["time"]
)

print("Rows:", len(df))
print("Nodes:", df["grid_id"].nunique())
print(
    "Time:",
    df["time"].min(),
    "to",
    df["time"].max()
)


# ============================================================
# SORT
# ============================================================

print("\n[3/8] Sorting chronological data...")

df = df.sort_values(
    ["time", "grid_id"]
).reset_index(drop=True)

print("✓ Data sorted chronologically")


# ============================================================
# TARGET
# ============================================================

print("\n[4/8] Preparing prediction target...")

TARGET = "future_flood_event_6h"

if TARGET not in df.columns:
    raise ValueError(
        f"Target column missing: {TARGET}"
    )

print("Prediction target:", TARGET)

print("\nTarget distribution:")

print(
    df[TARGET]
    .value_counts()
    .sort_index()
)

print("\nTarget percentage:")

print(
    (
        df[TARGET]
        .value_counts(normalize=True)
        .sort_index()
        * 100
    ).round(4)
)


# ============================================================
# FEATURE SELECTION
# ============================================================

print("\n[5/8] Selecting ML features...")


feature_columns = [

    # --------------------------------------------------------
    # WEATHER
    # --------------------------------------------------------

    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "wind_speed_10m",

    # --------------------------------------------------------
    # RAINFALL ACCUMULATION
    # --------------------------------------------------------

    "rainfall_1h",
    "rainfall_3h",
    "rainfall_6h",
    "rainfall_12h",
    "rainfall_24h",
    "rainfall_72h",

    # --------------------------------------------------------
    # RAINFALL INTENSITY
    # --------------------------------------------------------

    "rainfall_max_6h",
    "rainfall_max_24h",

    # --------------------------------------------------------
    # RAINFALL STATISTICS
    # --------------------------------------------------------

    "rainfall_baseline",
    "rainfall_std",
    "rainfall_anomaly",

    # --------------------------------------------------------
    # TEMPORAL RAINFALL DYNAMICS
    # --------------------------------------------------------

    "rainfall_3h_previous",
    "rainfall_change_3h",
    "rain_persistence_6h",

    # --------------------------------------------------------
    # WEATHER INTERACTION
    # --------------------------------------------------------

    "humidity_stress",
    "wind_stress",
    "rain_humidity_interaction",

    # --------------------------------------------------------
    # TERRAIN
    # --------------------------------------------------------

    "elevation_m",
    "slope_degrees",
    "distance_to_water_m",

    # --------------------------------------------------------
    # STATIC FLOOD RISK
    # --------------------------------------------------------

    "flood_probability",
    "elevation_risk",
    "water_risk",

    # --------------------------------------------------------
    # DYNAMIC FLOOD SIGNAL
    # --------------------------------------------------------

    "dynamic_rainfall_risk",
    "dynamic_flood_signal",

    # --------------------------------------------------------
    # ENVIRONMENTAL STRESS
    # --------------------------------------------------------

    "environmental_stress",

    # --------------------------------------------------------
    # RESOURCE / VULNERABILITY
    # --------------------------------------------------------

    "resource_demand",

    # --------------------------------------------------------
    # SPATIAL LOCATION
    # --------------------------------------------------------

    "latitude_static",
    "longitude_static",

    # --------------------------------------------------------
    # TIME
    # --------------------------------------------------------

    "hour",
    "day_of_year",
    "month",
    "monsoon"
]


# ============================================================
# VERIFY FEATURES
# ============================================================

missing_features = [
    col
    for col in feature_columns
    if col not in df.columns
]

if missing_features:

    print("\n❌ Missing features:")

    for col in missing_features:
        print("  -", col)

    raise ValueError(
        "Required ML features are missing."
    )

print(
    "\n✓ All",
    len(feature_columns),
    "ML features available"
)


# ============================================================
# BUILD ML DATAFRAME
# ============================================================

ml_df = df[
    [
        "grid_id",
        "time",
        *feature_columns,
        TARGET
    ]
].copy()


# ============================================================
# CLEAN NUMERIC VALUES
# ============================================================

print("\nCleaning numeric values...")

numeric_features = [
    col
    for col in feature_columns
    if col in ml_df.columns
]

for col in numeric_features:

    ml_df[col] = pd.to_numeric(
        ml_df[col],
        errors="coerce"
    )


ml_df[numeric_features] = (
    ml_df[numeric_features]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
)


# Fill numeric missing values using median

for col in numeric_features:

    if ml_df[col].isna().any():

        median_value = ml_df[col].median()

        ml_df[col] = (
            ml_df[col]
            .fillna(median_value)
        )


ml_df[TARGET] = (
    pd.to_numeric(
        ml_df[TARGET],
        errors="coerce"
    )
    .fillna(0)
    .astype(int)
)


# ============================================================
# CHECK DATA
# ============================================================

print("\nData quality:")

print(
    "Missing values:",
    ml_df.isnull().sum().sum()
)

print(
    "Duplicate node-time:",
    ml_df.duplicated(
        ["grid_id", "time"]
    ).sum()
)


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

print("\n[6/8] Creating chronological train/validation/test split...")


# Important:
# DO NOT randomly split this dataset.
#
# Flood forecasting must respect time.
#
# 70% -> training
# 15% -> validation
# 15% -> testing


unique_times = (
    ml_df["time"]
    .sort_values()
    .unique()
)

n_times = len(unique_times)

train_end = int(n_times * 0.70)

val_end = int(n_times * 0.85)

train_end_time = unique_times[train_end - 1]

val_end_time = unique_times[val_end - 1]


print(
    "\nTrain ends:",
    train_end_time
)

print(
    "Validation ends:",
    val_end_time
)


train_df = ml_df[
    ml_df["time"] <= train_end_time
].copy()


val_df = ml_df[
    (ml_df["time"] > train_end_time)
    &
    (ml_df["time"] <= val_end_time)
].copy()


test_df = ml_df[
    ml_df["time"] > val_end_time
].copy()


# ============================================================
# SPLIT VALIDATION
# ============================================================

print("\nSplit sizes:")

print(
    "Train:",
    len(train_df)
)

print(
    "Validation:",
    len(val_df)
)

print(
    "Test:",
    len(test_df)
)


# ============================================================
# TARGET DISTRIBUTION PER SPLIT
# ============================================================

print("\nTrain target:")

print(
    train_df[TARGET]
    .value_counts()
)


print("\nValidation target:")

print(
    val_df[TARGET]
    .value_counts()
)


print("\nTest target:")

print(
    test_df[TARGET]
    .value_counts()
)


# ============================================================
# EVENT RATES
# ============================================================

print("\nFlood event rates:")

for name, data in [
    ("Train", train_df),
    ("Validation", val_df),
    ("Test", test_df)
]:

    rate = data[TARGET].mean() * 100

    print(
        f"{name}: {rate:.4f}%"
    )


# ============================================================
# SAVE
# ============================================================

print("\n[7/8] Saving ML datasets...")


train_df.to_csv(
    TRAIN_FILE,
    index=False
)

print(
    "✓ Train saved:",
    TRAIN_FILE
)


val_df.to_csv(
    VAL_FILE,
    index=False
)

print(
    "✓ Validation saved:",
    VAL_FILE
)


test_df.to_csv(
    TEST_FILE,
    index=False
)

print(
    "✓ Test saved:",
    TEST_FILE
)


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n[8/8] Final validation...")


print(
    "\nTrain time:",
    train_df["time"].min(),
    "to",
    train_df["time"].max()
)

print(
    "Validation time:",
    val_df["time"].min(),
    "to",
    val_df["time"].max()
)

print(
    "Test time:",
    test_df["time"].min(),
    "to",
    test_df["time"].max()
)


# Make sure time periods don't overlap

assert (
    train_df["time"].max()
    <
    val_df["time"].min()
)

assert (
    val_df["time"].max()
    <
    test_df["time"].min()
)


print(
    "\n✓ No temporal leakage detected"
)


print(
    "✓ Missing values:",
    ml_df.isnull().sum().sum()
)

print(
    "✓ Feature count:",
    len(feature_columns)
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PHASE 4A COMPLETE")
print("=" * 70)

print("\nTarget:")
print("  future_flood_event_6h")

print("\nFeatures:", len(feature_columns))

print("\nDatasets:")

print(
    "  ✓",
    TRAIN_FILE.name
)

print(
    "  ✓",
    VAL_FILE.name
)

print(
    "  ✓",
    TEST_FILE.name
)

print("\nFinal sizes:")

print(
    "  Train:",
    len(train_df)
)

print(
    "  Validation:",
    len(val_df)
)

print(
    "  Test:",
    len(test_df)
)

print("\n✓ Phase 4A finished successfully.")
print("=" * 70)