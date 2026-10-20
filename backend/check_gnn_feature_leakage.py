
import os
import pandas as pd
import numpy as np

print("=" * 70)
print("DISASTERFLOW AI")
print("ST-GNN FEATURE / LEAKAGE DIAGNOSTIC")
print("=" * 70)

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

FILES = {
    "train": os.path.join(DATA_DIR, "ml_train.csv"),
    "validation": os.path.join(DATA_DIR, "ml_validation.csv"),
    "test": os.path.join(DATA_DIR, "ml_test.csv")
}

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

# ============================================================
# LOAD
# ============================================================

print("\n[1/6] Loading datasets...")

train = pd.read_csv(
    FILES["train"],
    parse_dates=["time"]
)

validation = pd.read_csv(
    FILES["validation"],
    parse_dates=["time"]
)

test = pd.read_csv(
    FILES["test"],
    parse_dates=["time"]
)

print("Train:", train.shape)
print("Validation:", validation.shape)
print("Test:", test.shape)

# ============================================================
# TARGET CHECK
# ============================================================

print("\n[2/6] Target information")
print("-" * 70)

print("Target:", TARGET)

for name, df in [
    ("TRAIN", train),
    ("VALIDATION", validation),
    ("TEST", test)
]:

    print(
        f"{name}:",
        df[TARGET].value_counts().to_dict()
    )

# ============================================================
# FEATURE CORRELATION
# ============================================================

print("\n[3/6] Feature correlation with future target")
print("-" * 70)

corr = train[FEATURES + [TARGET]].corr(
    numeric_only=True
)[TARGET].drop(TARGET).sort_values(
    key=lambda x: abs(x),
    ascending=False
)

print(corr.to_string())

# ============================================================
# UNIQUE VALUES
# ============================================================

print("\n[4/6] Feature variability")
print("-" * 70)

for feature in FEATURES:

    n_unique = train[feature].nunique()
    std = train[feature].std()

    print(
        f"{feature:32s} "
        f"unique={n_unique:8d} "
        f"std={std:.6f}"
    )

# ============================================================
# SUSPICIOUS FEATURES
# ============================================================

print("\n[5/6] Suspicious / leakage-sensitive features")
print("-" * 70)

suspicious = [
    "flood_probability",
    "dynamic_flood_signal",
    "resource_demand",
    "environmental_stress",
    "dynamic_rainfall_risk",
    "elevation_risk",
    "water_risk"
]

for feature in suspicious:

    if feature in train.columns:

        print(
            f"\n{feature}"
        )

        print(
            "  min:",
            train[feature].min()
        )

        print(
            "  max:",
            train[feature].max()
        )

        print(
            "  mean:",
            train[feature].mean()
        )

        print(
            "  unique:",
            train[feature].nunique()
        )

# ============================================================
# TEMPORAL ORDER
# ============================================================

print("\n[6/6] Temporal availability check")
print("-" * 70)

for name, df in [
    ("TRAIN", train),
    ("VALIDATION", validation),
    ("TEST", test)
]:

    print(
        f"{name}: "
        f"{df.time.min()} -> {df.time.max()}"
    )

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)
