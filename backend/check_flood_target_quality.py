# ============================================================
# DISASTERFLOW AI
# FLOOD TARGET QUALITY DIAGNOSTIC
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np


print("=" * 70)
print("DISASTERFLOW AI")
print("FLOOD TARGET QUALITY DIAGNOSTIC")
print("=" * 70)


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "disasterflow_flood_targets.csv"
)


if not FILE.exists():
    raise FileNotFoundError(FILE)


# ============================================================
# LOAD
# ============================================================

print("\n[1/7] Loading flood target dataset...")

df = pd.read_csv(
    FILE,
    parse_dates=["time"]
)

print("Rows:", len(df))
print("Nodes:", df["grid_id"].nunique())

print(
    "Time:",
    df["time"].min(),
    "->",
    df["time"].max()
)


# ============================================================
# TARGET SCORE DISTRIBUTION
# ============================================================

print("\n[2/7] TARGET SCORE DISTRIBUTION")
print("-" * 70)

score = pd.to_numeric(
    df["flood_target_score"],
    errors="coerce"
)

future_score = pd.to_numeric(
    df["future_flood_score_6h"],
    errors="coerce"
)

print("\nCurrent flood_target_score:")

print(
    score.describe(
        percentiles=[
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
            0.999
        ]
    )
)


print("\nFuture 6h flood score:")

print(
    future_score.describe(
        percentiles=[
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
            0.999
        ]
    )
)


# ============================================================
# THRESHOLD DISTRIBUTION
# ============================================================

print("\n[3/7] FLOOD THRESHOLD ANALYSIS")
print("-" * 70)

for threshold in [0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.75]:

    current_rate = (
        (score >= threshold).mean()
        * 100
    )

    future_rate = (
        (future_score >= threshold).mean()
        * 100
    )

    print(
        f"Threshold {threshold:.2f} | "
        f"Current: {current_rate:.4f}% | "
        f"Future 6h: {future_rate:.4f}%"
    )


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n[4/7] FLOOD CLASS DISTRIBUTION")
print("-" * 70)

print(
    df["flood_target"]
    .value_counts()
)

print("\nPercent:")

print(
    (
        df["flood_target"]
        .value_counts(normalize=True)
        * 100
    ).round(4)
)


# ============================================================
# COMPONENT CONTRIBUTION
# ============================================================

print("\n[5/7] TARGET COMPONENT ANALYSIS")
print("-" * 70)

components = [
    "rainfall_intensity_score",
    "water_proximity_score",
    "low_elevation_score",
    "low_slope_score",
    "humidity_score",
    "rainfall_anomaly_score",
    "flood_risk_score",
    "flood_probability"
]

for col in components:

    if col not in df.columns:
        print("Missing:", col)
        continue

    values = pd.to_numeric(
        df[col],
        errors="coerce"
    )

    print(
        f"\n{col}"
    )

    print(
        values.describe(
            percentiles=[
                0.50,
                0.75,
                0.90,
                0.95,
                0.99
            ]
        )
    )


# ============================================================
# CORRELATION WITH TARGET
# ============================================================

print("\n[6/7] TARGET CORRELATIONS")
print("-" * 70)

correlation_columns = [
    "rainfall_intensity_score",
    "water_proximity_score",
    "low_elevation_score",
    "low_slope_score",
    "humidity_score",
    "rainfall_anomaly_score",
    "flood_risk_score",
    "flood_probability",
    "flood_target_score"
]

available = [
    c for c in correlation_columns
    if c in df.columns
]

corr = (
    df[available]
    .corr()["flood_target_score"]
    .sort_values(
        ascending=False
    )
)

print(corr)


# ============================================================
# MONTHLY FUTURE FLOOD EVENTS
# ============================================================

print("\n[7/7] MONTHLY FUTURE FLOOD DISTRIBUTION")
print("-" * 70)

df["month_period"] = (
    df["time"]
    .dt.to_period("M")
)

monthly = (
    df.groupby("month_period")
      .agg(
          rows=("future_flood_event_6h", "size"),
          positive=("future_flood_event_6h", "sum")
      )
)

monthly["positive_rate_percent"] = (
    monthly["positive"]
    / monthly["rows"]
    * 100
)

print(
    monthly.to_string()
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)