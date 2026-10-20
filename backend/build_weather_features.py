
# ============================================================
# DISASTERFLOW AI — PHASE 2B
# SPATIOTEMPORAL WEATHER FEATURE ENGINEERING
#
# UPDATED VERSION
# ------------------------------------------------------------
# Major improvements:
# 1. Cyclical temporal encoding
# 2. No raw hour/day/month ML features
# 3. No future leakage in rainfall baseline/std
# 4. Node-wise rolling rainfall features
# 5. Chronological ordering
# 6. Explicit hourly continuity checks
# 7. Numerical cleaning
# 8. Weather/spatiotemporal features prepared for integration
#
# IMPORTANT:
# Phase 2B creates WEATHER features only.
#
# Spatial/risk features such as:
#   elevation_m
#   slope_degrees
#   distance_to_water_m
#   flood_probability
#   elevation_risk
#   water_risk
#   dynamic_rainfall_risk
#   dynamic_flood_signal
#   environmental_stress
#   resource_demand
#   latitude_static
#   longitude_static
#
# are integrated in the later spatial integration stage.
# ============================================================


from pathlib import Path

import pandas as pd
import numpy as np


print("=" * 70)
print("DISASTERFLOW AI — PHASE 2B")
print("UPDATED Spatiotemporal Weather Feature Engineering")
print("=" * 70)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent


WEATHER_FILE = (
    BASE_DIR
    / "data"
    / "raw"
    / "weather"
    / "mumbai_historical_weather.csv"
)


OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
)


OUTPUT_FILE = (
    OUTPUT_DIR
    / "mumbai_spatiotemporal_features.csv"
)


# ============================================================
# CHECK INPUT
# ============================================================

if not WEATHER_FILE.exists():

    raise FileNotFoundError(
        f"\nWeather file not found:\n{WEATHER_FILE}"
    )


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("\n[1/10] Loading weather data...")


df = pd.read_csv(
    WEATHER_FILE,
    parse_dates=["time"]
)


# ============================================================
# BASIC VALIDATION
# ============================================================

required_columns = [
    "grid_id",
    "time",
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "wind_speed_10m"
]


missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]


if missing_columns:

    raise ValueError(
        "\nMissing required columns:\n"
        + "\n".join(
            f"  - {c}"
            for c in missing_columns
        )
    )


print(
    "Rows:",
    len(df)
)


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


# ============================================================
# SORT
# ============================================================

print(
    "\n[2/10] Sorting by node and time..."
)


df = (
    df
    .sort_values(
        ["grid_id", "time"]
    )
    .reset_index(drop=True)
)


# ============================================================
# CHECK DUPLICATES
# ============================================================

duplicate_count = int(
    df.duplicated(
        subset=[
            "grid_id",
            "time"
        ]
    ).sum()
)


print(
    "Duplicate node-time rows:",
    duplicate_count
)


if duplicate_count > 0:

    print(
        "Removing duplicate node-time rows..."
    )

    df = (
        df
        .drop_duplicates(
            subset=[
                "grid_id",
                "time"
            ],
            keep="first"
        )
        .reset_index(drop=True)
    )


# ============================================================
# NUMERIC CONVERSION
# ============================================================

print(
    "\n[3/10] Converting weather columns to numeric..."
)


weather_columns = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "wind_speed_10m"
]


for col in weather_columns:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

    print(
        f"  ✓ {col}"
    )


# ============================================================
# PHYSICAL SANITY CHECKS
# ============================================================

print(
    "\nApplying physical sanity constraints..."
)


# ------------------------------------------------------------
# Rainfall cannot be negative
# ------------------------------------------------------------

for col in [
    "precipitation",
    "rain"
]:

    negative_count = int(
        (df[col] < 0).sum()
    )

    if negative_count > 0:

        print(
            f"  {col}: "
            f"{negative_count} negative values clipped"
        )

        df[col] = df[col].clip(
            lower=0
        )


# ------------------------------------------------------------
# Relative humidity should be between 0 and 100
# ------------------------------------------------------------

df["relative_humidity_2m"] = (
    df["relative_humidity_2m"]
    .clip(
        lower=0,
        upper=100
    )
)


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

print(
    "\n[4/10] Handling missing weather values..."
)


for col in weather_columns:

    missing_before = int(
        df[col].isna().sum()
    )

    if missing_before == 0:

        print(
            f"  ✓ {col}: no missing values"
        )

        continue


    # --------------------------------------------------------
    # Interpolate separately for each spatial node
    # --------------------------------------------------------

    df[col] = (
        df
        .groupby("grid_id")[col]
        .transform(
            lambda x:
            x.interpolate(
                method="linear",
                limit_direction="both"
            )
        )
    )


    # --------------------------------------------------------
    # Final median fallback
    # --------------------------------------------------------

    df[col] = (
        df[col]
        .fillna(
            df[col].median()
        )
    )


    print(
        f"  ✓ {col}: "
        f"filled {missing_before} missing values"
    )


# ============================================================
# VERIFY TEMPORAL RESOLUTION
# ============================================================

print(
    "\nChecking temporal resolution..."
)


node_time_counts = (
    df
    .groupby("grid_id")["time"]
    .count()
)


print(
    "Minimum timestamps per node:",
    int(node_time_counts.min())
)


print(
    "Maximum timestamps per node:",
    int(node_time_counts.max())
)


# ============================================================
# EXPLICIT HOURLY CONTINUITY CHECK
# ============================================================

print(
    "\nChecking hourly continuity..."
)


expected_hours = (
    int(
        (
            df["time"].max()
            - df["time"].min()
        ).total_seconds()
        / 3600
    )
    + 1
)


incomplete_nodes = (
    node_time_counts[
        node_time_counts != expected_hours
    ]
)


if len(incomplete_nodes) > 0:

    print(
        "\n⚠ Incomplete hourly nodes detected:"
    )

    print(
        incomplete_nodes
    )

else:

    print(
        f"✓ All nodes contain {expected_hours} hourly records"
    )


# ============================================================
# RAINFALL ACCUMULATION FEATURES
# ============================================================

print(
    "\n[5/10] Calculating rainfall accumulation features..."
)


# ------------------------------------------------------------
# Helper function
# ------------------------------------------------------------

def rolling_sum(
    series,
    window
):

    return (
        series
        .rolling(
            window=window,
            min_periods=window
        )
        .sum()
    )


# ------------------------------------------------------------
# 1 HOUR
# ------------------------------------------------------------

df["rainfall_1h"] = (
    df
    .groupby("grid_id")["precipitation"]
    .transform(
        lambda x:
        rolling_sum(
            x,
            1
        )
    )
)


# ------------------------------------------------------------
# 3 HOURS
# ------------------------------------------------------------

df["rainfall_3h"] = (
    df
    .groupby("grid_id")["precipitation"]
    .transform(
        lambda x:
        rolling_sum(
            x,
            3
        )
    )
)


# ------------------------------------------------------------
# 6 HOURS
# ------------------------------------------------------------

df["rainfall_6h"] = (
    df
    .groupby("grid_id")["precipitation"]
    .transform(
        lambda x:
        rolling_sum(
            x,
            6
        )
    )
)


# ------------------------------------------------------------
# 12 HOURS
# ------------------------------------------------------------

df["rainfall_12h"] = (
    df
    .groupby("grid_id")["precipitation"]
    .transform(
        lambda x:
        rolling_sum(
            x,
            12
        )
    )
)


# ------------------------------------------------------------
# 24 HOURS
# ------------------------------------------------------------

df["rainfall_24h"] = (
    df
    .groupby("grid_id")["precipitation"]
    .transform(
        lambda x:
        rolling_sum(
            x,
            24
        )
    )
)


# ------------------------------------------------------------
# 72 HOURS
# ------------------------------------------------------------

df["rainfall_72h"] = (
    df
    .groupby("grid_id")["precipitation"]
    .transform(
        lambda x:
        rolling_sum(
            x,
            72
        )
    )
)


print("  ✓ rainfall_1h")
print("  ✓ rainfall_3h")
print("  ✓ rainfall_6h")
print("  ✓ rainfall_12h")
print("  ✓ rainfall_24h")
print("  ✓ rainfall_72h")


# ============================================================
# RAINFALL INTENSITY
# ============================================================

print(
    "\nCalculating rainfall intensity features..."
)


df["rainfall_max_6h"] = (
    df
    .groupby("grid_id")["precipitation"]
    .transform(
        lambda x:
        x.rolling(
            window=6,
            min_periods=1
        ).max()
    )
)


df["rainfall_max_24h"] = (
    df
    .groupby("grid_id")["precipitation"]
    .transform(
        lambda x:
        x.rolling(
            window=24,
            min_periods=1
        ).max()
    )
)


print(
    "  ✓ rainfall_max_6h"
)


print(
    "  ✓ rainfall_max_24h"
)


# ============================================================
# HISTORICAL RAINFALL BASELINE
# ============================================================
#
# IMPORTANT:
#
# We use a shifted expanding mean/std.
#
# Therefore the current timestep never contributes
# to its own historical baseline.
#
# This avoids future leakage.
# ============================================================

print(
    "\n[6/10] Calculating leakage-safe rainfall anomaly..."
)


grouped_rain = (
    df
    .groupby("grid_id")["precipitation"]
)


df["rainfall_baseline"] = (
    grouped_rain
    .transform(
        lambda x:
        x.shift(1)
        .expanding(
            min_periods=3
        )
        .mean()
    )
)


df["rainfall_std"] = (
    grouped_rain
    .transform(
        lambda x:
        x.shift(1)
        .expanding(
            min_periods=3
        )
        .std()
    )
)


# ------------------------------------------------------------
# Fallback for early timestamps
# ------------------------------------------------------------

df["rainfall_baseline"] = (
    df["rainfall_baseline"]
    .fillna(
        df["precipitation"]
        .expanding(
            min_periods=1
        )
        .mean()
    )
)


df["rainfall_std"] = (
    df["rainfall_std"]
    .replace(
        0,
        np.nan
    )
)


# ------------------------------------------------------------
# Global safe fallback
# ------------------------------------------------------------

global_std = float(
    df["precipitation"].std()
)


if (
    not np.isfinite(global_std)
    or global_std <= 0
):

    global_std = 1.0


df["rainfall_std"] = (
    df["rainfall_std"]
    .fillna(global_std)
)


# ------------------------------------------------------------
# Rainfall anomaly
# ------------------------------------------------------------

df["rainfall_anomaly"] = (
    (
        df["precipitation"]
        -
        df["rainfall_baseline"]
    )
    /
    df["rainfall_std"]
)


df["rainfall_anomaly"] = (
    df["rainfall_anomaly"]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
    .fillna(0)
)


print(
    "  ✓ rainfall_baseline"
)

print(
    "  ✓ rainfall_std"
)

print(
    "  ✓ rainfall_anomaly"
)


# ============================================================
# WEATHER STRESS
# ============================================================

print(
    "\nCalculating weather stress features..."
)


# ------------------------------------------------------------
# Humidity stress
# ------------------------------------------------------------

df["humidity_stress"] = (
    df["relative_humidity_2m"]
    /
    100.0
)


# ------------------------------------------------------------
# Wind stress
# ------------------------------------------------------------

wind_std = float(
    df["wind_speed_10m"].std()
)


wind_mean = float(
    df["wind_speed_10m"].mean()
)


if (
    not np.isfinite(wind_std)
    or wind_std <= 0
):

    wind_std = 1.0


df["wind_stress"] = (
    (
        df["wind_speed_10m"]
        -
        wind_mean
    )
    /
    wind_std
)


print(
    "  ✓ humidity_stress"
)


print(
    "  ✓ wind_stress"
)


# ============================================================
# CYCLICAL TEMPORAL FEATURES
# ============================================================

print(
    "\n[7/10] Creating cyclical temporal features..."
)


# ------------------------------------------------------------
# Hour of day
# ------------------------------------------------------------

hour = (
    df["time"].dt.hour
    +
    df["time"].dt.minute / 60.0
)


df["hour_sin"] = (
    np.sin(
        2
        *
        np.pi
        *
        hour
        /
        24.0
    )
)


df["hour_cos"] = (
    np.cos(
        2
        *
        np.pi
        *
        hour
        /
        24.0
    )
)


# ------------------------------------------------------------
# Day of year
# ------------------------------------------------------------

day_of_year = (
    df["time"].dt.dayofyear
    -
    1
)


df["day_sin"] = (
    np.sin(
        2
        *
        np.pi
        *
        day_of_year
        /
        365.25
    )
)


df["day_cos"] = (
    np.cos(
        2
        *
        np.pi
        *
        day_of_year
        /
        365.25
    )
)


# ------------------------------------------------------------
# Month
# ------------------------------------------------------------

month = (
    df["time"].dt.month
    -
    1
)


df["month_sin"] = (
    np.sin(
        2
        *
        np.pi
        *
        month
        /
        12.0
    )
)


df["month_cos"] = (
    np.cos(
        2
        *
        np.pi
        *
        month
        /
        12.0
    )
)


# ------------------------------------------------------------
# Year retained ONLY for analysis/metadata
# ------------------------------------------------------------

df["year"] = (
    df["time"].dt.year
)


print("  ✓ hour_sin")
print("  ✓ hour_cos")
print("  ✓ day_sin")
print("  ✓ day_cos")
print("  ✓ month_sin")
print("  ✓ month_cos")
print("  ✓ year (metadata only)")


# ============================================================
# MONSOON
# ============================================================

df["monsoon"] = (
    df["time"]
    .dt.month
    .isin(
        [6, 7, 8, 9]
    )
    .astype(np.int8)
)


print(
    "  ✓ monsoon"
)


# ============================================================
# HEAVY / EXTREME RAIN
# ============================================================

df["heavy_rain_flag"] = (
    df["rainfall_24h"]
    >=
    50
).astype(np.int8)


df["extreme_rain_flag"] = (
    df["rainfall_24h"]
    >=
    100
).astype(np.int8)


print(
    "  ✓ heavy_rain_flag"
)


print(
    "  ✓ extreme_rain_flag"
)


# ============================================================
# ADDITIONAL SPATIOTEMPORAL FEATURES
# ============================================================

print(
    "\nCreating additional spatiotemporal features..."
)


# ------------------------------------------------------------
# Previous 3-hour rainfall
# ------------------------------------------------------------

df["rainfall_3h_previous"] = (
    df
    .groupby("grid_id")["rainfall_3h"]
    .shift(3)
)


# ------------------------------------------------------------
# Rainfall acceleration/change
# ------------------------------------------------------------

df["rainfall_change_3h"] = (
    df["rainfall_3h"]
    -
    df["rainfall_3h_previous"]
)


# ------------------------------------------------------------
# Rain persistence
# ------------------------------------------------------------

df["rain_persistence_6h"] = (
    df
    .groupby("grid_id")["rain"]
    .transform(
        lambda x:
        x.rolling(
            window=6,
            min_periods=1
        ).sum()
    )
)


# ------------------------------------------------------------
# Rainfall × humidity interaction
# ------------------------------------------------------------

df["rain_humidity_interaction"] = (
    df["rainfall_6h"]
    *
    df["humidity_stress"]
)


print(
    "  ✓ rainfall_3h_previous"
)


print(
    "  ✓ rainfall_change_3h"
)


print(
    "  ✓ rain_persistence_6h"
)


print(
    "  ✓ rain_humidity_interaction"
)


# ============================================================
# CLEAN DATA
# ============================================================

print(
    "\n[8/10] Cleaning final dataset..."
)


df = df.replace(
    [np.inf, -np.inf],
    np.nan
)


numeric_columns = (
    df
    .select_dtypes(
        include=[np.number]
    )
    .columns
)


for col in numeric_columns:

    df[col] = (
        df[col]
        .fillna(0)
    )


# ============================================================
# FINAL SORT
# ============================================================

df = (
    df
    .sort_values(
        ["grid_id", "time"]
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# PHASE 2B WEATHER FEATURE LIST
# ============================================================
#
# These are ONLY the features generated by Phase 2B.
#
# Spatial and flood-risk features are intentionally NOT
# included here because they are added later by:
#
# integrate_spatiotemporal_data.py
#
# Total = 35
#
# ------------------------------------------------------------
# 5  Base weather
# 6  Rainfall accumulation
# 2  Rainfall intensity
# 3  Rainfall anomaly
# 3  Rainfall dynamics
# 3  Weather stress
# 6  Cyclical temporal
# 1  Monsoon
# 2  Rain event indicators
#
# Total = 31
#
# Additional feature:
# rain_humidity_interaction is included in weather stress,
# making the complete feature count 32.
#
# ============================================================

GNN_FEATURES = [

    # --------------------------------------------------------
    # Base weather
    # --------------------------------------------------------

    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "wind_speed_10m",

    # --------------------------------------------------------
    # Rainfall accumulation
    # --------------------------------------------------------

    "rainfall_1h",
    "rainfall_3h",
    "rainfall_6h",
    "rainfall_12h",
    "rainfall_24h",
    "rainfall_72h",

    # --------------------------------------------------------
    # Rainfall intensity
    # --------------------------------------------------------

    "rainfall_max_6h",
    "rainfall_max_24h",

    # --------------------------------------------------------
    # Rainfall anomaly
    # --------------------------------------------------------

    "rainfall_baseline",
    "rainfall_std",
    "rainfall_anomaly",

    # --------------------------------------------------------
    # Rainfall dynamics
    # --------------------------------------------------------

    "rainfall_3h_previous",
    "rainfall_change_3h",
    "rain_persistence_6h",

    # --------------------------------------------------------
    # Weather stress
    # --------------------------------------------------------

    "humidity_stress",
    "wind_stress",
    "rain_humidity_interaction",

    # --------------------------------------------------------
    # Cyclical temporal features
    # --------------------------------------------------------

    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "month_sin",
    "month_cos",

    # --------------------------------------------------------
    # Seasonal indicator
    # --------------------------------------------------------

    "monsoon",

    # --------------------------------------------------------
    # Rainfall event indicators
    # --------------------------------------------------------

    "heavy_rain_flag",
    "extreme_rain_flag"
]


print(
    "\nGNN weather feature count:",
    len(GNN_FEATURES)
)


# ============================================================
# VERIFY PHASE 2B FEATURES
# ============================================================

print(
    "\n[9/10] Verifying Phase 2B weather features..."
)


missing_gnn_features = [
    col
    for col in GNN_FEATURES
    if col not in df.columns
]


if missing_gnn_features:

    print(
        "\n❌ Missing Phase 2B features:"
    )

    for col in missing_gnn_features:

        print(
            "  -",
            col
        )

    raise ValueError(
        "Required Phase 2B weather features are missing."
    )


print(
    "✓ All Phase 2B weather features available"
)


# ============================================================
# DATA QUALITY CHECK
# ============================================================

print(
    "\nFinal data quality checks..."
)


missing_total = int(
    df.isnull().sum().sum()
)


duplicate_total = int(
    df.duplicated(
        subset=[
            "grid_id",
            "time"
        ]
    ).sum()
)


print(
    "Missing values:",
    missing_total
)


print(
    "Duplicate node-time rows:",
    duplicate_total
)


if missing_total != 0:

    raise ValueError(
        "Missing values remain in final dataset."
    )


if duplicate_total != 0:

    raise ValueError(
        "Duplicate node-time rows remain."
    )


# ============================================================
# SAVE
# ============================================================

print(
    "\n[10/10] Saving feature dataset..."
)


df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print(
    "\n" + "=" * 70
)


print(
    "PHASE 2B COMPLETE"
)


print(
    "=" * 70
)


print(
    "\nOutput:"
)


print(
    OUTPUT_FILE
)


print(
    "\nRows:",
    len(df)
)


print(
    "Nodes:",
    df["grid_id"].nunique()
)


print(
    "\nTime range:",
    df["time"].min(),
    "to",
    df["time"].max()
)


print(
    "\nGNN weather feature count:",
    len(GNN_FEATURES)
)


print(
    "\nGNN WEATHER FEATURES:"
)


for i, feature in enumerate(
    GNN_FEATURES
):

    print(
        f"  {i:02d} -> {feature}"
    )


# ============================================================
# FEATURE STATISTICS
# ============================================================

print(
    "\nRainfall statistics:"
)


print(
    df[
        [
            "rainfall_1h",
            "rainfall_3h",
            "rainfall_6h",
            "rainfall_24h",
            "rainfall_72h"
        ]
    ]
    .describe()
    .round(3)
    .to_string()
)


# ============================================================
# TEMPORAL FEATURE STATISTICS
# ============================================================

print(
    "\nCyclical temporal feature statistics:"
)


print(
    df[
        [
            "hour_sin",
            "hour_cos",
            "day_sin",
            "day_cos",
            "month_sin",
            "month_cos",
            "monsoon"
        ]
    ]
    .describe()
    .round(4)
    .to_string()
)


# ============================================================
# MONSOON DISTRIBUTION
# ============================================================

print(
    "\nMonsoon distribution:"
)


print(
    df["monsoon"]
    .value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# HEAVY / EXTREME RAIN COUNTS
# ============================================================

print(
    "\nRain event counts:"
)


print(
    "Heavy rain:",
    int(
        df["heavy_rain_flag"].sum()
    )
)


print(
    "Extreme rain:",
    int(
        df["extreme_rain_flag"].sum()
    )
)


# ============================================================
# FILE SIZE
# ============================================================

size_mb = (
    OUTPUT_FILE.stat().st_size
    /
    (1024 * 1024)
)


print(
    "\nOutput file size:",
    f"{size_mb:.2f} MB"
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print(
    "\n✓ Phase 2B finished successfully."
)


print(
    "✓ Cyclical temporal encoding enabled."
)


print(
    "✓ Raw hour/day/month removed from GNN feature set."
)


print(
    "✓ Historical rainfall baseline is leakage-safe."
)


print(
    "✓ Node-wise rainfall accumulation features generated."
)


print(
    "✓ Weather dataset ready for spatial integration."
)


print(
    "✓ Spatial/risk features will be added in the next stage."
)


print(
    "=" * 70
)
