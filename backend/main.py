from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

import pandas as pd
import numpy as np
import json


# ============================================================
# DISASTERFLOW AI
# EMERGENCY DECISION SUPPORT API
# ============================================================

app = FastAPI(
    title="DisasterFlow AI",
    description=(
        "AI-powered spatiotemporal flood prediction, "
        "emergency prioritization and resource allocation system"
    ),
    version="2.2.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "processed"

MAIN_DATA_FILE = (
    DATA_DIR /
    "disasterflow_final_node_priorities.csv"
)

SPATIAL_NODES_FILE = (
    DATA_DIR /
    "spatial_graph_nodes.csv"
)

PREDICTION_FILE = (
    DATA_DIR /
    "predictions" /
    "stgnn_node_predictions.csv"
)


# ============================================================
# FILE VALIDATION
# ============================================================

if not MAIN_DATA_FILE.exists():

    raise FileNotFoundError(
        f"\nMain DisasterFlow dataset not found:\n"
        f"{MAIN_DATA_FILE}\n"
    )


if not SPATIAL_NODES_FILE.exists():

    raise FileNotFoundError(
        f"\nSpatial graph node file not found:\n"
        f"{SPATIAL_NODES_FILE}\n"
    )


# ============================================================
# LOAD MAIN DATASET
# ============================================================

print()
print("=" * 70)
print("DISASTERFLOW AI - LOADING DATA")
print("=" * 70)

df = pd.read_csv(
    MAIN_DATA_FILE
)

print(
    f"Main dataset loaded : {len(df):,} rows"
)

print(
    f"Main dataset columns: {len(df.columns)}"
)


# ============================================================
# LOAD SPATIAL NODE MAPPING
# ============================================================

spatial_nodes = pd.read_csv(
    SPATIAL_NODES_FILE
)

print(
    f"Spatial graph nodes : {len(spatial_nodes):,}"
)

print(
    f"Spatial node columns: {len(spatial_nodes.columns)}"
)


# ============================================================
# VALIDATE SPATIAL NODE FILE
# ============================================================

required_spatial_columns = [
    "node_index",
    "grid_id"
]

missing_spatial_columns = [
    column
    for column in required_spatial_columns
    if column not in spatial_nodes.columns
]

if missing_spatial_columns:

    raise ValueError(
        "Spatial node file is missing required columns: "
        + ", ".join(missing_spatial_columns)
    )


# ============================================================
# RECOVER NODE INDEX
#
# disasterflow_final_node_priorities.csv
# does not contain node_index.
#
# spatial_graph_nodes.csv provides:
#
# grid_id <-> node_index
# ============================================================

if "node_index" not in df.columns:

    node_mapping = (
        spatial_nodes[
            [
                "grid_id",
                "node_index"
            ]
        ]
        .drop_duplicates(
            subset=["grid_id"]
        )
    )

    df = df.merge(
        node_mapping,
        on="grid_id",
        how="left"
    )

    print(
        f"Node indices recovered: "
        f"{df['node_index'].notna().sum():,}/"
        f"{len(df):,}"
    )

else:

    print(
        "Main dataset already contains node_index."
    )


# ============================================================
# VALIDATE NODE MAPPING
# ============================================================

missing_nodes = int(
    df["node_index"].isna().sum()
)

unique_nodes = int(
    df["node_index"].nunique()
)

print(
    f"Unique mapped nodes   : {unique_nodes:,}"
)

print(
    f"Missing node indices  : {missing_nodes:,}"
)


if missing_nodes > 0:

    print(
        "WARNING: Some grid IDs could not be mapped "
        "to node indices."
    )


# ============================================================
# LOAD ST-GNN PREDICTIONS
# ============================================================

if PREDICTION_FILE.exists():

    print()
    print(
        "Loading ST-GNN predictions..."
    )

    # Only load the columns needed by the API.
    #
    # This avoids loading unnecessary data from the
    # 167 MB prediction CSV.

    prediction_columns = [
        "sequence_index",
        "forecast_time",
        "split",
        "node_index",
        "forecast_flood_probability"
    ]

    stgnn_predictions = pd.read_csv(
        PREDICTION_FILE,
        usecols=prediction_columns
    )

    print(
        f"ST-GNN prediction records: "
        f"{len(stgnn_predictions):,}"
    )

else:

    stgnn_predictions = pd.DataFrame()

    print(
        "WARNING: ST-GNN prediction file not found."
    )


# ============================================================
# NUMERIC CLEANING
# ============================================================

def numeric_clean(dataframe):

    result = dataframe.copy()

    numeric_columns = [

        "node_index",

        "latitude",
        "longitude",

        "elevation_m",
        "slope_degrees",
        "distance_to_water_m",

        "flood_probability",
        "elevation_risk",
        "water_risk",

        "resource_demand",
        "resources_required",
        "allocated_resources",

        "operational_priority",
        "priority_rank",

        "region_id",

        "dynamic_rainfall_risk",
        "dynamic_flood_signal",
        "environmental_stress",
        "dynamic_resource_demand",

        "resource_gap",
        "resource_shortage_flag"

    ]

    for column in numeric_columns:

        if column in result.columns:

            result[column] = pd.to_numeric(
                result[column],
                errors="coerce"
            )

    return result


df = numeric_clean(df)


# ============================================================
# RESOURCE GAP
# ============================================================

def calculate_resource_gap(dataframe):

    result = dataframe.copy()

    if "resources_required" in result.columns:

        required = pd.to_numeric(
            result["resources_required"],
            errors="coerce"
        ).fillna(0)

    else:

        required = pd.Series(
            np.zeros(len(result)),
            index=result.index
        )

    if "allocated_resources" in result.columns:

        allocated = pd.to_numeric(
            result["allocated_resources"],
            errors="coerce"
        ).fillna(0)

    else:

        allocated = pd.Series(
            np.zeros(len(result)),
            index=result.index
        )

    result["resource_gap"] = (
        required - allocated
    )

    result["resource_shortage_flag"] = (
        result["resource_gap"] > 0
    ).astype(int)

    return result


df = calculate_resource_gap(df)


# ============================================================
# RISK CLASSIFICATION
# ============================================================

def calculate_risk_levels(dataframe):

    result = dataframe.copy()

    if "flood_probability" not in result.columns:

        result["risk_level"] = "UNKNOWN"

        return result

    probabilities = pd.to_numeric(
        result["flood_probability"],
        errors="coerce"
    )

    valid = probabilities.dropna()

    if len(valid) == 0:

        result["risk_level"] = "UNKNOWN"

        return result

    q50 = valid.quantile(0.50)
    q75 = valid.quantile(0.75)
    q90 = valid.quantile(0.90)

    def classify(value):

        if pd.isna(value):

            return "UNKNOWN"

        if value >= q90:

            return "CRITICAL"

        if value >= q75:

            return "HIGH"

        if value >= q50:

            return "MEDIUM"

        return "LOW"

    result["risk_level"] = (
        probabilities.apply(classify)
    )

    return result


df = calculate_risk_levels(df)


# ============================================================
# PLACE NAME DETECTION
# ============================================================

def detect_place_column(dataframe):

    possible_columns = [

        "place_name",
        "location_name",
        "locality",
        "area",
        "area_name",
        "place",
        "location",
        "neighborhood",
        "neighbourhood",
        "ward_name",
        "region_name"

    ]

    for column in possible_columns:

        if column in dataframe.columns:

            return column

    return None


PLACE_COLUMN = detect_place_column(
    df
)


# ============================================================
# PLACE NAME GENERATOR
# ============================================================

def get_place_name(row):

    if PLACE_COLUMN is not None:

        value = row.get(
            PLACE_COLUMN
        )

        if pd.notna(value):

            value = str(value).strip()

            if value:

                return value

    grid = row.get(
        "grid_id",
        None
    )

    if pd.notna(grid):

        grid = str(grid).strip()

        if grid:

            region = row.get(
                "region_id",
                None
            )

            if pd.notna(region):

                try:

                    return (
                        f"Mumbai Grid {grid} "
                        f"- Region {int(region)}"
                    )

                except Exception:

                    return (
                        f"Mumbai Grid {grid} "
                        f"- Region {region}"
                    )

            return (
                f"Mumbai Grid {grid}"
            )

    node = row.get(
        "node_index",
        None
    )

    if pd.notna(node):

        try:

            node = int(node)

        except Exception:

            pass

        region = row.get(
            "region_id",
            None
        )

        if pd.notna(region):

            try:

                return (
                    f"Mumbai Grid NODE_{node} "
                    f"- Region {int(region)}"
                )

            except Exception:

                return (
                    f"Mumbai Grid NODE_{node} "
                    f"- Region {region}"
                )

        return (
            f"Mumbai Grid NODE_{node}"
        )

    return "Unknown Mumbai Location"


# ============================================================
# CREATE PLACE NAME
# ============================================================

df["place_name"] = df.apply(
    get_place_name,
    axis=1
)


# ============================================================
# ST-GNN FORECAST PREPARATION
# ============================================================

def build_latest_forecasts():

    if stgnn_predictions.empty:

        print(
            "WARNING: No ST-GNN predictions available."
        )

        return pd.DataFrame()

    predictions = (
        stgnn_predictions.copy()
    )

    print()
    print("=" * 70)
    print("ST-GNN DATA INSPECTION")
    print("=" * 70)

    print(
        "Prediction records:",
        f"{len(predictions):,}"
    )

    print(
        "Prediction columns:",
        list(predictions.columns)
    )

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = [
        "sequence_index",
        "node_index",
        "forecast_flood_probability"
    ]

    missing = [
        column
        for column in required_columns
        if column not in predictions.columns
    ]

    if missing:

        print(
            "WARNING: Missing ST-GNN columns:",
            missing
        )

        return pd.DataFrame()

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    predictions["sequence_index"] = pd.to_numeric(
        predictions["sequence_index"],
        errors="coerce"
    )

    predictions["node_index"] = pd.to_numeric(
        predictions["node_index"],
        errors="coerce"
    )

    predictions["forecast_flood_probability"] = pd.to_numeric(
        predictions["forecast_flood_probability"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Remove invalid records
    # --------------------------------------------------------

    predictions = predictions.dropna(
        subset=[
            "sequence_index",
            "node_index",
            "forecast_flood_probability"
        ]
    )

    if predictions.empty:

        print(
            "WARNING: No valid ST-GNN prediction records."
        )

        return pd.DataFrame()

    # --------------------------------------------------------
    # Latest sequence
    # --------------------------------------------------------

    latest_sequence = (
        predictions["sequence_index"].max()
    )

    latest = predictions[
        predictions["sequence_index"]
        == latest_sequence
    ].copy()

    print(
        f"Latest ST-GNN sequence: "
        f"{int(latest_sequence)}"
    )

    print(
        f"Latest sequence records: "
        f"{len(latest):,}"
    )

    print(
        f"Latest unique nodes: "
        f"{latest['node_index'].nunique():,}"
    )

    # --------------------------------------------------------
    # Forecast time
    # --------------------------------------------------------

    if "forecast_time" not in latest.columns:

        latest["forecast_time"] = None

    # --------------------------------------------------------
    # Keep required columns
    # --------------------------------------------------------

    latest = latest[
        [
            "node_index",
            "forecast_time",
            "forecast_flood_probability"
        ]
    ].copy()

    # --------------------------------------------------------
    # One record per node
    # --------------------------------------------------------

    latest = (
        latest
        .dropna(
            subset=["node_index"]
        )
        .drop_duplicates(
            subset=["node_index"],
            keep="last"
        )
    )

    # --------------------------------------------------------
    # Forecast range
    # --------------------------------------------------------

    if len(latest) > 0:

        print(
            "Forecast probability range: "
            f"{latest['forecast_flood_probability'].min():.8f} "
            f"to "
            f"{latest['forecast_flood_probability'].max():.8f}"
        )

        print(
            "Forecast probability mean: "
            f"{latest['forecast_flood_probability'].mean():.8f}"
        )

        print(
            "Forecast time:",
            latest["forecast_time"].iloc[0]
        )

    print("=" * 70)

    return latest


latest_forecasts = (
    build_latest_forecasts()
)


# ============================================================
# MERGE ST-GNN FORECASTS
# ============================================================

def merge_forecasts(
    main_dataframe,
    forecasts
):

    result = main_dataframe.copy()

    # --------------------------------------------------------
    # No forecasts
    # --------------------------------------------------------

    if forecasts.empty:

        result[
            "forecast_flood_probability"
        ] = np.nan

        result[
            "forecast_time"
        ] = None

        return result

    # --------------------------------------------------------
    # Node index is now guaranteed by mapping above
    # --------------------------------------------------------

    if "node_index" not in result.columns:

        result[
            "forecast_flood_probability"
        ] = np.nan

        result[
            "forecast_time"
        ] = None

        print(
            "WARNING: node_index unavailable. "
            "ST-GNN merge skipped."
        )

        return result

    # --------------------------------------------------------
    # Prepare merge keys
    # --------------------------------------------------------

    result[
        "_merge_node_index"
    ] = pd.to_numeric(
        result["node_index"],
        errors="coerce"
    )

    forecasts_copy = forecasts.copy()

    forecasts_copy[
        "_merge_node_index"
    ] = pd.to_numeric(
        forecasts_copy["node_index"],
        errors="coerce"
    )

    forecasts_copy = forecasts_copy[
        [
            "_merge_node_index",
            "forecast_time",
            "forecast_flood_probability"
        ]
    ]

    # --------------------------------------------------------
    # Ensure unique forecast node
    # --------------------------------------------------------

    forecasts_copy = (
        forecasts_copy
        .drop_duplicates(
            subset=["_merge_node_index"],
            keep="last"
        )
    )

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    result = result.merge(
        forecasts_copy,
        on="_merge_node_index",
        how="left"
    )

    result = result.drop(
        columns=[
            "_merge_node_index"
        ],
        errors="ignore"
    )

    # --------------------------------------------------------
    # Match statistics
    # --------------------------------------------------------

    matched = int(
        result[
            "forecast_flood_probability"
        ].notna().sum()
    )

    total = len(result)

    print()
    print("=" * 70)
    print("ST-GNN FORECAST MERGE")
    print("=" * 70)

    print(
        f"Main nodes        : {total:,}"
    )

    print(
        f"Matched forecasts : {matched:,}"
    )

    print(
        f"Unmatched         : {total - matched:,}"
    )

    if total > 0:

        print(
            f"Coverage          : "
            f"{matched / total * 100:.2f}%"
        )

    print("=" * 70)

    return result


df = merge_forecasts(
    df,
    latest_forecasts
)


# ============================================================
# SAFE NORMALIZATION
# ============================================================

def normalize_series(series):

    series = pd.to_numeric(
        series,
        errors="coerce"
    ).fillna(0)

    if len(series) == 0:

        return series

    minimum = series.min()
    maximum = series.max()

    if (
        pd.isna(minimum)
        or pd.isna(maximum)
        or maximum <= minimum
    ):

        return pd.Series(
            np.zeros(len(series)),
            index=series.index
        )

    return (
        (series - minimum)
        /
        (maximum - minimum)
    )


# ============================================================
# ENSURE REQUIRED COLUMNS
# ============================================================

required_columns_defaults = {

    "flood_probability": 0,
    "water_risk": 0,

    "dynamic_rainfall_risk": 0,
    "dynamic_flood_signal": 0,

    "environmental_stress": 0,

    "resource_demand": 0,
    "dynamic_resource_demand": 0,

    "resources_required": 0,
    "allocated_resources": 0,

    "resource_gap": 0,
    "resource_shortage_flag": 0

}


for column, default in (
    required_columns_defaults.items()
):

    if column not in df.columns:

        df[column] = default


# ============================================================
# EMERGENCY PRIORITY ENGINE
# ============================================================

def build_emergency_engine(dataframe):
    result = dataframe.copy()

    # ========================================================
    # 1. RAW ST-GNN FORECAST
    # ========================================================
    forecast_raw = pd.to_numeric(
        result.get(
            'forecast_flood_probability',
            pd.Series(0.0, index=result.index)
        ),
        errors='coerce'
    ).fillna(0.0).clip(0, 1)

    result['forecast_flood_probability'] = forecast_raw

    # ========================================================
    # 2. ST-GNN RELATIVE RISK
    #
    # The trained model currently produces a narrow probability
    # range (~0.00022). Directly multiplying this by 45% makes
    # the signal effectively zero.
    #
    # Therefore we convert the forecast into a relative risk
    # signal across the 140 spatial nodes.
    # ========================================================
    forecast_risk = normalize_series(forecast_raw)

    # If forecasts are almost identical, use a neutral signal
    # instead of falsely declaring all nodes zero-risk.
    if forecast_raw.max() - forecast_raw.min() < 1e-8:
        forecast_risk = pd.Series(
            0.5,
            index=result.index
        )

    result['stgnn_forecast_risk'] = forecast_risk

    # ========================================================
    # 3. CURRENT FLOOD SUSCEPTIBILITY
    # ========================================================
    flood = normalize_series(
        pd.to_numeric(
            result.get(
                'flood_probability',
                pd.Series(0.0, index=result.index)
            ),
            errors='coerce'
        ).fillna(0.0)
    )

    result['current_flood_risk'] = flood

    # ========================================================
    # 4. RAINFALL SIGNAL
    # ========================================================
    rainfall_raw = pd.to_numeric(
        result.get(
            'dynamic_rainfall_risk',
            pd.Series(0.0, index=result.index)
        ),
        errors='coerce'
    ).fillna(0.0)

    rainfall = normalize_series(rainfall_raw)

    result['rainfall_risk_signal'] = rainfall

    # ========================================================
    # 5. WATER EXPOSURE
    # ========================================================
    water = normalize_series(
        pd.to_numeric(
            result.get(
                'water_risk',
                pd.Series(0.0, index=result.index)
            ),
            errors='coerce'
        ).fillna(0.0)
    )

    result['water_exposure_signal'] = water

    # ========================================================
    # 6. ENVIRONMENTAL STRESS
    # ========================================================
    environmental = normalize_series(
        pd.to_numeric(
            result.get(
                'environmental_stress',
                pd.Series(0.0, index=result.index)
            ),
            errors='coerce'
        ).fillna(0.0)
    )

    result['environmental_risk_signal'] = environmental

    # ========================================================
    # 7. RESOURCE SHORTAGE
    #
    # resource_gap is expected to be:
    # required - allocated
    #
    # Positive = shortage
    # Zero/negative = sufficient
    # ========================================================
    gap = pd.to_numeric(
        result.get(
            'resource_gap',
            pd.Series(0.0, index=result.index)
        ),
        errors='coerce'
    ).fillna(0.0)

    shortage = normalize_series(
        gap.clip(lower=0)
    )

    result['resource_shortage_signal'] = shortage

    # ========================================================
    # 8. RESOURCE DEMAND
    # ========================================================
    demand = normalize_series(
        pd.to_numeric(
            result.get(
                'resource_demand',
                pd.Series(0.0, index=result.index)
            ),
            errors='coerce'
        ).fillna(0.0)
    )

    result['resource_demand_signal'] = demand

    # ========================================================
    # 9. COMBINED EMERGENCY SCORE
    #
    # ST-GNN forecast          30%
    # Current flood risk      25%
    # Water exposure          15%
    # Rainfall                10%
    # Environmental stress     5%
    # Resource demand          5%
    # Resource shortage       10%
    #
    # Total                   100%
    # ========================================================
    score = (
        0.30 * forecast_risk
        + 0.25 * flood
        + 0.15 * water
        + 0.10 * rainfall
        + 0.05 * environmental
        + 0.05 * demand
        + 0.10 * shortage
    )

    # ========================================================
    # 10. 0-100 EMERGENCY SCORE
    # ========================================================
    result['emergency_priority_score'] = (
        score.clip(0, 1) * 100
    )

    # ========================================================
    # 11. PRIORITY CLASSIFICATION
    #
    # Use score + current flood risk + forecast risk.
    # This prevents contradictory CRITICAL/LOW situations.
    # ========================================================
    def classify_priority(row):
        score_value = float(
            pd.to_numeric(
                row.get('emergency_priority_score', 0),
                errors='coerce'
            ) or 0
        )

        flood_value = float(
            pd.to_numeric(
                row.get('current_flood_risk', 0),
                errors='coerce'
            ) or 0
        )

        forecast_value = float(
            pd.to_numeric(
                row.get('stgnn_forecast_risk', 0),
                errors='coerce'
            ) or 0
        )

        # Strong combined signals
        if (
            score_value >= 70
            or (
                score_value >= 55
                and flood_value >= 0.75
                and forecast_value >= 0.75
            )
        ):
            return 'CRITICAL'

        if (
            score_value >= 50
            or (
                score_value >= 40
                and (
                    flood_value >= 0.75
                    or forecast_value >= 0.75
                )
            )
        ):
            return 'HIGH'

        if (
            score_value >= 30
            or flood_value >= 0.50
            or forecast_value >= 0.70
        ):
            return 'MEDIUM'

        return 'LOW'

    result['emergency_priority'] = result.apply(
        classify_priority,
        axis=1
    )

    # ========================================================
    # 12. FORECAST PERCENTAGE
    # ========================================================
    result['forecast_probability_percent'] = (
        forecast_raw * 100
    )

    # Relative ST-GNN risk as percentage
    result['stgnn_relative_risk_percent'] = (
        forecast_risk * 100
    )

    # ========================================================
    # 13. DYNAMIC FLOOD SIGNAL
    #
    # This represents how strongly the forecast agrees with
    # the existing flood susceptibility.
    # ========================================================
    result['dynamic_flood_signal'] = (
        0.60 * forecast_risk
        + 0.40 * flood
    ).clip(0, 1)

    # ========================================================
    # 14. URGENCY REASON
    # ========================================================
    reasons = []

    for _, row in result.iterrows():

        row_reasons = []

        score_value = float(
            pd.to_numeric(
                row.get('emergency_priority_score', 0),
                errors='coerce'
            ) or 0
        )

        forecast_value = float(
            pd.to_numeric(
                row.get('stgnn_forecast_risk', 0),
                errors='coerce'
            ) or 0
        )

        flood_value = float(
            pd.to_numeric(
                row.get('current_flood_risk', 0),
                errors='coerce'
            ) or 0
        )

        water_value = float(
            pd.to_numeric(
                row.get('water_exposure_signal', 0),
                errors='coerce'
            ) or 0
        )

        rainfall_value = float(
            pd.to_numeric(
                row.get('rainfall_risk_signal', 0),
                errors='coerce'
            ) or 0
        )

        shortage_value = float(
            pd.to_numeric(
                row.get('resource_shortage_signal', 0),
                errors='coerce'
            ) or 0
        )

        if forecast_value >= 0.75:
            row_reasons.append(
                'ST-GNN predicts elevated spatial flood risk'
            )
        elif forecast_value >= 0.50:
            row_reasons.append(
                'ST-GNN detects increased relative flood risk'
            )

        if flood_value >= 0.75:
            row_reasons.append(
                'high current flood susceptibility'
            )

        if water_value >= 0.75:
            row_reasons.append(
                'high proximity-to-water exposure'
            )

        if rainfall_value >= 0.75:
            row_reasons.append(
                'strong rainfall risk signal'
            )

        if shortage_value >= 0.50:
            row_reasons.append(
                'emergency resource shortage'
            )

        if score_value >= 70:
            row_reasons.append(
                'multiple high-severity emergency signals'
            )

        if not row_reasons:
            if score_value >= 30:
                row_reasons.append(
                    'moderate combined environmental risk'
                )
            else:
                row_reasons.append(
                    'low current emergency signal'
                )

        reasons.append(
            ' + '.join(row_reasons[:3])
        )

    result['urgency_reason'] = reasons

    # ========================================================
    # 15. RECOMMENDED ACTION
    # ========================================================
    actions = []

    for _, row in result.iterrows():

        score_value = float(
            pd.to_numeric(
                row.get('emergency_priority_score', 0),
                errors='coerce'
            ) or 0
        )

        gap_value = float(
            pd.to_numeric(
                row.get('resource_gap', 0),
                errors='coerce'
            ) or 0
        )

        priority_value = str(
            row.get(
                'emergency_priority',
                'LOW'
            )
        )

        if priority_value == 'CRITICAL':
            action = (
                'IMMEDIATE RESPONSE - '
                'activate emergency operations, '
                'prioritize evacuation readiness, '
                'and deploy resources'
            )

        elif priority_value == 'HIGH':
            action = (
                'HIGH ALERT - '
                'pre-position emergency resources '
                'and increase continuous monitoring'
            )

        elif priority_value == 'MEDIUM':
            action = (
                'ENHANCED MONITORING - '
                'prepare nearby resources and '
                'increase surveillance'
            )

        else:
            action = (
                'ROUTINE MONITORING - '
                'maintain disaster readiness'
            )

        if gap_value > 0:
            action += (
                f' - Resource gap: {gap_value:.0f} units'
            )

        actions.append(action)

    result['recommended_action'] = actions

    # ========================================================
    # 16. FINAL EMERGENCY RANKING
    # ========================================================
    result = result.sort_values(
        by=[
            'emergency_priority_score',
            'stgnn_forecast_risk',
            'flood_probability'
        ],
        ascending=False
    ).reset_index(drop=True)

    result['emergency_rank'] = np.arange(
        1,
        len(result) + 1
    )

    return result



df = build_emergency_engine(
    df
)


# ============================================================
# JSON SERIALIZATION
# ============================================================

def dataframe_to_records(dataframe):

    result = dataframe.copy()

    result = result.replace(
        [np.inf, -np.inf],
        np.nan
    )

    result = result.where(
        pd.notnull(result),
        None
    )

    return json.loads(
        result.to_json(
            orient="records"
        )
    )


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health_check():

    forecast_matches = 0

    if (
        "forecast_flood_probability"
        in df.columns
    ):

        forecast_matches = int(
            df[
                "forecast_flood_probability"
            ].notna().sum()
        )

    return {

        "status": "healthy",

        "service": "DisasterFlow AI",

        "version": "2.2.0",

        "nodes": int(
            len(df)
        ),

        "stgnn_predictions": (
            not stgnn_predictions.empty
        ),

        "stgnn_forecast_nodes_matched":
            forecast_matches,

        "stgnn_forecast_coverage_percent":
            float(
                forecast_matches
                /
                max(len(df), 1)
                *
                100
            ),

        "data_source":
            MAIN_DATA_FILE.name,

        "spatial_node_source":
            SPATIAL_NODES_FILE.name,

        "prediction_source":
            PREDICTION_FILE.name

    }


# ============================================================
# ALL RISK NODES
# ============================================================

@app.get("/risk")
def get_risk():

    return dataframe_to_records(
        df
    )


# ============================================================
# EMERGENCY PRIORITIES
# ============================================================

@app.get("/priorities")
def get_priorities():

    columns = [

        "emergency_rank",

        "grid_id",

        "node_index",

        "place_name",

        "latitude",
        "longitude",

        "emergency_priority",

        "emergency_priority_score",

        "forecast_flood_probability",

        "forecast_probability_percent",

        "forecast_time",

        "flood_probability",

        "dynamic_rainfall_risk",

        "dynamic_flood_signal",

        "environmental_stress",

        "water_risk",

        "resource_demand",

        "dynamic_resource_demand",

        "resources_required",

        "allocated_resources",

        "resource_gap",

        "resource_shortage_flag",

        "risk_level",

        "urgency_reason",

        "recommended_action"

    ]

    available = [
        column
        for column in columns
        if column in df.columns
    ]

    result = df[
        available
    ].copy()

    return dataframe_to_records(
        result
    )


# ============================================================
# TOP 10 EMERGENCY LOCATIONS
# ============================================================

@app.get("/emergency-top10")
def get_top_emergencies():

    result = df.head(
        10
    ).copy()

    return dataframe_to_records(
        result
    )


# ============================================================
# RESOURCE ALLOCATION
# ============================================================

@app.get("/resources")
def get_resources():

    columns = [

        "emergency_rank",

        "grid_id",

        "node_index",

        "place_name",

        "latitude",
        "longitude",

        "emergency_priority",

        "emergency_priority_score",

        "resource_demand",

        "dynamic_resource_demand",

        "resources_required",

        "allocated_resources",

        "resource_gap",

        "resource_shortage_flag",

        "risk_level"

    ]

    available = [
        column
        for column in columns
        if column in df.columns
    ]

    result = df[
        available
    ].copy()

    return dataframe_to_records(
        result
    )


# ============================================================
# PREDICTION SUMMARY
# ============================================================

@app.get("/prediction")
def get_prediction_summary():

    # --------------------------------------------------------
    # Current flood probability
    # --------------------------------------------------------

    flood_probability = pd.to_numeric(
        df.get(
            "flood_probability",
            pd.Series(
                np.zeros(len(df))
            )
        ),
        errors="coerce"
    ).fillna(0)

    # --------------------------------------------------------
    # ST-GNN forecast
    # --------------------------------------------------------

    forecast = pd.to_numeric(
        df.get(
            "forecast_flood_probability",
            pd.Series(
                np.zeros(len(df))
            )
        ),
        errors="coerce"
    )

    valid_forecast = forecast.dropna()

    # --------------------------------------------------------
    # Priority distribution
    # --------------------------------------------------------

    priority_column = (
        df[
            "emergency_priority"
        ]
        .fillna("UNKNOWN")
    )

    risk_distribution = (
        priority_column
        .value_counts()
        .to_dict()
    )

    critical = int(
        (
            priority_column
            == "CRITICAL"
        ).sum()
    )

    high = int(
        (
            priority_column
            == "HIGH"
        ).sum()
    )

    medium = int(
        (
            priority_column
            == "MEDIUM"
        ).sum()
    )

    low = int(
        (
            priority_column
            == "LOW"
        ).sum()
    )

    # --------------------------------------------------------
    # Resource gap
    # --------------------------------------------------------

    required = pd.to_numeric(
        df.get(
            "resources_required",
            pd.Series(
                np.zeros(len(df))
            )
        ),
        errors="coerce"
    ).fillna(0)

    allocated = pd.to_numeric(
        df.get(
            "allocated_resources",
            pd.Series(
                np.zeros(len(df))
            )
        ),
        errors="coerce"
    ).fillna(0)

    resource_gap = (
        required - allocated
    )

    shortage = int(
        (
            resource_gap > 0
        ).sum()
    )

    # --------------------------------------------------------
    # Forecast coverage
    # --------------------------------------------------------

    forecast_nodes = int(
        forecast.notna().sum()
    )

    # --------------------------------------------------------
    # Average emergency score
    # --------------------------------------------------------

    average_score = pd.to_numeric(
        df[
            "emergency_priority_score"
        ],
        errors="coerce"
    ).fillna(0).mean()

    # --------------------------------------------------------
    # Forecast time
    # --------------------------------------------------------

    forecast_time = None

    if (
        "forecast_time" in df.columns
        and forecast_nodes > 0
    ):

        valid_times = (
            df["forecast_time"]
            .dropna()
        )

        if len(valid_times) > 0:

            forecast_time = str(
                valid_times.iloc[0]
            )

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return {

        "total_nodes":
            int(len(df)),

        "average_flood_probability":
            float(
                flood_probability.mean()
            ),

        "maximum_flood_probability":
            float(
                flood_probability.max()
            ),

        "stgnn_forecast_nodes":
            forecast_nodes,

        "stgnn_forecast_coverage_percent":
            float(
                (
                    forecast_nodes
                    /
                    max(len(df), 1)
                )
                * 100
            ),

        "stgnn_average_forecast":
            float(
                valid_forecast.mean()
            )
            if len(valid_forecast) > 0
            else 0.0,

        "stgnn_max_forecast":
            float(
                valid_forecast.max()
            )
            if len(valid_forecast) > 0
            else 0.0,

        "stgnn_min_forecast":
            float(
                valid_forecast.min()
            )
            if len(valid_forecast) > 0
            else 0.0,

        "latest_forecast_time":
            forecast_time,

        "average_emergency_score":
            float(
                average_score
            ),

        "critical_locations":
            critical,

        "high_priority_locations":
            high,

        "medium_priority_locations":
            medium,

        "low_priority_locations":
            low,

        "resource_shortage_locations":
            shortage,

        "total_resources_required":
            float(
                required.sum()
            ),

        "total_resources_allocated":
            float(
                allocated.sum()
            ),

        "total_resource_gap":
            float(
                resource_gap[
                    resource_gap > 0
                ].sum()
            ),

        "risk_distribution": {

            str(k): int(v)

            for k, v
            in risk_distribution.items()

        }

    }


# ============================================================
# SYSTEM INSIGHT
# ============================================================

@app.get("/insight")
def get_system_insight():

    if df.empty:

        return {
            "message":
                "No disaster nodes available."
        }

    top = df.iloc[0]

    return {

        "headline":
            "AI has identified the highest-priority "
            "emergency location.",

        "place_name":
            str(
                top.get(
                    "place_name",
                    "Unknown"
                )
            ),

        "grid_id":
            str(
                top.get(
                    "grid_id",
                    "Unknown"
                )
            ),

        "node_index":
            int(
                top.get(
                    "node_index",
                    0
                )
            )
            if pd.notna(
                top.get(
                    "node_index",
                    0
                )
            )
            else None,

        "priority":
            str(
                top.get(
                    "emergency_priority",
                    "UNKNOWN"
                )
            ),

        "score":
            float(
                top.get(
                    "emergency_priority_score",
                    0
                )
            ),

        "forecast":
            float(
                top.get(
                    "forecast_probability_percent",
                    0
                )
            ),

        "current_flood_probability":
            float(
                pd.to_numeric(
                    top.get(
                        "flood_probability",
                        0
                    ),
                    errors="coerce"
                )
                if pd.notna(
                    top.get(
                        "flood_probability",
                        0
                    )
                )
                else 0
            ),

        "reason":
            str(
                top.get(
                    "urgency_reason",
                    ""
                )
            ),

        "action":
            str(
                top.get(
                    "recommended_action",
                    ""
                )
            )

    }


# ============================================================
# RISK THRESHOLDS
# ============================================================

@app.get("/risk-thresholds")
def get_risk_thresholds():

    probabilities = pd.to_numeric(
        df[
            "flood_probability"
        ],
        errors="coerce"
    )

    valid = probabilities.dropna()

    if len(valid) == 0:

        return {

            "low_max": 0.0,
            "medium_max": 0.0,
            "high_max": 0.0,
            "critical_min": 0.0

        }

    return {

        "low_max":
            float(
                valid.quantile(
                    0.50
                )
            ),

        "medium_max":
            float(
                valid.quantile(
                    0.75
                )
            ),

        "high_max":
            float(
                valid.quantile(
                    0.90
                )
            ),

        "critical_min":
            float(
                valid.quantile(
                    0.90
                )
            )

    }


# ============================================================
# ST-GNN STATUS
# ============================================================

@app.get("/stgnn-status")
def get_stgnn_status():

    if stgnn_predictions.empty:

        return {

            "available": False,

            "message":
                "ST-GNN prediction file is empty or unavailable."

        }

    forecast_column = (
        "forecast_flood_probability"
    )

    if forecast_column not in df.columns:

        return {

            "available": False,

            "message":
                "Forecast column was not created."

        }

    matched = int(
        df[
            forecast_column
        ].notna().sum()
    )

    total = len(df)

    forecast_values = pd.to_numeric(
        df[
            forecast_column
        ],
        errors="coerce"
    ).dropna()

    latest_sequence = None

    if (
        "sequence_index"
        in stgnn_predictions.columns
    ):

        latest_sequence = int(
            pd.to_numeric(
                stgnn_predictions[
                    "sequence_index"
                ],
                errors="coerce"
            ).max()
        )

    latest_time = None

    if (
        "forecast_time"
        in df.columns
        and matched > 0
    ):

        times = (
            df[
                "forecast_time"
            ]
            .dropna()
        )

        if len(times) > 0:

            latest_time = str(
                times.iloc[0]
            )

    return {

        "available": True,

        "prediction_records":
            int(
                len(
                    stgnn_predictions
                )
            ),

        "main_nodes":
            int(total),

        "matched_nodes":
            matched,

        "unmatched_nodes":
            int(
                total - matched
            ),

        "coverage_percent":
            float(
                (
                    matched
                    /
                    max(total, 1)
                )
                * 100
            ),

        "latest_sequence":
            latest_sequence,

        "latest_forecast_time":
            latest_time,

        "forecast_min":
            float(
                forecast_values.min()
            )
            if len(forecast_values) > 0
            else None,

        "forecast_max":
            float(
                forecast_values.max()
            )
            if len(forecast_values) > 0
            else None,

        "forecast_mean":
            float(
                forecast_values.mean()
            )
            if len(forecast_values) > 0
            else None

    }


# ============================================================
# NODE MAPPING STATUS
# ============================================================

@app.get("/node-status")
def get_node_status():

    node_values = pd.to_numeric(
        df["node_index"],
        errors="coerce"
    ).dropna()

    return {

        "main_nodes":
            int(len(df)),

        "unique_node_indices":
            int(node_values.nunique()),

        "node_min":
            int(node_values.min())
            if len(node_values) > 0
            else None,

        "node_max":
            int(node_values.max())
            if len(node_values) > 0
            else None,

        "missing_node_indices":
            int(
                len(df)
                -
                len(node_values)
            ),

        "spatial_graph_nodes":
            int(
                len(spatial_nodes)
            ),

        "mapping_valid":
            bool(
                len(node_values) == len(df)
                and node_values.nunique() == len(df)
            )

    }


# ============================================================
# STARTUP INFORMATION
# ============================================================

print()
print("=" * 70)
print(
    "DISASTERFLOW AI - EMERGENCY DECISION SUPPORT API"
)
print("=" * 70)

print(
    f"Nodes loaded       : {len(df):,}"
)

print(
    f"Spatial nodes      : {len(spatial_nodes):,}"
)

print(
    f"ST-GNN predictions : "
    f"{len(stgnn_predictions):,}"
)

print(
    f"Place column       : "
    f"{PLACE_COLUMN}"
)

forecast_matches = int(
    df[
        "forecast_flood_probability"
    ].notna().sum()
)

print(
    f"ST-GNN node match  : "
    f"{forecast_matches}/{len(df)}"
)

if len(df) > 0:

    top = df.iloc[0]

    print(
        f"#1 Emergency      : "
        f"{top['place_name']}"
    )

    print(
        f"Grid ID            : "
        f"{top.get('grid_id', 'N/A')}"
    )

    print(
        f"Node index         : "
        f"{top.get('node_index', 'N/A')}"
    )

    print(
        f"Priority           : "
        f"{top['emergency_priority']}"
    )

    print(
        f"Score              : "
        f"{top['emergency_priority_score']:.2f}/100"
    )

    print(
        f"ST-GNN Forecast    : "
        f"{top['forecast_probability_percent']:.4f}%"
    )

print("=" * 70)
print(
    "DisasterFlow AI API ready."
)
print("=" * 70)