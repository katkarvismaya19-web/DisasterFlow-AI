import requests
import pandas as pd
from pathlib import Path
import time


# ============================================================
# DISASTERFLOW AI
# PHASE 2A — HISTORICAL WEATHER COLLECTION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

GRID_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "disasterflow_final_node_priorities.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "raw"
    / "weather"
)

OUTPUT_FILE = OUTPUT_DIR / "mumbai_historical_weather.csv"


# ============================================================
# SETTINGS
# ============================================================

START_DATE = "2024-01-01"
END_DATE = "2025-12-31"

API_URL = "https://archive-api.open-meteo.com/v1/archive"


# ============================================================
# LOAD SPATIAL GRID
# ============================================================

print("=" * 60)
print("DISASTERFLOW AI — PHASE 2A")
print("Historical Weather Collection")
print("=" * 60)

if not GRID_FILE.exists():
    raise FileNotFoundError(
        f"Grid file not found: {GRID_FILE}"
    )

grid = pd.read_csv(GRID_FILE)

print(f"\nLoaded spatial grid: {len(grid)} nodes")


required_columns = [
    "grid_id",
    "latitude",
    "longitude"
]

for column in required_columns:
    if column not in grid.columns:
        raise ValueError(
            f"Missing required column: {column}"
        )


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DOWNLOAD WEATHER
# ============================================================

all_weather = []

total_nodes = len(grid)

for index, row in grid.iterrows():

    grid_id = row["grid_id"]
    latitude = float(row["latitude"])
    longitude = float(row["longitude"])

    print(
        f"\n[{index + 1}/{total_nodes}] "
        f"Downloading {grid_id} "
        f"({latitude:.4f}, {longitude:.4f})"
    )

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": START_DATE,
        "end_date": END_DATE,

        "hourly": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "precipitation,"
            "rain,"
            "wind_speed_10m"
        ),

        "timezone": "Asia/Kolkata",
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm"
    }

    try:

        response = requests.get(
            API_URL,
            params=params,
            timeout=60
        )

        response.raise_for_status()

        data = response.json()

        if "hourly" not in data:
            print(
                f"WARNING: No hourly data for {grid_id}"
            )
            continue

        hourly = pd.DataFrame(
            data["hourly"]
        )

        hourly["grid_id"] = grid_id

        hourly["latitude"] = latitude
        hourly["longitude"] = longitude

        all_weather.append(hourly)

        print(
            f"  ✓ Retrieved {len(hourly):,} hourly records"
        )

    except Exception as error:

        print(
            f"  ✗ Failed for {grid_id}: {error}"
        )

    # Small delay to avoid hammering the API
    time.sleep(0.2)


# ============================================================
# COMBINE DATA
# ============================================================

if not all_weather:

    raise RuntimeError(
        "No weather data was downloaded."
    )


weather = pd.concat(
    all_weather,
    ignore_index=True
)


# ============================================================
# CLEAN COLUMN ORDER
# ============================================================

weather = weather[
    [
        "grid_id",
        "latitude",
        "longitude",
        "time",
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "rain",
        "wind_speed_10m"
    ]
]


# ============================================================
# SORT
# ============================================================

weather["time"] = pd.to_datetime(
    weather["time"]
)

weather = weather.sort_values(
    [
        "grid_id",
        "time"
    ]
).reset_index(drop=True)


# ============================================================
# SAVE
# ============================================================

weather.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("WEATHER DOWNLOAD COMPLETE")
print("=" * 60)

print(
    f"\nOutput file:\n{OUTPUT_FILE}"
)

print(
    f"\nRows: {len(weather):,}"
)

print(
    f"Columns: {len(weather.columns)}"
)

print(
    f"\nNodes: {weather['grid_id'].nunique()}"
)

print(
    f"Start: {weather['time'].min()}"
)

print(
    f"End: {weather['time'].max()}"
)

print("\nMissing values:")
print(
    weather.isnull().sum()
)

print("\nSample:")
print(
    weather.head().to_string()
)

print("\n✓ Phase 2A weather collection finished.")