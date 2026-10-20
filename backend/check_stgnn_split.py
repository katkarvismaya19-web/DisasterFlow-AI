import os
import json
import numpy as np
import pandas as pd


# ============================================================
# DISASTERFLOW AI
# ST-GNN TEMPORAL SPLIT DIAGNOSTIC
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

GNN_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "gnn"
)

SPLIT_FILE = os.path.join(
    GNN_DIR,
    "sequence_splits.npy"
)

TIME_FILE = os.path.join(
    GNN_DIR,
    "sequence_times.npy"
)

Y_FILE = os.path.join(
    GNN_DIR,
    "Y_targets.dat"
)

METADATA_FILE = os.path.join(
    GNN_DIR,
    "gnn_metadata.json"
)


print("=" * 70)
print("DISASTERFLOW AI")
print("ST-GNN TEMPORAL SPLIT DIAGNOSTIC")
print("=" * 70)


# ============================================================
# LOAD METADATA
# ============================================================

with open(METADATA_FILE, "r") as f:
    metadata = json.load(f)

num_nodes = int(metadata["num_nodes"])
sequence_length = int(metadata["sequence_length"])

print("\nMetadata")
print("-" * 70)
print("Nodes:", num_nodes)
print("Sequence length:", sequence_length)


# ============================================================
# LOAD SPLITS / TIMES
# ============================================================

splits = np.load(SPLIT_FILE)
times = np.load(TIME_FILE)

print("\nSequence information")
print("-" * 70)
print("Number of sequences:", len(splits))
print("Number of timestamps:", len(times))


# ============================================================
# CONVERT TIMES
# ============================================================

times = pd.to_datetime(
    times,
    errors="coerce"
)

print("\nTimestamp range")
print("-" * 70)
print("First:", times.min())
print("Last :", times.max())


# ============================================================
# SPLIT SUMMARY
# ============================================================

print("\nSplit summary")
print("-" * 70)

for split_name in ["train", "validation", "test"]:

    indices = np.where(
        splits == split_name
    )[0]

    if len(indices) == 0:
        print(split_name, ": EMPTY")
        continue

    split_times = times[indices]

    print(
        f"\n{split_name.upper()}"
    )

    print(
        "Sequences:",
        len(indices)
    )

    print(
        "First timestamp:",
        split_times.min()
    )

    print(
        "Last timestamp:",
        split_times.max()
    )

    print(
        "Unique days:",
        split_times.normalize().nunique()
    )

    print(
        "Unique months:",
        split_times.to_period("M").nunique()
    )


# ============================================================
# CHECK CHRONOLOGICAL ORDER
# ============================================================

print("\n" + "=" * 70)
print("CHRONOLOGICAL ORDER CHECK")
print("=" * 70)

split_order = [
    "train",
    "validation",
    "test"
]

previous_max = None

for split_name in split_order:

    indices = np.where(
        splits == split_name
    )[0]

    if len(indices) == 0:
        continue

    current_min = times[indices].min()
    current_max = times[indices].max()

    print(
        f"{split_name:12s}: "
        f"{current_min} -> {current_max}"
    )

    if previous_max is not None:

        if current_min <= previous_max:

            print(
                "WARNING: temporal overlap detected!"
            )

    previous_max = current_max


# ============================================================
# LOAD TARGETS
# ============================================================

NUM_SEQUENCES = len(splits)

Y = np.memmap(
    Y_FILE,
    dtype=np.int8,
    mode="r",
    shape=(
        NUM_SEQUENCES,
        num_nodes
    )
)


# ============================================================
# TARGET DISTRIBUTION BY SPLIT
# ============================================================

print("\n" + "=" * 70)
print("TARGET DISTRIBUTION")
print("=" * 70)

for split_name in split_order:

    indices = np.where(
        splits == split_name
    )[0]

    if len(indices) == 0:
        continue

    y_split = Y[indices]

    positives = int(
        y_split.sum()
    )

    total = (
        len(indices) *
        num_nodes
    )

    negatives = (
        total -
        positives
    )

    prevalence = (
        positives / total
        if total > 0
        else 0
    )

    print(
        f"\n{split_name.upper()}"
    )

    print(
        "Samples:",
        len(indices)
    )

    print(
        "Positive:",
        positives
    )

    print(
        "Negative:",
        negatives
    )

    print(
        "Positive rate:",
        round(prevalence, 6)
    )


# ============================================================
# MONTHLY TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("MONTHLY TARGET DISTRIBUTION")
print("=" * 70)

summary_rows = []

for split_name in split_order:

    indices = np.where(
        splits == split_name
    )[0]

    if len(indices) == 0:
        continue

    split_times = times[indices]

    y_split = Y[indices]

    months = (
        split_times
        .to_period("M")
        .astype(str)
    )

    unique_months = sorted(
        np.unique(months)
    )

    for month in unique_months:

        month_mask = (
            months == month
        )

        y_month = y_split[
            month_mask
        ]

        positives = int(
            y_month.sum()
        )

        total = (
            y_month.shape[0] *
            num_nodes
        )

        summary_rows.append({
            "split": split_name,
            "month": month,
            "sequences": int(
                y_month.shape[0]
            ),
            "positive": positives,
            "negative": int(
                total - positives
            ),
            "positive_rate": (
                positives / total
                if total > 0
                else 0
            )
        })


monthly_df = pd.DataFrame(
    summary_rows
)

print(
    monthly_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "models",
    "stgnn_split_diagnostic.csv"
)

monthly_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved:")
print(OUTPUT_FILE)

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)