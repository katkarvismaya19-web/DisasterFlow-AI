import os
import json
import numpy as np
import pandas as pd

# ============================================================
# DISASTERFLOW AI — PHASE 5C.3
# ST-GNN Temporal / Node Distribution Shift Diagnosis
# ============================================================

print("=" * 75)
print("DISASTERFLOW AI — PHASE 5C.3")
print("ST-GNN Distribution Shift Diagnosis")
print("=" * 75)

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

GNN_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "gnn"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

NUM_SEQUENCES = 17491
SEQUENCE_LENGTH = 24
NUM_NODES = 140
NUM_FEATURES = 38

X_FILE = os.path.join(
    GNN_DIR,
    "X_sequences.dat"
)

Y_FILE = os.path.join(
    GNN_DIR,
    "Y_targets.dat"
)

SPLIT_FILE = os.path.join(
    GNN_DIR,
    "sequence_splits.npy"
)

TIME_FILE = os.path.join(
    GNN_DIR,
    "sequence_times.npy"
)

# ============================================================
# LOAD
# ============================================================

print("\n[1/8] Loading data...")

X = np.memmap(
    X_FILE,
    dtype=np.float32,
    mode="r",
    shape=(
        NUM_SEQUENCES,
        SEQUENCE_LENGTH,
        NUM_NODES,
        NUM_FEATURES
    )
)

Y = np.memmap(
    Y_FILE,
    dtype=np.int8,
    mode="r",
    shape=(
        NUM_SEQUENCES,
        NUM_NODES
    )
)

splits = np.load(SPLIT_FILE)
times = np.load(TIME_FILE)

train_idx = np.where(splits == "train")[0]
val_idx = np.where(splits == "validation")[0]
test_idx = np.where(splits == "test")[0]

print("X:", X.shape)
print("Y:", Y.shape)

print(
    "Train:",
    len(train_idx),
    "| Validation:",
    len(val_idx),
    "| Test:",
    len(test_idx)
)

# ============================================================
# BASIC TEMPORAL DISTRIBUTION
# ============================================================

print("\n[2/8] Temporal distribution...")

for name, idx in [
    ("TRAIN", train_idx),
    ("VALIDATION", val_idx),
    ("TEST", test_idx)
]:

    print("\n" + "-" * 60)
    print(name)

    print(
        "First:",
        times[idx[0]]
    )

    print(
        "Last :",
        times[idx[-1]]
    )

    positives = int(Y[idx].sum())
    total = len(idx) * NUM_NODES

    print(
        "Positive labels:",
        positives
    )

    print(
        "Positive rate:",
        round(positives / total, 8)
    )

# ============================================================
# POSITIVES BY NODE
# ============================================================

print("\n[3/8] Positive labels by node...")

for name, idx in [
    ("TRAIN", train_idx),
    ("VALIDATION", val_idx),
    ("TEST", test_idx)
]:

    counts = Y[idx].sum(axis=0)

    order = np.argsort(counts)[::-1]

    print("\n" + "-" * 60)
    print(name)

    print(
        "Top 15 nodes by positive count:"
    )

    for node in order[:15]:

        print(
            f"Node {node:3d}: "
            f"{int(counts[node]):5d}"
        )

# ============================================================
# NODE COVERAGE
# ============================================================

print("\n[4/8] Node coverage...")

for name, idx in [
    ("TRAIN", train_idx),
    ("VALIDATION", val_idx),
    ("TEST", test_idx)
]:

    counts = Y[idx].sum(axis=0)

    print(
        f"{name}: "
        f"{np.sum(counts > 0)}/"
        f"{NUM_NODES} nodes have positives"
    )

# ============================================================
# SPECIFIC NODE 49
# ============================================================

print("\n[5/8] Investigating node 49...")

node = 49

for name, idx in [
    ("TRAIN", train_idx),
    ("VALIDATION", val_idx),
    ("TEST", test_idx)
]:

    labels = Y[idx, node]

    print("\n" + "-" * 60)
    print(name)

    print(
        "Positive:",
        int(labels.sum())
    )

    print(
        "Total:",
        len(labels)
    )

    print(
        "Positive rate:",
        round(labels.mean(), 8)
    )

# ============================================================
# FEATURE DISTRIBUTION SHIFT
# ============================================================

print("\n[6/8] Feature distribution shift...")

# Use last timestep because prediction is based on recent state.
# Sampling keeps this CPU-friendly.

SAMPLE_SIZE = 1000

rng = np.random.default_rng(42)

def sample_indices(idx):

    if len(idx) <= SAMPLE_SIZE:
        return idx

    return rng.choice(
        idx,
        size=SAMPLE_SIZE,
        replace=False
    )

train_sample = sample_indices(train_idx)
val_sample = sample_indices(val_idx)
test_sample = sample_indices(test_idx)

# Last timestep
train_last = np.asarray(
    X[train_sample, -1, :, :]
)

val_last = np.asarray(
    X[val_sample, -1, :, :]
)

test_last = np.asarray(
    X[test_sample, -1, :, :]
)

print(
    "\nComparing feature means:"
)

rows = []

for f in range(NUM_FEATURES):

    train_mean = float(
        np.mean(train_last[:, :, f])
    )

    val_mean = float(
        np.mean(val_last[:, :, f])
    )

    test_mean = float(
        np.mean(test_last[:, :, f])
    )

    train_std = float(
        np.std(train_last[:, :, f])
    )

    rows.append({
        "feature": f,
        "train_mean": train_mean,
        "validation_mean": val_mean,
        "test_mean": test_mean,
        "train_std": train_std,
        "val_minus_train": val_mean - train_mean,
        "test_minus_train": test_mean - train_mean
    })

feature_shift = pd.DataFrame(rows)

feature_shift["abs_test_shift"] = (
    feature_shift["test_minus_train"].abs()
)

feature_shift = feature_shift.sort_values(
    "abs_test_shift",
    ascending=False
)

print(
    feature_shift.head(15).to_string(
        index=False
    )
)

feature_shift_file = os.path.join(
    MODEL_DIR,
    "stgnn_feature_distribution_shift.csv"
)

feature_shift.to_csv(
    feature_shift_file,
    index=False
)

# ============================================================
# NODE FEATURE SHIFT
# ============================================================

print("\n[7/8] Node 49 feature comparison...")

node_rows = []

for f in range(NUM_FEATURES):

    train_values = train_last[:, node, f]
    val_values = val_last[:, node, f]
    test_values = test_last[:, node, f]

    node_rows.append({
        "feature": f,
        "train_mean": float(
            np.mean(train_values)
        ),
        "validation_mean": float(
            np.mean(val_values)
        ),
        "test_mean": float(
            np.mean(test_values)
        ),
        "train_std": float(
            np.std(train_values)
        ),
        "validation_std": float(
            np.std(val_values)
        ),
        "test_std": float(
            np.std(test_values)
        )
    })

node_shift = pd.DataFrame(node_rows)

node_shift["abs_test_shift"] = (
    node_shift["test_mean"] -
    node_shift["train_mean"]
).abs()

node_shift = node_shift.sort_values(
    "abs_test_shift",
    ascending=False
)

print(
    node_shift.head(15).to_string(
        index=False
    )
)

node_shift_file = os.path.join(
    MODEL_DIR,
    "stgnn_node49_feature_shift.csv"
)

node_shift.to_csv(
    node_shift_file,
    index=False
)

# ============================================================
# EVENT CONCENTRATION
# ============================================================

print("\n[8/8] Event concentration analysis...")

for name, idx in [
    ("TRAIN", train_idx),
    ("VALIDATION", val_idx),
    ("TEST", test_idx)
]:

    counts = Y[idx].sum(axis=0)

    total_positive = counts.sum()

    if total_positive > 0:

        top5 = np.sort(
            counts
        )[-5:][::-1]

        print(
            f"{name} top-5 node positive share:",
            round(
                top5.sum() /
                total_positive,
                4
            )
        )

# ============================================================
# SAVE SUMMARY
# ============================================================

summary = {

    "train_sequences": int(len(train_idx)),
    "validation_sequences": int(len(val_idx)),
    "test_sequences": int(len(test_idx)),

    "train_positive": int(
        Y[train_idx].sum()
    ),

    "validation_positive": int(
        Y[val_idx].sum()
    ),

    "test_positive": int(
        Y[test_idx].sum()
    ),

    "node49_train_positive": int(
        Y[train_idx, 49].sum()
    ),

    "node49_validation_positive": int(
        Y[val_idx, 49].sum()
    ),

    "node49_test_positive": int(
        Y[test_idx, 49].sum()
    ),

    "feature_shift_file": feature_shift_file,
    "node49_shift_file": node_shift_file
}

summary_file = os.path.join(
    MODEL_DIR,
    "stgnn_distribution_shift_summary.json"
)

with open(
    summary_file,
    "w"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )

print("\nSaved:")
print(feature_shift_file)
print(node_shift_file)
print(summary_file)

print("\n" + "=" * 75)
print("PHASE 5C.3 COMPLETE")
print("=" * 75)