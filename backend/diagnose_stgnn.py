import os
import json
import numpy as np

# ============================================================
# DISASTERFLOW AI
# ST-GNN DATA + SPLIT DIAGNOSTIC
# ============================================================

print("=" * 80)
print("DISASTERFLOW AI")
print("ST-GNN DATA + SPLIT DIAGNOSTIC")
print("=" * 80)

# ============================================================
# PATHS
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

META_FILE = os.path.join(
    GNN_DIR,
    "gnn_metadata.json"
)

X_FILE = os.path.join(
    GNN_DIR,
    "X_sequences.dat"
)

Y_FILE = os.path.join(
    GNN_DIR,
    "Y_targets.dat"
)

EDGE_FILE = os.path.join(
    GNN_DIR,
    "edge_index.npy"
)

SPLIT_FILE = os.path.join(
    GNN_DIR,
    "sequence_splits.npy"
)

# ============================================================
# LOAD METADATA
# ============================================================

print("\n[1] Loading metadata...")

with open(META_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)

print("Metadata:")
for key, value in metadata.items():
    print(f"  {key}: {value}")

NUM_NODES = int(metadata["num_nodes"])
NUM_FEATURES = int(metadata["num_features"])
SEQ_LEN = int(metadata["sequence_length"])
NUM_SEQUENCES = int(metadata["num_sequences"])

X_SHAPE = tuple(metadata["X_shape"])
Y_SHAPE = tuple(metadata["Y_shape"])

# ============================================================
# LOAD MEMMAP
# ============================================================

print("\n[2] Loading X/Y...")

X = np.memmap(
    X_FILE,
    dtype=np.float32,
    mode="r",
    shape=X_SHAPE
)

Y = np.memmap(
    Y_FILE,
    dtype=np.int8,
    mode="r",
    shape=Y_SHAPE
)

print("X shape:", X.shape)
print("Y shape:", Y.shape)

# ============================================================
# BASIC DATA CHECK
# ============================================================

print("\n[3] Basic data statistics")

print("X dtype:", X.dtype)
print("Y dtype:", Y.dtype)

print("X min:", float(np.nanmin(X)))
print("X max:", float(np.nanmax(X)))
print("X mean:", float(np.nanmean(X)))
print("X std:", float(np.nanstd(X)))

print("Y unique:", np.unique(Y))

positive_total = int(np.asarray(Y).sum())
total_targets = int(Y.shape[0] * Y.shape[1])
negative_total = total_targets - positive_total

print("Total targets:", total_targets)
print("Positive:", positive_total)
print("Negative:", negative_total)

print(
    "Positive rate:",
    positive_total / max(total_targets, 1)
)

# ============================================================
# SPLITS
# ============================================================

print("\n[4] Sequence split diagnostic")

sequence_splits = np.load(
    SPLIT_FILE,
    allow_pickle=True
)

print("Split array shape:", sequence_splits.shape)
print("Split dtype:", sequence_splits.dtype)

unique_splits, split_counts = np.unique(
    sequence_splits,
    return_counts=True
)

for split_name, count in zip(
    unique_splits,
    split_counts
):
    print(
        f"  {split_name}: {count}"
    )

# ============================================================
# INDEX LOCATIONS
# ============================================================

print("\n[5] Split index ranges")

for split_name in [
    "train",
    "validation",
    "test"
]:

    indices = np.where(
        sequence_splits == split_name
    )[0]

    if len(indices) == 0:
        print(
            split_name,
            ": EMPTY"
        )
        continue

    print(
        f"{split_name:12s} "
        f"count={len(indices):6d} "
        f"min={indices.min():6d} "
        f"max={indices.max():6d}"
    )

# ============================================================
# CHECK CHRONOLOGICAL ORDER
# ============================================================

print("\n[6] Chronological split check")

train_idx = np.where(
    sequence_splits == "train"
)[0]

val_idx = np.where(
    sequence_splits == "validation"
)[0]

test_idx = np.where(
    sequence_splits == "test"
)[0]

if (
    len(train_idx) > 0
    and
    len(val_idx) > 0
    and
    len(test_idx) > 0
):

    print(
        "Train last index:",
        train_idx.max()
    )

    print(
        "Validation first index:",
        val_idx.min()
    )

    print(
        "Validation last index:",
        val_idx.max()
    )

    print(
        "Test first index:",
        test_idx.min()
    )

    print(
        "Test last index:",
        test_idx.max()
    )

    chronological = (
        train_idx.max()
        <
        val_idx.min()
        and
        val_idx.max()
        <
        test_idx.min()
    )

    print(
        "Chronological split:",
        chronological
    )

# ============================================================
# TARGET DISTRIBUTION BY SPLIT
# ============================================================

print("\n[7] Target distribution by split")

for split_name in [
    "train",
    "validation",
    "test"
]:

    indices = np.where(
        sequence_splits == split_name
    )[0]

    if len(indices) == 0:
        continue

    split_y = np.asarray(
        Y[indices]
    )

    positives = int(
        split_y.sum()
    )

    total = int(
        split_y.size
    )

    rate = positives / max(
        total,
        1
    )

    print(
        f"{split_name:12s} "
        f"positive={positives:8d} "
        f"total={total:10d} "
        f"rate={rate:.8f}"
    )

# ============================================================
# POSITIVE SAMPLES PER SEQUENCE
# ============================================================

print("\n[8] Positive nodes per sequence")

positive_per_sequence = np.asarray(
    Y.sum(axis=1)
)

print(
    "Min:",
    int(positive_per_sequence.min())
)

print(
    "Max:",
    int(positive_per_sequence.max())
)

print(
    "Mean:",
    float(positive_per_sequence.mean())
)

print(
    "Median:",
    float(np.median(positive_per_sequence))
)

print(
    "Sequences with >=1 positive:",
    int(
        np.sum(
            positive_per_sequence > 0
        )
    )
)

# ============================================================
# NODE-LEVEL DISTRIBUTION
# ============================================================

print("\n[9] Node-level target distribution")

node_positive_counts = np.asarray(
    Y.sum(axis=0)
)

print(
    "Most positive nodes:"
)

top_nodes = np.argsort(
    node_positive_counts
)[::-1][:15]

for node in top_nodes:
    print(
        f"  Node {node:3d}: "
        f"{int(node_positive_counts[node])}"
    )

print(
    "\nNodes with ZERO positives:",
    int(
        np.sum(
            node_positive_counts == 0
        )
    )
)

# ============================================================
# GRAPH CHECK
# ============================================================

print("\n[10] Graph diagnostic")

edge_index = np.load(
    EDGE_FILE
)

print(
    "edge_index shape:",
    edge_index.shape
)

print(
    "Number of edges:",
    edge_index.shape[1]
)

if edge_index.shape[1] > 0:

    print(
        "Minimum node index:",
        int(edge_index.min())
    )

    print(
        "Maximum node index:",
        int(edge_index.max())
    )

    invalid_edges = np.sum(
        (edge_index < 0)
        |
        (edge_index >= NUM_NODES)
    )

    print(
        "Invalid edge entries:",
        int(invalid_edges)
    )

# ============================================================
# SAMPLE DATA
# ============================================================

print("\n[11] Sample sequence inspection")

sample_indices = [
    0,
    len(X) // 2,
    len(X) - 1
]

for idx in sample_indices:

    x_sample = np.asarray(
        X[idx]
    )

    y_sample = np.asarray(
        Y[idx]
    )

    print("\nSequence:", idx)

    print(
        "X shape:",
        x_sample.shape
    )

    print(
        "Y shape:",
        y_sample.shape
    )

    print(
        "X mean:",
        float(x_sample.mean())
    )

    print(
        "X std:",
        float(x_sample.std())
    )

    print(
        "Positive nodes:",
        int(y_sample.sum())
    )

# ============================================================
# TEMPORAL FEATURE CHECK
# ============================================================

print("\n[12] Temporal feature variation")

sample = np.asarray(
    X[0]
)

# sample shape:
# T × N × F

for feature_idx in range(
    min(NUM_FEATURES, 10)
):

    feature_values = sample[
        :,
        :,
        feature_idx
    ]

    print(
        f"Feature {feature_idx:2d}: "
        f"mean={feature_values.mean():.6f} "
        f"std={feature_values.std():.6f} "
        f"min={feature_values.min():.6f} "
        f"max={feature_values.max():.6f}"
    )

# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)