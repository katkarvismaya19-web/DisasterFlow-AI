import os
import json
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# DISASTERFLOW AI — PHASE 5C
# Spatiotemporal Graph Neural Network
# ============================================================

print("=" * 70)
print("DISASTERFLOW AI — PHASE 5C")
print("Spatiotemporal GNN Training")
print("=" * 70)


# ============================================================
# CONFIGURATION
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

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(MODEL_DIR, exist_ok=True)


NUM_NODES = 140
NUM_FEATURES = 38

SEQUENCE_LENGTH = 24
HIDDEN_SIZE = 32

BATCH_SIZE = 16
EPOCHS = 20

LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5

PATIENCE = 4

SEED = 42

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

TIME_FILE = os.path.join(
    GNN_DIR,
    "sequence_times.npy"
)

METADATA_FILE = os.path.join(
    GNN_DIR,
    "gnn_metadata.json"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


DEVICE = torch.device("cpu")

print("\nDevice:", DEVICE)
print("PyTorch:", torch.__version__)

if torch.cuda.is_available():
    print("CUDA available:", True)
else:
    print("CUDA available:", False)
    print("✓ CPU-safe training mode")


# ============================================================
# STEP 1 — LOAD METADATA
# ============================================================

print("\n[1/9] Loading GNN dataset metadata...")


with open(METADATA_FILE, "r") as f:
    metadata = json.load(f)


print("Nodes:", metadata["num_nodes"])
print("Features:", metadata["num_features"])
print("Sequence length:", metadata["sequence_length"])
print("Forecast horizon:", metadata["forecast_horizon"])
print("Edges:", metadata["num_edges"])


if metadata["num_nodes"] != NUM_NODES:
    raise ValueError("Unexpected number of nodes")

if metadata["num_features"] != NUM_FEATURES:
    raise ValueError("Unexpected number of features")


# ============================================================
# STEP 2 — MEMORY-MAPPED DATA
# ============================================================

print("\n[2/9] Opening memory-mapped datasets...")


NUM_SEQUENCES = 17491


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


print("X:", X.shape)
print("Y:", Y.shape)

print("X size:",
      round(X.nbytes / (1024 ** 3), 2),
      "GB")


# ============================================================
# STEP 3 — LOAD GRAPH
# ============================================================

print("\n[3/9] Loading spatial graph...")


edge_index = np.load(
    EDGE_FILE
)

print("Edge index:", edge_index.shape)

if edge_index.shape[0] != 2:
    raise ValueError(
        "edge_index must have shape (2, E)"
    )

if edge_index.min() < 0:
    raise ValueError(
        "Negative node index found"
    )

if edge_index.max() >= NUM_NODES:
    raise ValueError(
        "Node index exceeds node count"
    )


edge_index = torch.tensor(
    edge_index,
    dtype=torch.long,
    device=DEVICE
)

print("✓ Graph validated")


# ============================================================
# STEP 4 — SPLITS
# ============================================================

print("\n[4/9] Loading temporal splits...")


sequence_splits = np.load(
    SPLIT_FILE
)

sequence_times = np.load(
    TIME_FILE
)


train_indices = np.where(
    sequence_splits == "train"
)[0]

validation_indices = np.where(
    sequence_splits == "validation"
)[0]

test_indices = np.where(
    sequence_splits == "test"
)[0]


print("Train:", len(train_indices))
print("Validation:", len(validation_indices))
print("Test:", len(test_indices))


# ============================================================
# DATASET
# ============================================================

class MemmapDataset(Dataset):

    def __init__(
        self,
        X,
        Y,
        indices
    ):

        self.X = X
        self.Y = Y
        self.indices = indices

    def __len__(self):

        return len(self.indices)

    def __getitem__(self, idx):

        real_idx = self.indices[idx]

        x = np.asarray(
            self.X[real_idx],
            dtype=np.float32
        )

        y = np.asarray(
            self.Y[real_idx],
            dtype=np.float32
        )

        return (
            torch.from_numpy(x),
            torch.from_numpy(y)
        )


train_dataset = MemmapDataset(
    X,
    Y,
    train_indices
)

validation_dataset = MemmapDataset(
    X,
    Y,
    validation_indices
)

test_dataset = MemmapDataset(
    X,
    Y,
    test_indices
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# CLASS IMBALANCE
# ============================================================

print("\nCalculating class imbalance...")


train_positive = int(
    Y[train_indices].sum()
)

train_total = (
    len(train_indices) *
    NUM_NODES
)

train_negative = (
    train_total -
    train_positive
)

pos_weight_value = (
    train_negative /
    max(train_positive, 1)
)


print("Positive:", train_positive)
print("Negative:", train_negative)

print(
    "Positive weight:",
    round(pos_weight_value, 3)
)


# ============================================================
# GRAPH AGGREGATION
# ============================================================

class GraphSAGELayer(nn.Module):

    def __init__(
        self,
        in_features,
        out_features
    ):

        super().__init__()

        self.linear_self = nn.Linear(
            in_features,
            out_features
        )

        self.linear_neigh = nn.Linear(
            in_features,
            out_features
        )

        self.activation = nn.ReLU()

    def forward(
        self,
        x,
        edge_index
    ):

        # x:
        # [batch, nodes, features]

        source = edge_index[0]
        target = edge_index[1]

        neighbor = torch.zeros_like(x)

        neighbor.index_add_(
            1,
            target,
            x[:, source, :]
        )

        degree = torch.zeros(
            NUM_NODES,
            device=x.device
        )

        degree.index_add_(
            0,
            target,
            torch.ones(
                target.shape[0],
                device=x.device
            )
        )

        degree = degree.clamp(
            min=1
        )

        neighbor = (
            neighbor /
            degree.view(1, -1, 1)
        )

        out = (
            self.linear_self(x) +
            self.linear_neigh(neighbor)
        )

        return self.activation(out)


# ============================================================
# SPATIOTEMPORAL GNN
# ============================================================

class STGNN(nn.Module):

    def __init__(
        self,
        num_features,
        hidden_size
    ):

        super().__init__()

        self.gnn = GraphSAGELayer(
            num_features,
            hidden_size
        )

        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.dropout = nn.Dropout(
            0.2
        )

        self.output = nn.Linear(
            hidden_size,
            1
        )

    def forward(
        self,
        x,
        edge_index
    ):

        # Input:
        #
        # [batch, time, nodes, features]

        batch_size = x.shape[0]

        temporal_outputs = []

        for t in range(
            SEQUENCE_LENGTH
        ):

            xt = x[:, t, :, :]

            spatial = self.gnn(
                xt,
                edge_index
            )

            temporal_outputs.append(
                spatial
            )

        # [batch, time, nodes, hidden]

        spatial_sequence = torch.stack(
            temporal_outputs,
            dim=1
        )

        # Rearrange:
        #
        # [batch, nodes, time, hidden]

        spatial_sequence = spatial_sequence.permute(
            0,
            2,
            1,
            3
        )

        # Merge batch and nodes

        spatial_sequence = spatial_sequence.reshape(
            batch_size * NUM_NODES,
            SEQUENCE_LENGTH,
            HIDDEN_SIZE
        )

        temporal_output, _ = self.gru(
            spatial_sequence
        )

        last_hidden = temporal_output[:, -1, :]

        last_hidden = self.dropout(
            last_hidden
        )

        logits = self.output(
            last_hidden
        )

        logits = logits.reshape(
            batch_size,
            NUM_NODES
        )

        return logits


# ============================================================
# STEP 5 — MODEL
# ============================================================

print("\n[5/9] Creating ST-GNN...")


model = STGNN(
    num_features=NUM_FEATURES,
    hidden_size=HIDDEN_SIZE
).to(DEVICE)


parameter_count = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)


print("Trainable parameters:", parameter_count)

print(model)


# ============================================================
# LOSS + OPTIMIZER
# ============================================================

pos_weight = torch.tensor(
    [pos_weight_value],
    dtype=torch.float32,
    device=DEVICE
)

criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate(
    model,
    loader
):

    model.eval()

    all_probabilities = []
    all_targets = []

    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():

        for X_batch, Y_batch in loader:

            X_batch = X_batch.to(
                DEVICE
            )

            Y_batch = Y_batch.to(
                DEVICE
            )

            logits = model(
                X_batch,
                edge_index
            )

            loss = criterion(
                logits,
                Y_batch
            )

            probabilities = torch.sigmoid(
                logits
            )

            all_probabilities.append(
                probabilities.cpu().numpy()
            )

            all_targets.append(
                Y_batch.cpu().numpy()
            )

            total_loss += loss.item()
            total_batches += 1

    probabilities = np.concatenate(
        all_probabilities
    ).ravel()

    targets = np.concatenate(
        all_targets
    ).ravel()

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    precision = precision_score(
        targets,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        targets,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        targets,
        predictions,
        zero_division=0
    )

    pr_auc = average_precision_score(
        targets,
        probabilities
    )

    roc_auc = roc_auc_score(
        targets,
        probabilities
    )

    cm = confusion_matrix(
        targets,
        predictions
    )

    return {
        "loss": total_loss / max(total_batches, 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "probabilities": probabilities,
        "targets": targets
    }


# ============================================================
# STEP 6 — TRAINING
# ============================================================

print("\n[6/9] Training ST-GNN...")

best_pr_auc = -np.inf

patience_counter = 0

best_model_file = os.path.join(
    MODEL_DIR,
    "stgnn_best_model.pt"
)

history = []


for epoch in range(
    1,
    EPOCHS + 1
):

    model.train()

    running_loss = 0.0
    batches = 0

    for X_batch, Y_batch in train_loader:

        X_batch = X_batch.to(
            DEVICE
        )

        Y_batch = Y_batch.to(
            DEVICE
        )

        optimizer.zero_grad()

        logits = model(
            X_batch,
            edge_index
        )

        loss = criterion(
            logits,
            Y_batch
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        running_loss += loss.item()

        batches += 1

    train_loss = (
        running_loss /
        max(batches, 1)
    )

    validation_metrics = evaluate(
        model,
        validation_loader
    )

    print(
        f"\nEpoch {epoch}/{EPOCHS}"
    )

    print(
        f"Train Loss : {train_loss:.6f}"
    )

    print(
        f"Val Loss   : "
        f"{validation_metrics['loss']:.6f}"
    )

    print(
        f"Val PR-AUC : "
        f"{validation_metrics['pr_auc']:.6f}"
    )

    print(
        f"Val ROC-AUC: "
        f"{validation_metrics['roc_auc']:.6f}"
    )

    print(
        f"Val Recall : "
        f"{validation_metrics['recall']:.6f}"
    )

    print(
        f"Val F1     : "
        f"{validation_metrics['f1']:.6f}"
    )

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": validation_metrics["loss"],
        "val_pr_auc": validation_metrics["pr_auc"],
        "val_roc_auc": validation_metrics["roc_auc"],
        "val_precision": validation_metrics["precision"],
        "val_recall": validation_metrics["recall"],
        "val_f1": validation_metrics["f1"]
    })

    if (
        validation_metrics["pr_auc"]
        >
        best_pr_auc
    ):

        best_pr_auc = (
            validation_metrics["pr_auc"]
        )

        patience_counter = 0

        torch.save(
            model.state_dict(),
            best_model_file
        )

        print(
            "✓ Best model saved"
        )

    else:

        patience_counter += 1

        print(
            "No improvement:",
            patience_counter,
            "/",
            PATIENCE
        )

        if patience_counter >= PATIENCE:

            print(
                "✓ Early stopping"
            )

            break


# ============================================================
# STEP 7 — LOAD BEST MODEL
# ============================================================

print("\n[7/9] Loading best ST-GNN...")


model.load_state_dict(
    torch.load(
        best_model_file,
        map_location=DEVICE
    )
)

model.eval()

print(
    "Best validation PR-AUC:",
    round(best_pr_auc, 6)
)


# ============================================================
# STEP 8 — TEST
# ============================================================

print("\n[8/9] Evaluating untouched test set...")


test_metrics = evaluate(
    model,
    test_loader
)


print("\n" + "-" * 70)
print("ST-GNN — TEST PERFORMANCE")
print("-" * 70)

print(
    "Precision:",
    round(test_metrics["precision"], 6)
)

print(
    "Recall   :",
    round(test_metrics["recall"], 6)
)

print(
    "F1       :",
    round(test_metrics["f1"], 6)
)

print(
    "ROC-AUC  :",
    round(test_metrics["roc_auc"], 6)
)

print(
    "PR-AUC   :",
    round(test_metrics["pr_auc"], 6)
)

print("\nConfusion Matrix:")

print(
    test_metrics["confusion_matrix"]
)


# ============================================================
# SAVE RESULTS
# ============================================================

print("\n[9/9] Saving results...")


history_file = os.path.join(
    MODEL_DIR,
    "stgnn_training_history.csv"
)

pd.DataFrame(history).to_csv(
    history_file,
    index=False
)


results = pd.DataFrame([
    {
        "model": "ST-GNN",
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "f1": test_metrics["f1"],
        "roc_auc": test_metrics["roc_auc"],
        "pr_auc": test_metrics["pr_auc"]
    }
])


results_file = os.path.join(
    MODEL_DIR,
    "stgnn_results.csv"
)

results.to_csv(
    results_file,
    index=False
)


info = {
    "model": "GraphSAGE-GRU ST-GNN",
    "num_nodes": NUM_NODES,
    "num_features": NUM_FEATURES,
    "sequence_length": SEQUENCE_LENGTH,
    "forecast_horizon": 6,
    "hidden_size": HIDDEN_SIZE,
    "num_edges": int(edge_index.shape[1]),
    "best_validation_pr_auc": float(best_pr_auc),
    "test_pr_auc": float(test_metrics["pr_auc"]),
    "test_roc_auc": float(test_metrics["roc_auc"]),
    "test_precision": float(test_metrics["precision"]),
    "test_recall": float(test_metrics["recall"]),
    "test_f1": float(test_metrics["f1"])
}


info_file = os.path.join(
    MODEL_DIR,
    "stgnn_model_info.json"
)

with open(
    info_file,
    "w"
) as f:

    json.dump(
        info,
        f,
        indent=2
    )


print("\n" + "=" * 70)
print("PHASE 5C COMPLETE")
print("=" * 70)

print("\nBest validation PR-AUC:",
      round(best_pr_auc, 6))

print(
    "Test PR-AUC:",
    round(test_metrics["pr_auc"], 6)
)

print(
    "Test Recall:",
    round(test_metrics["recall"], 6)
)

print(
    "Test F1:",
    round(test_metrics["f1"], 6)
)

print("\nSaved:")

print(best_model_file)
print(history_file)
print(results_file)
print(info_file)

print("\n✓ Spatiotemporal GNN training complete.")
print("=" * 70)
