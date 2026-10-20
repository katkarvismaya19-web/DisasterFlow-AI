import os
import json
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
# DISASTERFLOW AI
# PHASE 6 — SPATIOTEMPORAL GRAPH NEURAL NETWORK
# ============================================================

print("=" * 75)
print("DISASTERFLOW AI")
print("PHASE 6 — SPATIOTEMPORAL GRAPH NEURAL NETWORK")
print("=" * 75)

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

META_FILE = os.path.join(
    GNN_DIR,
    "gnn_metadata.json"
)

SPLIT_FILE = os.path.join(
    GNN_DIR,
    "sequence_splits.npy"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "predictions"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# CONFIG
# ============================================================

BATCH_SIZE = 8
EPOCHS = 12
LEARNING_RATE = 0.001

HIDDEN_SIZE = 64
GRAPH_HIDDEN = 64

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("\nDevice:", DEVICE)

# ============================================================
# LOAD METADATA
# ============================================================

with open(META_FILE, "r") as f:
    metadata = json.load(f)

NUM_NODES = metadata["num_nodes"]
NUM_FEATURES = metadata["num_features"]
SEQ_LEN = metadata["sequence_length"]
NUM_SEQUENCES = metadata["num_sequences"]

print("\nDataset configuration:")
print("Nodes:", NUM_NODES)
print("Features:", NUM_FEATURES)
print("Sequence length:", SEQ_LEN)
print("Sequences:", NUM_SEQUENCES)

# ============================================================
# MEMMAP DATASET
# ============================================================

X_shape = tuple(metadata["X_shape"])
Y_shape = tuple(metadata["Y_shape"])

X = np.memmap(
    X_FILE,
    dtype=np.float32,
    mode="r",
    shape=X_shape
)

Y = np.memmap(
    Y_FILE,
    dtype=np.int8,
    mode="r",
    shape=Y_shape
)

sequence_splits = np.load(
    SPLIT_FILE,
    allow_pickle=True
)

# ============================================================
# DATASET CLASS
# ============================================================

class GNNSequenceDataset(Dataset):

    def __init__(self, indices):

        self.indices = np.asarray(
            indices,
            dtype=np.int64
        )

    def __len__(self):

        return len(self.indices)

    def __getitem__(self, idx):

        real_idx = self.indices[idx]

        x = np.asarray(
            X[real_idx],
            dtype=np.float32
        )

        y = np.asarray(
            Y[real_idx],
            dtype=np.float32
        )

        return (
            torch.from_numpy(x),
            torch.from_numpy(y)
        )


# ============================================================
# SPLITS
# ============================================================

train_indices = np.where(
    sequence_splits == "train"
)[0]

val_indices = np.where(
    sequence_splits == "validation"
)[0]

test_indices = np.where(
    sequence_splits == "test"
)[0]

print("\nSequence split:")
print("Train:", len(train_indices))
print("Validation:", len(val_indices))
print("Test:", len(test_indices))

# ============================================================
# CLASS DISTRIBUTION
# ============================================================

train_positive = int(
    np.asarray(
        Y[train_indices]
    ).sum()
)

train_total = (
    len(train_indices) *
    NUM_NODES
)

train_negative = (
    train_total -
    train_positive
)

print("\nTraining target:")
print("Positive:", train_positive)
print("Negative:", train_negative)

POS_WEIGHT = (
    train_negative /
    max(train_positive, 1)
)

# Prevent extreme instability
POS_WEIGHT = min(
    POS_WEIGHT,
    500.0
)

print(
    "Positive class weight:",
    round(POS_WEIGHT, 2)
)

# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    GNNSequenceDataset(train_indices),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    GNNSequenceDataset(val_indices),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    GNNSequenceDataset(test_indices),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# ============================================================
# GRAPH
# ============================================================

edge_index = np.load(
    EDGE_FILE
)

print("\nGraph:")
print("Edges:", edge_index.shape[1])

adjacency = torch.zeros(
    (NUM_NODES, NUM_NODES),
    dtype=torch.float32
)

for source, target in edge_index.T:

    if (
        0 <= source < NUM_NODES
        and
        0 <= target < NUM_NODES
    ):

        adjacency[source, target] = 1.0
        adjacency[target, source] = 1.0

# Self loops
adjacency += torch.eye(NUM_NODES)

degree = adjacency.sum(
    dim=1
)

degree_inv_sqrt = torch.pow(
    degree,
    -0.5
)

degree_inv_sqrt[
    torch.isinf(degree_inv_sqrt)
] = 0

normalized_adjacency = (
    degree_inv_sqrt[:, None]
    *
    adjacency
    *
    degree_inv_sqrt[None, :]
)

normalized_adjacency = (
    normalized_adjacency
    .to(DEVICE)
)

# ============================================================
# GRAPH CONVOLUTION
# ============================================================

class GraphConv(nn.Module):

    def __init__(
        self,
        in_features,
        out_features
    ):

        super().__init__()

        self.linear = nn.Linear(
            in_features,
            out_features
        )

    def forward(
        self,
        x,
        adjacency
    ):

        # x:
        # B × N × F

        x = torch.einsum(
            "ij,bjf->bif",
            adjacency,
            x
        )

        return self.linear(x)


# ============================================================
# ST-GNN
# ============================================================

class STGNN(nn.Module):

    def __init__(
        self,
        input_features,
        hidden_size,
        graph_hidden,
        nodes
    ):

        super().__init__()

        self.nodes = nodes

        # Temporal encoder
        self.temporal = nn.GRU(
            input_size=input_features,
            hidden_size=hidden_size,
            batch_first=True
        )

        # Spatial graph layers
        self.graph1 = GraphConv(
            hidden_size,
            graph_hidden
        )

        self.graph2 = GraphConv(
            graph_hidden,
            graph_hidden
        )

        # Attention over node representation
        self.node_attention = nn.Sequential(
            nn.Linear(
                graph_hidden,
                32
            ),
            nn.ReLU(),
            nn.Linear(
                32,
                1
            )
        )

        # Final node-level predictor
        self.classifier = nn.Sequential(
            nn.Linear(
                graph_hidden,
                32
            ),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(
                32,
                1
            )
        )

    def forward(
        self,
        x,
        adjacency
    ):

        # ----------------------------------------------------
        # x = B × T × N × F
        # ----------------------------------------------------

        batch_size = x.size(0)

        # Rearrange:
        # B × N × T × F

        x = x.permute(
            0,
            2,
            1,
            3
        )

        # B*N × T × F

        x = x.reshape(
            batch_size * self.nodes,
            x.size(2),
            x.size(3)
        )

        # Temporal representation

        temporal_output, _ = self.temporal(x)

        temporal_last = (
            temporal_output[:, -1, :]
        )

        # B × N × H

        node_features = temporal_last.reshape(
            batch_size,
            self.nodes,
            -1
        )

        # Spatial graph propagation

        node_features = self.graph1(
            node_features,
            adjacency
        )

        node_features = torch.relu(
            node_features
        )

        node_features = self.graph2(
            node_features,
            adjacency
        )

        node_features = torch.relu(
            node_features
        )

        # Attention weighting

        attention = torch.softmax(
            self.node_attention(
                node_features
            ).squeeze(-1),
            dim=1
        )

        node_features = (
            node_features
            *
            (1.0 + attention.unsqueeze(-1))
        )

        # Node-level flood prediction

        logits = self.classifier(
            node_features
        ).squeeze(-1)

        return logits


# ============================================================
# MODEL
# ============================================================

model = STGNN(
    input_features=NUM_FEATURES,
    hidden_size=HIDDEN_SIZE,
    graph_hidden=GRAPH_HIDDEN,
    nodes=NUM_NODES
).to(DEVICE)

print("\nModel parameters:")

print(
    sum(
        p.numel()
        for p in model.parameters()
    )
)

# ============================================================
# LOSS
# ============================================================

criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor(
        POS_WEIGHT,
        dtype=torch.float32,
        device=DEVICE
    )
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)

# ============================================================
# EVALUATION
# ============================================================

def evaluate(loader):

    model.eval()

    all_probs = []
    all_targets = []

    total_loss = 0
    batches = 0

    with torch.no_grad():

        for x, y in loader:

            x = x.to(
                DEVICE,
                non_blocking=True
            )

            y = y.to(
                DEVICE,
                non_blocking=True
            )

            logits = model(
                x,
                normalized_adjacency
            )

            loss = criterion(
                logits,
                y
            )

            probabilities = torch.sigmoid(
                logits
            )

            total_loss += (
                loss.item()
            )

            batches += 1

            all_probs.append(
                probabilities
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )

            all_targets.append(
                y.detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )

    probs = np.concatenate(
        all_probs
    )

    targets = np.concatenate(
        all_targets
    )

    predictions = (
        probs >= 0.5
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

    try:
        pr_auc = average_precision_score(
            targets,
            probs
        )
    except:
        pr_auc = 0.0

    try:
        roc_auc = roc_auc_score(
            targets,
            probs
        )
    except:
        roc_auc = 0.0

    return {
        "loss": total_loss / max(batches, 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc
    }


# ============================================================
# TRAINING
# ============================================================

best_pr_auc = -1

print("\n")
print("=" * 75)
print("TRAINING")
print("=" * 75)

history = []

for epoch in range(
    1,
    EPOCHS + 1
):

    model.train()

    running_loss = 0
    batches = 0

    for batch_idx, (
        x,
        y
    ) in enumerate(train_loader):

        x = x.to(
            DEVICE,
            non_blocking=True
        )

        y = y.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            x,
            normalized_adjacency
        )

        loss = criterion(
            logits,
            y
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

        running_loss += (
            loss.item()
        )

        batches += 1

        if batch_idx % 100 == 0:

            print(
                f"Epoch {epoch}/{EPOCHS} | "
                f"Batch {batch_idx}/{len(train_loader)} | "
                f"Loss {loss.item():.4f}"
            )

    train_loss = (
        running_loss /
        max(batches, 1)
    )

    validation_metrics = evaluate(
        val_loader
    )

    scheduler.step(
        validation_metrics["pr_auc"]
    )

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        **validation_metrics
    })

    print("\n" + "-" * 75)

    print(
        f"Epoch {epoch} COMPLETE"
    )

    print(
        f"Train Loss : {train_loss:.5f}"
    )

    print(
        f"Val PR-AUC : "
        f"{validation_metrics['pr_auc']:.5f}"
    )

    print(
        f"Val ROC-AUC: "
        f"{validation_metrics['roc_auc']:.5f}"
    )

    print(
        f"Val Recall : "
        f"{validation_metrics['recall']:.5f}"
    )

    print(
        f"Val F1     : "
        f"{validation_metrics['f1']:.5f}"
    )

    print("-" * 75)

    if (
        validation_metrics["pr_auc"]
        >
        best_pr_auc
    ):

        best_pr_auc = (
            validation_metrics["pr_auc"]
        )

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "num_nodes":
                    NUM_NODES,

                "num_features":
                    NUM_FEATURES,

                "sequence_length":
                    SEQ_LEN,

                "hidden_size":
                    HIDDEN_SIZE,

                "graph_hidden":
                    GRAPH_HIDDEN,

                "pos_weight":
                    POS_WEIGHT
            },
            os.path.join(
                MODEL_DIR,
                "stgnn_best.pt"
            )
        )

        print(
            "✓ Best ST-GNN model saved"
        )


# ============================================================
# SAVE HISTORY
# ============================================================

with open(
    os.path.join(
        MODEL_DIR,
        "training_history.json"
    ),
    "w"
) as f:

    json.dump(
        history,
        f,
        indent=2
    )

# ============================================================
# FINAL TEST
# ============================================================

print("\n")
print("=" * 75)
print("FINAL TEST EVALUATION")
print("=" * 75)

checkpoint = torch.load(
    os.path.join(
        MODEL_DIR,
        "stgnn_best.pt"
    ),
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

test_metrics = evaluate(
    test_loader
)

for key, value in test_metrics.items():

    print(
        f"{key.upper():12s}: "
        f"{value:.6f}"
    )

# ============================================================
# CONFUSION MATRIX
# ============================================================

model.eval()

all_test_probs = []
all_test_targets = []

with torch.no_grad():

    for x, y in test_loader:

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        logits = model(
            x,
            normalized_adjacency
        )

        probs = torch.sigmoid(
            logits
        )

        all_test_probs.append(
            probs.cpu()
            .numpy()
            .reshape(-1)
        )

        all_test_targets.append(
            y.cpu()
            .numpy()
            .reshape(-1)
        )

test_probs = np.concatenate(
    all_test_probs
)

test_targets = np.concatenate(
    all_test_targets
)

test_predictions = (
    test_probs >= 0.5
).astype(int)

cm = confusion_matrix(
    test_targets,
    test_predictions
)

print("\nConfusion Matrix:")
print(cm)

# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

np.save(
    os.path.join(
        OUTPUT_DIR,
        "test_predictions.npy"
    ),
    test_probs
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "test_targets.npy"
    ),
    test_targets
)

with open(
    os.path.join(
        OUTPUT_DIR,
        "test_metrics.json"
    ),
    "w"
) as f:

    json.dump(
        test_metrics,
        f,
        indent=2
    )

print("\n")
print("=" * 75)
print("PHASE 6 COMPLETE")
print("=" * 75)

print(
    "\nModel:",
    os.path.join(
        MODEL_DIR,
        "stgnn_best.pt"
    )
)

print(
    "Test predictions:",
    os.path.join(
        OUTPUT_DIR,
        "test_predictions.npy"
    )
)

print(
    "\n✓ ST-GNN training finished."
)