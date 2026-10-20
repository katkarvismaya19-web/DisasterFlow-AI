import os
import json
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

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
# PHASE 6 — IMPROVED SPATIOTEMPORAL GRAPH NEURAL NETWORK
# ============================================================

print("=" * 80)
print("DISASTERFLOW AI")
print("PHASE 6 — SPATIOTEMPORAL GRAPH NEURAL NETWORK")
print("=" * 80)


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
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

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 8

EPOCHS = 20

LEARNING_RATE = 0.0005

WEIGHT_DECAY = 1e-4

HIDDEN_SIZE = 64

GRAPH_HIDDEN = 64

DROPOUT = 0.25

GRADIENT_CLIP = 1.0

EARLY_STOPPING_PATIENCE = 5

NUM_WORKERS = 0


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print()
print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# CHECK FILES
# ============================================================

required_files = [
    X_FILE,
    Y_FILE,
    EDGE_FILE,
    META_FILE,
    SPLIT_FILE
]

for file_path in required_files:

    if not os.path.exists(file_path):

        raise FileNotFoundError(
            f"\nRequired file not found:\n{file_path}"
        )


# ============================================================
# LOAD METADATA
# ============================================================

with open(
    META_FILE,
    "r",
    encoding="utf-8"
) as f:

    metadata = json.load(f)


NUM_NODES = int(
    metadata["num_nodes"]
)

NUM_FEATURES = int(
    metadata["num_features"]
)

SEQ_LEN = int(
    metadata["sequence_length"]
)

NUM_SEQUENCES = int(
    metadata["num_sequences"]
)


X_shape = tuple(
    metadata["X_shape"]
)

Y_shape = tuple(
    metadata["Y_shape"]
)


print()
print("=" * 80)
print("DATASET CONFIGURATION")
print("=" * 80)

print("Sequences :", NUM_SEQUENCES)
print("Time steps:", SEQ_LEN)
print("Nodes     :", NUM_NODES)
print("Features  :", NUM_FEATURES)

print("X shape   :", X_shape)
print("Y shape   :", Y_shape)


# ============================================================
# LOAD MEMMAP DATA
# ============================================================

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


# ============================================================
# BASIC DATA VALIDATION
# ============================================================

print()
print("=" * 80)
print("DATA VALIDATION")
print("=" * 80)

print("X min:", float(np.nanmin(X)))
print("X max:", float(np.nanmax(X)))

print("Y min:", int(np.min(Y)))
print("Y max:", int(np.max(Y)))

if np.isnan(X).any():

    raise ValueError(
        "X contains NaN values."
    )

if np.isinf(X).any():

    raise ValueError(
        "X contains infinite values."
    )

unique_y, counts_y = np.unique(
    Y,
    return_counts=True
)

print(
    "Y distribution:",
    dict(
        zip(
            unique_y.tolist(),
            counts_y.tolist()
        )
    )
)


# ============================================================
# LOAD SPLITS
# ============================================================

sequence_splits = np.load(
    SPLIT_FILE,
    allow_pickle=True
)


train_indices = np.where(
    sequence_splits == "train"
)[0]

val_indices = np.where(
    sequence_splits == "validation"
)[0]

test_indices = np.where(
    sequence_splits == "test"
)[0]


print()
print("=" * 80)
print("SEQUENCE SPLIT")
print("=" * 80)

print("Train      :", len(train_indices))
print("Validation :", len(val_indices))
print("Test       :", len(test_indices))


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

train_positive = int(
    np.asarray(
        Y[train_indices]
    ).sum()
)

train_total = (
    len(train_indices)
    * NUM_NODES
)

train_negative = (
    train_total
    - train_positive
)

positive_rate = (
    train_positive
    / max(train_total, 1)
)

print()
print("=" * 80)
print("TRAINING TARGET DISTRIBUTION")
print("=" * 80)

print(
    "Positive:",
    train_positive
)

print(
    "Negative:",
    train_negative
)

print(
    "Positive rate:",
    f"{positive_rate * 100:.6f}%"
)


# ============================================================
# POSITIVE CLASS WEIGHT
# ============================================================

raw_pos_weight = (
    train_negative
    / max(train_positive, 1)
)

print(
    "Raw positive weight:",
    f"{raw_pos_weight:.4f}"
)


# Extremely large weights can destabilize training.
#
# We use a conservative capped value.
#
# The actual value can be adjusted after observing
# validation PR-AUC and recall.

POS_WEIGHT = min(
    raw_pos_weight,
    100.0
)

print(
    "Used positive weight:",
    f"{POS_WEIGHT:.4f}"
)


# ============================================================
# DATASET
# ============================================================

class GNNSequenceDataset(Dataset):

    def __init__(
        self,
        indices
    ):

        self.indices = np.asarray(
            indices,
            dtype=np.int64
        )

    def __len__(self):

        return len(
            self.indices
        )

    def __getitem__(
        self,
        idx
    ):

        real_idx = int(
            self.indices[idx]
        )

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
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    GNNSequenceDataset(
        train_indices
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=(
        DEVICE.type == "cuda"
    )
)

val_loader = DataLoader(
    GNNSequenceDataset(
        val_indices
    ),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(
        DEVICE.type == "cuda"
    )
)

test_loader = DataLoader(
    GNNSequenceDataset(
        test_indices
    ),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(
        DEVICE.type == "cuda"
    )
)


# ============================================================
# GRAPH
# ============================================================

edge_index = np.load(
    EDGE_FILE
)

if edge_index.ndim != 2:

    raise ValueError(
        "edge_index.npy must be 2-dimensional."
    )

if edge_index.shape[0] != 2:

    raise ValueError(
        "edge_index.npy must have shape (2, E)."
    )


print()
print("=" * 80)
print("GRAPH")
print("=" * 80)

print(
    "Edge index shape:",
    edge_index.shape
)

print(
    "Edges:",
    edge_index.shape[1]
)


# ============================================================
# BUILD ADJACENCY MATRIX
# ============================================================

adjacency = torch.zeros(
    (
        NUM_NODES,
        NUM_NODES
    ),
    dtype=torch.float32
)


valid_edges = 0

for source, target in edge_index.T:

    source = int(source)
    target = int(target)

    if (
        0 <= source < NUM_NODES
        and
        0 <= target < NUM_NODES
    ):

        adjacency[
            source,
            target
        ] = 1.0

        adjacency[
            target,
            source
        ] = 1.0

        valid_edges += 1


# ============================================================
# SELF LOOPS
# ============================================================

adjacency += torch.eye(
    NUM_NODES
)


# ============================================================
# NORMALIZED GRAPH
# ============================================================

degree = adjacency.sum(
    dim=1
)

degree_inv_sqrt = torch.pow(
    degree,
    -0.5
)

degree_inv_sqrt[
    torch.isinf(
        degree_inv_sqrt
    )
] = 0.0


normalized_adjacency = (
    degree_inv_sqrt[:, None]
    *
    adjacency
    *
    degree_inv_sqrt[None, :]
)


normalized_adjacency = normalized_adjacency.to(
    DEVICE
)


print(
    "Valid edges:",
    valid_edges
)

print(
    "Average degree:",
    float(
        degree.mean()
    )
)


# ============================================================
# GRAPH CONVOLUTION
# ============================================================

class GraphConv(nn.Module):

    def __init__(
        self,
        in_features,
        out_features,
        dropout=0.0
    ):

        super().__init__()

        self.linear = nn.Linear(
            in_features,
            out_features
        )

        self.norm = nn.LayerNorm(
            out_features
        )

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(
        self,
        x,
        adjacency
    ):

        # ----------------------------------------------------
        # x:
        # B × N × F
        # ----------------------------------------------------

        x = torch.einsum(
            "ij,bjf->bif",
            adjacency,
            x
        )

        x = self.linear(
            x
        )

        x = self.norm(
            x
        )

        x = F.relu(
            x
        )

        x = self.dropout(
            x
        )

        return x


# ============================================================
# SPATIOTEMPORAL GNN
# ============================================================

class STGNN(nn.Module):

    def __init__(
        self,
        input_features,
        hidden_size,
        graph_hidden,
        nodes,
        dropout=0.25
    ):

        super().__init__()

        self.nodes = nodes

        # ====================================================
        # TEMPORAL ENCODER
        # ====================================================

        self.temporal = nn.GRU(
            input_size=input_features,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )


        # ====================================================
        # TEMPORAL NORMALIZATION
        # ====================================================

        self.temporal_norm = nn.LayerNorm(
            hidden_size
        )


        # ====================================================
        # SPATIAL GRAPH LAYERS
        # ====================================================

        self.graph1 = GraphConv(
            hidden_size,
            graph_hidden,
            dropout
        )

        self.graph2 = GraphConv(
            graph_hidden,
            graph_hidden,
            dropout
        )


        # ====================================================
        # RESIDUAL PROJECTION
        # ====================================================

        self.residual = nn.Linear(
            hidden_size,
            graph_hidden
        )


        # ====================================================
        # NODE ATTENTION
        # ====================================================

        self.node_attention = nn.Sequential(

            nn.Linear(
                graph_hidden,
                32
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                32,
                1
            )
        )


        # ====================================================
        # FINAL CLASSIFIER
        # ====================================================

        self.classifier = nn.Sequential(

            nn.Linear(
                graph_hidden,
                64
            ),

            nn.LayerNorm(
                64
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                64,
                32
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

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

        # ====================================================
        # INPUT
        #
        # B × T × N × F
        # ====================================================

        batch_size = x.size(0)


        # ====================================================
        # REARRANGE
        #
        # B × N × T × F
        # ====================================================

        x = x.permute(
            0,
            2,
            1,
            3
        )


        # ====================================================
        # EACH NODE GETS ITS OWN TEMPORAL SEQUENCE
        #
        # B*N × T × F
        # ====================================================

        x = x.reshape(
            batch_size * self.nodes,
            x.size(2),
            x.size(3)
        )


        # ====================================================
        # TEMPORAL GRU
        # ====================================================

        temporal_output, _ = self.temporal(
            x
        )


        # Last timestep
        temporal_last = temporal_output[
            :,
            -1,
            :
        ]


        temporal_last = self.temporal_norm(
            temporal_last
        )


        # ====================================================
        # RESTORE NODE DIMENSION
        #
        # B × N × H
        # ====================================================

        node_features = temporal_last.reshape(
            batch_size,
            self.nodes,
            -1
        )


        # ====================================================
        # RESIDUAL
        # ====================================================

        residual = self.residual(
            node_features
        )


        # ====================================================
        # GRAPH CONVOLUTION 1
        # ====================================================

        node_features = self.graph1(
            node_features,
            adjacency
        )


        # ====================================================
        # GRAPH CONVOLUTION 2
        # ====================================================

        node_features = self.graph2(
            node_features,
            adjacency
        )


        # ====================================================
        # RESIDUAL CONNECTION
        # ====================================================

        node_features = (
            node_features
            +
            residual
        )


        node_features = F.relu(
            node_features
        )


        # ====================================================
        # NODE ATTENTION
        # ====================================================

        attention_logits = self.node_attention(
            node_features
        ).squeeze(-1)


        attention = torch.softmax(
            attention_logits,
            dim=1
        )


        # ====================================================
        # ATTENTION ENHANCEMENT
        # ====================================================

        node_features = (
            node_features
            *
            (
                1.0
                +
                attention.unsqueeze(-1)
            )
        )


        # ====================================================
        # NODE-LEVEL FLOOD PREDICTION
        # ====================================================

        logits = self.classifier(
            node_features
        ).squeeze(-1)


        return logits


# ============================================================
# CREATE MODEL
# ============================================================

model = STGNN(
    input_features=NUM_FEATURES,
    hidden_size=HIDDEN_SIZE,
    graph_hidden=GRAPH_HIDDEN,
    nodes=NUM_NODES,
    dropout=DROPOUT
).to(DEVICE)


# ============================================================
# MODEL INFORMATION
# ============================================================

total_parameters = sum(
    p.numel()
    for p in model.parameters()
)

trainable_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)


print()
print("=" * 80)
print("MODEL")
print("=" * 80)

print(
    "Total parameters:",
    f"{total_parameters:,}"
)

print(
    "Trainable parameters:",
    f"{trainable_parameters:,}"
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


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2,
    min_lr=1e-6
)


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(
    targets,
    probs,
    threshold=0.5
):

    predictions = (
        probs >= threshold
    ).astype(
        np.int8
    )


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

    except Exception:

        pr_auc = 0.0


    try:

        roc_auc = roc_auc_score(
            targets,
            probs
        )

    except Exception:

        roc_auc = 0.0


    return {
        "precision": float(
            precision
        ),

        "recall": float(
            recall
        ),

        "f1": float(
            f1
        ),

        "pr_auc": float(
            pr_auc
        ),

        "roc_auc": float(
            roc_auc
        )
    }


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    loader
):

    model.eval()

    all_probs = []
    all_targets = []

    total_loss = 0.0
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


            total_loss += loss.item()

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


    metrics = calculate_metrics(
        targets,
        probs,
        threshold=0.5
    )


    metrics["loss"] = (
        total_loss
        /
        max(
            batches,
            1
        )
    )


    return (
        metrics,
        probs,
        targets
    )


# ============================================================
# THRESHOLD OPTIMIZATION
# ============================================================

def find_best_threshold(
    targets,
    probs
):

    best_threshold = 0.5

    best_f1 = -1.0

    best_precision = 0.0

    best_recall = 0.0


    thresholds = np.linspace(
        0.01,
        0.99,
        99
    )


    for threshold in thresholds:

        predictions = (
            probs >= threshold
        ).astype(
            np.int8
        )


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


        # Primary objective:
        # maximize F1 while retaining
        # meaningful recall.

        if f1 > best_f1:

            best_f1 = f1

            best_threshold = float(
                threshold
            )

            best_precision = float(
                precision
            )

            best_recall = float(
                recall
            )


    return {
        "threshold": best_threshold,
        "f1": best_f1,
        "precision": best_precision,
        "recall": best_recall
    }


# ============================================================
# TRAINING
# ============================================================

print()
print("=" * 80)
print("TRAINING")
print("=" * 80)


best_pr_auc = -1.0

best_epoch = 0

epochs_without_improvement = 0

history = []


for epoch in range(
    1,
    EPOCHS + 1
):

    model.train()


    running_loss = 0.0

    batches = 0


    for batch_idx, (
        x,
        y
    ) in enumerate(
        train_loader
    ):

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
            GRADIENT_CLIP
        )


        optimizer.step()


        running_loss += loss.item()

        batches += 1


        if batch_idx % 100 == 0:

            print(
                f"Epoch {epoch}/{EPOCHS} | "
                f"Batch {batch_idx}/{len(train_loader)} | "
                f"Loss {loss.item():.6f}"
            )


    train_loss = (
        running_loss
        /
        max(
            batches,
            1
        )
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    val_metrics, val_probs, val_targets = evaluate(
        val_loader
    )


    scheduler.step(
        val_metrics["pr_auc"]
    )


    current_lr = optimizer.param_groups[
        0
    ]["lr"]


    # ========================================================
    # HISTORY
    # ========================================================

    epoch_record = {

        "epoch": epoch,

        "train_loss": float(
            train_loss
        ),

        "val_loss": float(
            val_metrics["loss"]
        ),

        "val_precision": float(
            val_metrics["precision"]
        ),

        "val_recall": float(
            val_metrics["recall"]
        ),

        "val_f1": float(
            val_metrics["f1"]
        ),

        "val_pr_auc": float(
            val_metrics["pr_auc"]
        ),

        "val_roc_auc": float(
            val_metrics["roc_auc"]
        ),

        "learning_rate": float(
            current_lr
        )
    }


    history.append(
        epoch_record
    )


    # ========================================================
    # PRINT EPOCH
    # ========================================================

    print()
    print("-" * 80)

    print(
        f"EPOCH {epoch} COMPLETE"
    )

    print("-" * 80)

    print(
        f"Train Loss   : "
        f"{train_loss:.6f}"
    )

    print(
        f"Val Loss     : "
        f"{val_metrics['loss']:.6f}"
    )

    print(
        f"Val PR-AUC   : "
        f"{val_metrics['pr_auc']:.6f}"
    )

    print(
        f"Val ROC-AUC  : "
        f"{val_metrics['roc_auc']:.6f}"
    )

    print(
        f"Val Precision: "
        f"{val_metrics['precision']:.6f}"
    )

    print(
        f"Val Recall   : "
        f"{val_metrics['recall']:.6f}"
    )

    print(
        f"Val F1       : "
        f"{val_metrics['f1']:.6f}"
    )

    print(
        f"Learning rate: "
        f"{current_lr:.8f}"
    )

    print("-" * 80)


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if (
        val_metrics["pr_auc"]
        >
        best_pr_auc
    ):

        best_pr_auc = (
            val_metrics["pr_auc"]
        )

        best_epoch = epoch

        epochs_without_improvement = 0


        checkpoint = {

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

            "dropout":
                DROPOUT,

            "pos_weight":
                POS_WEIGHT,

            "best_epoch":
                best_epoch,

            "best_val_pr_auc":
                best_pr_auc
        }


        torch.save(
            checkpoint,
            os.path.join(
                MODEL_DIR,
                "stgnn_best.pt"
            )
        )


        print(
            "BEST MODEL SAVED"
        )


    else:

        epochs_without_improvement += 1


    # ========================================================
    # EARLY STOPPING
    # ========================================================

    if (
        epochs_without_improvement
        >=
        EARLY_STOPPING_PATIENCE
    ):

        print()
        print(
            "Early stopping triggered."
        )

        print(
            "Best epoch:",
            best_epoch
        )

        print(
            "Best validation PR-AUC:",
            f"{best_pr_auc:.6f}"
        )

        break


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_file = os.path.join(
    MODEL_DIR,
    "training_history.json"
)


with open(
    history_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        history,
        f,
        indent=2
    )


# ============================================================
# LOAD BEST MODEL
# ============================================================

print()
print("=" * 80)
print("LOADING BEST MODEL")
print("=" * 80)


best_model_path = os.path.join(
    MODEL_DIR,
    "stgnn_best.pt"
)


if not os.path.exists(
    best_model_path
):

    raise FileNotFoundError(
        "Best model checkpoint was not created."
    )


checkpoint = torch.load(
    best_model_path,
    map_location=DEVICE
)


model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)


print(
    "Best epoch:",
    checkpoint[
        "best_epoch"
    ]
)

print(
    "Best validation PR-AUC:",
    f"{checkpoint['best_val_pr_auc']:.6f}"
)


# ============================================================
# VALIDATION THRESHOLD OPTIMIZATION
# ============================================================

print()
print("=" * 80)
print("VALIDATION THRESHOLD OPTIMIZATION")
print("=" * 80)


val_metrics, val_probs, val_targets = evaluate(
    val_loader
)


threshold_result = find_best_threshold(
    val_targets,
    val_probs
)


BEST_THRESHOLD = threshold_result[
    "threshold"
]


print(
    "Optimal threshold:",
    f"{BEST_THRESHOLD:.4f}"
)

print(
    "Validation precision:",
    f"{threshold_result['precision']:.6f}"
)

print(
    "Validation recall:",
    f"{threshold_result['recall']:.6f}"
)

print(
    "Validation F1:",
    f"{threshold_result['f1']:.6f}"
)


# ============================================================
# SAVE THRESHOLD
# ============================================================

threshold_file = os.path.join(
    MODEL_DIR,
    "decision_threshold.json"
)


with open(
    threshold_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        {
            "threshold":
                BEST_THRESHOLD,

            "validation_precision":
                threshold_result["precision"],

            "validation_recall":
                threshold_result["recall"],

            "validation_f1":
                threshold_result["f1"],

            "selection_method":
                "maximum_validation_f1"
        },
        f,
        indent=2
    )


# ============================================================
# FINAL TEST
# ============================================================

print()
print("=" * 80)
print("FINAL TEST EVALUATION")
print("=" * 80)


test_metrics, test_probs, test_targets = evaluate(
    test_loader
)


# ============================================================
# DEFAULT 0.5 METRICS
# ============================================================

default_test_metrics = calculate_metrics(
    test_targets,
    test_probs,
    threshold=0.5
)


# ============================================================
# OPTIMIZED THRESHOLD METRICS
# ============================================================

optimized_test_metrics = calculate_metrics(
    test_targets,
    test_probs,
    threshold=BEST_THRESHOLD
)


print()
print("TEST RESULTS — THRESHOLD 0.50")
print("-" * 50)

print(
    f"Precision : "
    f"{default_test_metrics['precision']:.6f}"
)

print(
    f"Recall    : "
    f"{default_test_metrics['recall']:.6f}"
)

print(
    f"F1        : "
    f"{default_test_metrics['f1']:.6f}"
)

print(
    f"PR-AUC    : "
    f"{default_test_metrics['pr_auc']:.6f}"
)

print(
    f"ROC-AUC   : "
    f"{default_test_metrics['roc_auc']:.6f}"
)


print()
print(
    f"TEST RESULTS — OPTIMIZED THRESHOLD "
    f"{BEST_THRESHOLD:.4f}"
)

print("-" * 50)

print(
    f"Precision : "
    f"{optimized_test_metrics['precision']:.6f}"
)

print(
    f"Recall    : "
    f"{optimized_test_metrics['recall']:.6f}"
)

print(
    f"F1        : "
    f"{optimized_test_metrics['f1']:.6f}"
)

print(
    f"PR-AUC    : "
    f"{optimized_test_metrics['pr_auc']:.6f}"
)

print(
    f"ROC-AUC   : "
    f"{optimized_test_metrics['roc_auc']:.6f}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

test_predictions = (
    test_probs
    >=
    BEST_THRESHOLD
).astype(
    np.int8
)


cm = confusion_matrix(
    test_targets,
    test_predictions
)


print()
print("=" * 80)
print("CONFUSION MATRIX")
print("=" * 80)

print(cm)


if cm.shape == (2, 2):

    tn, fp, fn, tp = cm.ravel()

    print()
    print("True Negatives :", int(tn))
    print("False Positives:", int(fp))
    print("False Negatives:", int(fn))
    print("True Positives :", int(tp))


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

test_predictions_file = os.path.join(
    OUTPUT_DIR,
    "test_predictions.npy"
)

test_targets_file = os.path.join(
    OUTPUT_DIR,
    "test_targets.npy"
)

test_metrics_file = os.path.join(
    OUTPUT_DIR,
    "test_metrics.json"
)


np.save(
    test_predictions_file,
    test_probs
)

np.save(
    test_targets_file,
    test_targets
)


# ============================================================
# SAVE METRICS
# ============================================================

final_results = {

    "model":
        "Improved ST-GNN",

    "num_sequences":
        NUM_SEQUENCES,

    "num_nodes":
        NUM_NODES,

    "num_features":
        NUM_FEATURES,

    "sequence_length":
        SEQ_LEN,

    "best_epoch":
        int(best_epoch),

    "best_validation_pr_auc":
        float(best_pr_auc),

    "decision_threshold":
        float(BEST_THRESHOLD),

    "threshold_selection":
        "maximum_validation_f1",

    "test_default_threshold":
        {
            key: float(value)
            for key, value
            in default_test_metrics.items()
        },

    "test_optimized_threshold":
        {
            key: float(value)
            for key, value
            in optimized_test_metrics.items()
        },

    "confusion_matrix":
        cm.tolist()
}


with open(
    test_metrics_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        final_results,
        f,
        indent=2
    )


# ============================================================
# SAVE PREDICTION CSV
# ============================================================

prediction_csv = os.path.join(
    OUTPUT_DIR,
    "test_predictions.csv"
)


prediction_df = pd.DataFrame(
    {
        "target":
            test_targets.astype(
                np.int8
            ),

        "flood_probability":
            test_probs.astype(
                np.float32
            ),

        "prediction":
            test_predictions.astype(
                np.int8
            )
    }
)


prediction_df.to_csv(
    prediction_csv,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 80)
print("PHASE 6 COMPLETE")
print("=" * 80)

print()

print(
    "Best model:",
    best_model_path
)

print(
    "Training history:",
    history_file
)

print(
    "Decision threshold:",
    threshold_file
)

print(
    "Test predictions:",
    test_predictions_file
)

print(
    "Test targets:",
    test_targets_file
)

print(
    "Test metrics:",
    test_metrics_file
)

print(
    "Prediction CSV:",
    prediction_csv
)

print()
print(
    "Best validation PR-AUC:",
    f"{best_pr_auc:.6f}"
)

print(
    "Final test PR-AUC:",
    f"{optimized_test_metrics['pr_auc']:.6f}"
)

print(
    "Final test ROC-AUC:",
    f"{optimized_test_metrics['roc_auc']:.6f}"
)

print(
    "Final test F1:",
    f"{optimized_test_metrics['f1']:.6f}"
)

print(
    "Final test Recall:",
    f"{optimized_test_metrics['recall']:.6f}"
)

print()
print("=" * 80)
print("ST-GNN TRAINING FINISHED SUCCESSFULLY")
print("=" * 80)