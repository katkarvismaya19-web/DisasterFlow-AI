import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ============================================================
# DISASTERFLOW AI
# PHASE 6B — NODE-LEVEL FORECAST GENERATION
#
# IMPORTANT:
# This architecture matches:
# models/stgnn_best_model.pt
#
# Saved checkpoint keys:
#   gnn.linear_self.*
#   gnn.linear_neigh.*
#   gru.*
#   output.*
# ============================================================


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
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# FILES
# ============================================================

META_FILE = os.path.join(
    GNN_DIR,
    "gnn_metadata.json"
)

X_FILE = os.path.join(
    GNN_DIR,
    "X_sequences.dat"
)

EDGE_FILE = os.path.join(
    GNN_DIR,
    "edge_index.npy"
)

TIMES_FILE = os.path.join(
    GNN_DIR,
    "sequence_times.npy"
)

SPLIT_FILE = os.path.join(
    GNN_DIR,
    "sequence_splits.npy"
)

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "stgnn_best_model.pt"
)


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

required_files = {
    "Metadata": META_FILE,
    "Feature sequences": X_FILE,
    "Edge index": EDGE_FILE,
    "Sequence times": TIMES_FILE,
    "Sequence splits": SPLIT_FILE,
    "STGNN model": MODEL_FILE,
}

print("=" * 70)
print("DISASTERFLOW AI — ST-GNN NODE FORECAST GENERATOR")
print("=" * 70)

for name, path in required_files.items():

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"\nMissing {name} file:\n{path}"
        )

    print(
        f"✓ {name}: {os.path.basename(path)}"
    )


# ============================================================
# LOAD METADATA
# ============================================================

with open(
    META_FILE,
    "r"
) as f:

    metadata = json.load(f)


N = int(
    metadata["num_nodes"]
)

F = int(
    metadata["num_features"]
)

T = int(
    metadata["sequence_length"]
)

S = int(
    metadata["num_sequences"]
)


print("\nModel/Data configuration:")
print(
    f"  Nodes             : {N}"
)

print(
    f"  Features          : {F}"
)

print(
    f"  Sequence length   : {T}"
)

print(
    f"  Sequences         : {S}"
)


# ============================================================
# LOAD FEATURE SEQUENCES
# ============================================================

X = np.memmap(
    X_FILE,
    dtype=np.float32,
    mode="r",
    shape=tuple(
        metadata["X_shape"]
    )
)


# ============================================================
# LOAD TIMES
# ============================================================

times = np.load(
    TIMES_FILE,
    allow_pickle=True
)


# ============================================================
# LOAD SPLITS
# ============================================================

splits = np.load(
    SPLIT_FILE,
    allow_pickle=True
)


# ============================================================
# LOAD GRAPH EDGES
# ============================================================

edges = np.load(
    EDGE_FILE
)


print(
    f"\n✓ Loaded graph edges: {edges.shape}"
)


# ============================================================
# BUILD NORMALIZED ADJACENCY MATRIX
# ============================================================

adjacency = torch.zeros(
    N,
    N,
    dtype=torch.float32
)


for a, b in edges.T:

    a = int(a)
    b = int(b)

    adjacency[a, b] = 1.0
    adjacency[b, a] = 1.0


# Add self-loops
adjacency += torch.eye(
    N,
    dtype=torch.float32
)


# ============================================================
# SYMMETRIC NORMALIZATION
# ============================================================

degree = adjacency.sum(
    dim=1
)

degree_inv = degree.pow(
    -0.5
)

degree_inv[
    torch.isinf(
        degree_inv
    )
] = 0.0


adjacency = (
    degree_inv[:, None]
    *
    adjacency
    *
    degree_inv[None, :]
)


print(
    f"✓ Adjacency matrix: {adjacency.shape}"
)


# ============================================================
# GRAPH SAGE LAYER
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


    def forward(
        self,
        x,
        adjacency
    ):
        """
        x:

            [batch, nodes, features]

        adjacency:

            [nodes, nodes]
        """

        # ----------------------------------------------------
        # Neighbor aggregation
        # ----------------------------------------------------

        neighbor = torch.matmul(
            adjacency,
            x
        )


        # ----------------------------------------------------
        # Self + neighborhood representation
        # ----------------------------------------------------

        output = (
            self.linear_self(x)
            +
            self.linear_neigh(
                neighbor
            )
        )


        return torch.relu(
            output
        )


# ============================================================
# GRAPH SAGE + GRU ST-GNN
# ============================================================

class STGNN(nn.Module):

    def __init__(
        self,
        input_features,
        hidden_size,
        output_size,
        num_nodes
    ):

        super().__init__()

        self.num_nodes = num_nodes


        # ----------------------------------------------------
        # SPATIAL GRAPH COMPONENT
        # ----------------------------------------------------

        self.gnn = GraphSAGELayer(
            input_features,
            hidden_size
        )


        # ----------------------------------------------------
        # TEMPORAL GRU
        # ----------------------------------------------------

        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True
        )


        # ----------------------------------------------------
        # OUTPUT LAYER
        # ----------------------------------------------------

        self.output = nn.Linear(
            hidden_size,
            output_size
        )


    def forward(
        self,
        x,
        adjacency
    ):
        """
        Input:

            x =
            [batch, sequence, nodes, features]

        Output:

            [batch, nodes]
        """

        batch_size = x.size(0)

        sequence_length = x.size(1)


        # ----------------------------------------------------
        # Process each timestep spatially
        # ----------------------------------------------------

        temporal_outputs = []


        for t in range(
            sequence_length
        ):

            # ----------------------------------------------
            # Current timestep
            # ----------------------------------------------

            xt = x[
                :,
                t,
                :,
                :
            ]


            # ----------------------------------------------
            # GraphSAGE
            # ----------------------------------------------

            spatial = self.gnn(
                xt,
                adjacency
            )


            temporal_outputs.append(
                spatial
            )


        # ----------------------------------------------------
        # Shape:
        #
        # [batch, sequence, nodes, hidden]
        # ----------------------------------------------------

        spatial_sequence = torch.stack(
            temporal_outputs,
            dim=1
        )


        # ----------------------------------------------------
        # Prepare GRU input
        #
        # Each node gets its own temporal sequence.
        #
        # [B,T,N,H]
        #
        # ->
        #
        # [B*N,T,H]
        # ----------------------------------------------------

        spatial_sequence = spatial_sequence.permute(
            0,
            2,
            1,
            3
        )


        spatial_sequence = spatial_sequence.reshape(
            batch_size * self.num_nodes,
            sequence_length,
            -1
        )


        # ----------------------------------------------------
        # GRU
        # ----------------------------------------------------

        temporal_output, _ = self.gru(
            spatial_sequence
        )


        # ----------------------------------------------------
        # Last timestep
        # ----------------------------------------------------

        last_hidden = temporal_output[
            :,
            -1,
            :
        ]


        # ----------------------------------------------------
        # Restore nodes
        #
        # [B*N,H]
        #
        # ->
        #
        # [B,N,H]
        # ----------------------------------------------------

        last_hidden = last_hidden.reshape(
            batch_size,
            self.num_nodes,
            -1
        )


        # ----------------------------------------------------
        # Flood prediction
        # ----------------------------------------------------

        logits = self.output(
            last_hidden
        )


        return logits.squeeze(-1)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print(
    f"\n✓ Device: {device}"
)


# ============================================================
# CREATE MODEL
# ============================================================

# The trained model metadata shows:
#
# num_features = 38
# hidden_size  = 24
# output_size  = 1
# num_nodes    = 140
#
# Therefore:
#
# 38 → GraphSAGE → 24 → GRU → 24 → Linear → 1


HIDDEN_SIZE = 24

OUTPUT_SIZE = 1


model = STGNN(
    input_features=F,
    hidden_size=HIDDEN_SIZE,
    output_size=OUTPUT_SIZE,
    num_nodes=N
).to(device)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print(
    "\nLoading trained ST-GNN..."
)


checkpoint = torch.load(
    MODEL_FILE,
    map_location=device,
    weights_only=False
)


# ============================================================
# VERIFY CHECKPOINT FORMAT
# ============================================================

print(
    f"Checkpoint type: {type(checkpoint)}"
)


if not isinstance(
    checkpoint,
    dict
):

    raise RuntimeError(
        "\nUnexpected checkpoint format."
    )


checkpoint_keys = list(
    checkpoint.keys()
)


print(
    "\nCheckpoint parameters:"
)


for key in checkpoint_keys:

    print(
        f"  ✓ {key}"
    )


# ============================================================
# EXPECTED CHECKPOINT STRUCTURE
# ============================================================

expected_prefixes = [
    "gnn.linear_self",
    "gnn.linear_neigh",
    "gru",
    "output"
]


checkpoint_text = " ".join(
    checkpoint_keys
)


for prefix in expected_prefixes:

    if prefix not in checkpoint_text:

        raise RuntimeError(
            "\nCheckpoint does not match "
            "the GraphSAGE-GRU architecture."
        )


# ============================================================
# LOAD WEIGHTS
# ============================================================

try:

    model.load_state_dict(
        checkpoint,
        strict=True
    )

except RuntimeError as error:

    print(
        "\n❌ MODEL ARCHITECTURE MISMATCH"
    )

    print(
        "\nThe checkpoint contains:"
    )

    for key in checkpoint_keys:

        print(
            f"  {key}"
        )

    print(
        "\nFull PyTorch error:"
    )

    print(error)

    raise


print(
    "\n✓ STGNN model loaded successfully."
)

print(
    "✓ Architecture: GraphSAGE-GRU"
)

print(
    "✓ Model weights matched successfully."
)


model.eval()


# ============================================================
# MOVE GRAPH TO DEVICE
# ============================================================

adjacency = adjacency.to(
    device
)


# ============================================================
# PREDICTION
# ============================================================

batch_size = 8

records = []


print("\n")
print("=" * 70)
print("GENERATING NODE-LEVEL 6-HOUR FLOOD FORECASTS")
print("=" * 70)

print(
    f"Total sequences : {S}"
)

print(
    f"Nodes/sequence  : {N}"
)

print(
    f"Forecast horizon : 6 hours"
)

print(
    f"Batch size       : {batch_size}"
)

print("=" * 70)


# ============================================================
# INFERENCE
# ============================================================

with torch.no_grad():

    for start in range(
        0,
        S,
        batch_size
    ):

        end = min(
            start + batch_size,
            S
        )


        # ----------------------------------------------------
        # Load batch
        # ----------------------------------------------------

        batch_np = np.asarray(
            X[start:end],
            dtype=np.float32
        )


        batch = torch.from_numpy(
            batch_np
        ).to(device)


        # ----------------------------------------------------
        # Model prediction
        # ----------------------------------------------------

        logits = model(
            batch,
            adjacency
        )


        # ----------------------------------------------------
        # Convert logits → probabilities
        # ----------------------------------------------------

        probabilities = torch.sigmoid(
            logits
        ).cpu().numpy()


        # ----------------------------------------------------
        # Store predictions
        # ----------------------------------------------------

        for i in range(
            end - start
        ):

            sequence_index = (
                start + i
            )


            forecast_time = str(
                times[
                    sequence_index
                ]
            )


            split_name = str(
                splits[
                    sequence_index
                ]
            )


            for node in range(
                N
            ):

                probability = float(
                    probabilities[
                        i,
                        node
                    ]
                )


                records.append({

                    "sequence_index":
                        sequence_index,

                    "forecast_time":
                        forecast_time,

                    "split":
                        split_name,

                    "node_index":
                        node,

                    "forecast_flood_probability":
                        probability

                })


        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            start == 0
            or
            start % 500 == 0
            or
            end == S
        ):

            print(
                f"✓ {end}/{S} sequences processed"
            )


# ============================================================
# CREATE DATAFRAME
# ============================================================

predictions = pd.DataFrame(
    records
)


# ============================================================
# OUTPUT FILE
# ============================================================

output = os.path.join(
    OUTPUT_DIR,
    "stgnn_node_predictions.csv"
)


predictions.to_csv(
    output,
    index=False
)


# ============================================================
# STATISTICS
# ============================================================

print("\n")
print("=" * 70)
print("PREDICTION COMPLETE")
print("=" * 70)


print(
    f"\nTotal prediction records: "
    f"{len(predictions):,}"
)


print(
    f"\nSaved to:"
)

print(
    output
)


print(
    "\nPrediction statistics:"
)


print(
    predictions[
        "forecast_flood_probability"
    ].describe()
)


# ============================================================
# ADDITIONAL SUMMARY
# ============================================================

print(
    "\nProbability range:"
)

print(
    "Minimum:",
    predictions[
        "forecast_flood_probability"
    ].min()
)

print(
    "Maximum:",
    predictions[
        "forecast_flood_probability"
    ].max()
)

print(
    "Mean:",
    predictions[
        "forecast_flood_probability"
    ].mean()
)


# ============================================================
# HIGH-RISK NODE COUNT
# ============================================================

high_risk_count = (
    predictions[
        "forecast_flood_probability"
    ]
    >= 0.5
).sum()


critical_count = (
    predictions[
        "forecast_flood_probability"
    ]
    >= 0.8
).sum()


print(
    "\nForecast risk indicators:"
)

print(
    f"  Nodes with probability >= 0.50: "
    f"{high_risk_count:,}"
)

print(
    f"  Nodes with probability >= 0.80: "
    f"{critical_count:,}"
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n")
print("=" * 70)
print("✓ NODE-LEVEL FORECASTS GENERATED SUCCESSFULLY")
print("=" * 70)
print(
    "✓ Spatial modelling   : GraphSAGE"
)
print(
    "✓ Temporal modelling  : GRU"
)
print(
    "✓ Forecast horizon    : 6 hours"
)
print(
    f"✓ Spatial nodes       : {N}"
)
print(
    f"✓ Temporal sequence   : {T} hours"
)
print(
    f"✓ Input features      : {F}"
)
print(
    f"✓ Output records      : {len(predictions):,}"
)
print(
    f"✓ Output file         : {output}"
)
print("=" * 70)