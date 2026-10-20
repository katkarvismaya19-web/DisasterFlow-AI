import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ============================================================
# DISASTERFLOW AI
# PHASE 6B — NODE-LEVEL FORECAST GENERATION
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

with open(META_FILE) as f:
    metadata = json.load(f)

N = metadata["num_nodes"]
F = metadata["num_features"]
T = metadata["sequence_length"]
S = metadata["num_sequences"]

X = np.memmap(
    X_FILE,
    dtype=np.float32,
    mode="r",
    shape=tuple(metadata["X_shape"])
)

times = np.load(
    TIMES_FILE,
    allow_pickle=True
)

splits = np.load(
    SPLIT_FILE,
    allow_pickle=True
)

edges = np.load(
    EDGE_FILE
)

adjacency = torch.zeros(
    N,
    N
)

for a, b in edges.T:

    adjacency[a, b] = 1
    adjacency[b, a] = 1

adjacency += torch.eye(N)

degree = adjacency.sum(
    dim=1
)

degree_inv = degree.pow(
    -0.5
)

degree_inv[
    torch.isinf(degree_inv)
] = 0

adjacency = (
    degree_inv[:, None]
    *
    adjacency
    *
    degree_inv[None, :]
)

# ============================================================
# MODEL
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
        adj
    ):

        x = torch.einsum(
            "ij,bjf->bif",
            adj,
            x
        )

        return self.linear(x)


class STGNN(nn.Module):

    def __init__(
        self,
        input_features,
        hidden,
        graph_hidden,
        nodes
    ):

        super().__init__()

        self.nodes = nodes

        self.temporal = nn.GRU(
            input_features,
            hidden,
            batch_first=True
        )

        self.graph1 = GraphConv(
            hidden,
            graph_hidden
        )

        self.graph2 = GraphConv(
            graph_hidden,
            graph_hidden
        )

        self.attention = nn.Sequential(
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
        adj
    ):

        B = x.size(0)

        x = x.permute(
            0,
            2,
            1,
            3
        )

        x = x.reshape(
            B * self.nodes,
            x.size(2),
            x.size(3)
        )

        output, _ = self.temporal(x)

        x = output[:, -1, :]

        x = x.reshape(
            B,
            self.nodes,
            -1
        )

        x = torch.relu(
            self.graph1(
                x,
                adj
            )
        )

        x = torch.relu(
            self.graph2(
                x,
                adj
            )
        )

        att = torch.softmax(
            self.attention(x)
            .squeeze(-1),
            dim=1
        )

        x = x * (
            1 +
            att.unsqueeze(-1)
        )

        return self.classifier(
            x
        ).squeeze(-1)


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

model = STGNN(
    F,
    64,
    64,
    N
).to(device)

checkpoint = torch.load(
    MODEL_FILE,
    map_location=device,
    weights_only=False
)

# The trained ST-GNN was saved directly as a PyTorch state_dict.
model.load_state_dict(checkpoint)

model.eval()

adjacency = adjacency.to(device)

# ============================================================
# PREDICT
# ============================================================

batch_size = 8
records = []

print("=" * 70)
print("GENERATING NODE-LEVEL 6-HOUR FORECASTS")
print("=" * 70)

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

        batch = torch.from_numpy(
            np.asarray(
                X[start:end]
            )
        ).to(device)

        logits = model(
            batch,
            adjacency
        )

        probabilities = torch.sigmoid(
            logits
        ).cpu().numpy()

        for i in range(
            end - start
        ):

            sequence_index = (
                start + i
            )

            for node in range(N):

                records.append({
                    "sequence_index":
                        sequence_index,

                    "forecast_time":
                        str(
                            times[
                                sequence_index
                            ]
                        ),

                    "split":
                        str(
                            splits[
                                sequence_index
                            ]
                        ),

                    "node_index":
                        node,

                    "forecast_flood_probability":
                        float(
                            probabilities[
                                i,
                                node
                            ]
                        )
                })

        if start % 500 == 0:

            print(
                f"{end}/{S} sequences processed"
            )

predictions = pd.DataFrame(
    records
)

output = os.path.join(
    OUTPUT_DIR,
    "stgnn_node_predictions.csv"
)

predictions.to_csv(
    output,
    index=False
)

print("\nSaved:")
print(output)

print("\nPrediction statistics:")

print(
    predictions[
        "forecast_flood_probability"
    ].describe()
)

print("\n✓ Node-level forecasts generated.")