# ============================================================
# DISASTERFLOW AI — PHASE 4C
# THRESHOLD OPTIMIZATION & MODEL SELECTION
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
    roc_auc_score
)


print("=" * 70)
print("DISASTERFLOW AI — PHASE 4C")
print("Threshold Optimization & Model Selection")
print("=" * 70)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "ml_train.csv"
)

VALIDATION_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "ml_validation.csv"
)

TEST_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "ml_test.csv"
)

MODEL_DIR = BASE_DIR / "models"

OUTPUT_FILE = (
    MODEL_DIR
    / "threshold_optimization_results.csv"
)

SELECTED_MODEL_FILE = (
    MODEL_DIR
    / "selected_model_info.csv"
)


# ============================================================
# CHECK FILES
# ============================================================

print("\n[1/8] Checking files...")

required_files = [
    TRAIN_FILE,
    VALIDATION_FILE,
    TEST_FILE,
    MODEL_DIR / "logistic_flood_model.joblib",
    MODEL_DIR / "logistic_scaler.joblib",
    MODEL_DIR / "random_forest_flood_model.joblib",
    MODEL_DIR / "xgboost_flood_model.joblib",
]

for file in required_files:

    if not file.exists():

        raise FileNotFoundError(
            f"\nRequired file not found:\n{file}"
        )

    print("✓", file.name)


# ============================================================
# LOAD DATA
# ============================================================

print("\n[2/8] Loading validation and test datasets...")

train_df = pd.read_csv(TRAIN_FILE)
val_df = pd.read_csv(VALIDATION_FILE)
test_df = pd.read_csv(TEST_FILE)

print("Train rows:", len(train_df))
print("Validation rows:", len(val_df))
print("Test rows:", len(test_df))


# ============================================================
# TARGET
# ============================================================

TARGET = "future_flood_event_6h"

if TARGET not in val_df.columns:

    raise ValueError(
        f"Target column not found: {TARGET}"
    )


# ============================================================
# IDENTIFY FEATURES
# ============================================================

print("\n[3/8] Preparing feature matrices...")


EXCLUDE_COLUMNS = [
    TARGET,
    "grid_id",
    "time",
    "future_flood_score_6h",
    "future_flood_event_12h",
    "future_flood_event_24h",
    "future_flood_score_12h",
    "future_flood_score_24h",
]


feature_columns = [
    col
    for col in train_df.columns
    if col not in EXCLUDE_COLUMNS
]


print("Feature count:", len(feature_columns))

missing_features = [
    col
    for col in feature_columns
    if col not in val_df.columns
]

if missing_features:

    raise ValueError(
        f"Missing validation features: {missing_features}"
    )


X_val = val_df[feature_columns]
y_val = val_df[TARGET].astype(int)

X_test = test_df[feature_columns]
y_test = test_df[TARGET].astype(int)


print("\nValidation target distribution:")
print(y_val.value_counts())

print("\nTest target distribution:")
print(y_test.value_counts())


# ============================================================
# LOAD MODELS
# ============================================================

print("\n[4/8] Loading trained models...")

logistic_model = joblib.load(
    MODEL_DIR / "logistic_flood_model.joblib"
)

scaler = joblib.load(
    MODEL_DIR / "logistic_scaler.joblib"
)

random_forest_model = joblib.load(
    MODEL_DIR / "random_forest_flood_model.joblib"
)

xgboost_model = joblib.load(
    MODEL_DIR / "xgboost_flood_model.joblib"
)

print("✓ Logistic Regression loaded")
print("✓ Random Forest loaded")
print("✓ XGBoost loaded")


# ============================================================
# PREPARE LOGISTIC DATA
# ============================================================

print("\nPreparing standardized Logistic Regression data...")

X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# GENERATE VALIDATION PROBABILITIES
# ============================================================

print("\n[5/8] Generating validation probabilities...")


val_probabilities = {}

print("Generating Logistic Regression probabilities...")

val_probabilities["Logistic Regression"] = (
    logistic_model.predict_proba(X_val_scaled)[:, 1]
)


print("Generating Random Forest probabilities...")

val_probabilities["Random Forest"] = (
    random_forest_model.predict_proba(X_val)[:, 1]
)


print("Generating XGBoost probabilities...")

val_probabilities["XGBoost"] = (
    xgboost_model.predict_proba(X_val)[:, 1]
)


# ============================================================
# VALIDATION BASELINE METRICS
# ============================================================

print("\nValidation PR-AUC:")

for model_name, probabilities in val_probabilities.items():

    pr_auc = average_precision_score(
        y_val,
        probabilities
    )

    roc_auc = roc_auc_score(
        y_val,
        probabilities
    )

    print(
        f"{model_name:<22} "
        f"PR-AUC={pr_auc:.6f} "
        f"ROC-AUC={roc_auc:.6f}"
    )


# ============================================================
# THRESHOLD GRID
# ============================================================

print("\n[6/8] Searching thresholds...")


thresholds = np.arange(
    0.001,
    0.501,
    0.001
)


results = []


for model_name, probabilities in val_probabilities.items():

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_val,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_val,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_val,
            predictions,
            zero_division=0
        )

        tn, fp, fn, tp = confusion_matrix(
            y_val,
            predictions,
            labels=[0, 1]
        ).ravel()

        total_alerts = int(
            predictions.sum()
        )

        alert_rate = (
            total_alerts / len(predictions)
        )

        results.append({

            "model": model_name,

            "threshold": threshold,

            "precision": precision,

            "recall": recall,

            "f1": f1,

            "true_negatives": tn,

            "false_positives": fp,

            "false_negatives": fn,

            "true_positives": tp,

            "total_alerts": total_alerts,

            "alert_rate": alert_rate,

            "pr_auc": average_precision_score(
                y_val,
                probabilities
            ),

            "roc_auc": roc_auc_score(
                y_val,
                probabilities
            )
        })


results_df = pd.DataFrame(results)


# ============================================================
# BEST THRESHOLD BY F1
# ============================================================

print("\nBest thresholds based on F1:")

best_by_f1 = (
    results_df
    .sort_values(
        ["model", "f1"],
        ascending=[True, False]
    )
    .groupby("model")
    .head(1)
)


for _, row in best_by_f1.iterrows():

    print(
        f"\n{row['model']}"
    )

    print(
        f"Threshold : {row['threshold']:.3f}"
    )

    print(
        f"Precision : {row['precision']:.6f}"
    )

    print(
        f"Recall    : {row['recall']:.6f}"
    )

    print(
        f"F1        : {row['f1']:.6f}"
    )

    print(
        f"Alerts    : {int(row['total_alerts'])}"
    )


# ============================================================
# HIGH-RECALL OPERATIONAL THRESHOLDS
# ============================================================

print("\n" + "-" * 70)
print("Operational Threshold Analysis")
print("-" * 70)


operational_targets = [
    0.70,
    0.80,
    0.90,
    0.95
]


operational_rows = []


for model_name in val_probabilities.keys():

    model_results = results_df[
        results_df["model"] == model_name
    ].copy()

    for target_recall in operational_targets:

        candidates = model_results[
            model_results["recall"] >= target_recall
        ]

        if len(candidates) == 0:

            continue

        best = candidates.sort_values(
            [
                "precision",
                "f1"
            ],
            ascending=False
        ).iloc[0]

        operational_rows.append({

            "model": model_name,

            "target_recall": target_recall,

            "threshold": best["threshold"],

            "precision": best["precision"],

            "recall": best["recall"],

            "f1": best["f1"],

            "alerts": best["total_alerts"],

            "false_positives": best["false_positives"],

            "false_negatives": best["false_negatives"]
        })


operational_df = pd.DataFrame(
    operational_rows
)


print(
    operational_df.to_string(
        index=False
    )
)


# ============================================================
# MODEL SELECTION
# ============================================================

print("\n[7/8] Selecting operational model...")


# ------------------------------------------------------------
# PRIMARY CRITERION:
# PR-AUC
#
# SECONDARY CRITERION:
# F1
# ------------------------------------------------------------

model_summary = []


for model_name in val_probabilities.keys():

    model_results = results_df[
        results_df["model"] == model_name
    ]

    best_row = (
        model_results
        .sort_values(
            "f1",
            ascending=False
        )
        .iloc[0]
    )

    model_summary.append({

        "model": model_name,

        "pr_auc": best_row["pr_auc"],

        "roc_auc": best_row["roc_auc"],

        "best_threshold": best_row["threshold"],

        "best_precision": best_row["precision"],

        "best_recall": best_row["recall"],

        "best_f1": best_row["f1"],

        "alerts_at_best_threshold":
            best_row["total_alerts"]
    })


model_summary_df = pd.DataFrame(
    model_summary
)


# ------------------------------------------------------------
# Select using PR-AUC first
# ------------------------------------------------------------

selected_model_row = (
    model_summary_df
    .sort_values(
        [
            "pr_auc",
            "best_f1"
        ],
        ascending=False
    )
    .iloc[0]
)


selected_model = (
    selected_model_row["model"]
)

selected_threshold = (
    selected_model_row["best_threshold"]
)


print("\nSelected model:")

print(
    "Model:",
    selected_model
)

print(
    "Threshold:",
    f"{selected_threshold:.3f}"
)

print(
    "Validation PR-AUC:",
    f"{selected_model_row['pr_auc']:.6f}"
)

print(
    "Validation Recall:",
    f"{selected_model_row['best_recall']:.6f}"
)

print(
    "Validation Precision:",
    f"{selected_model_row['best_precision']:.6f}"
)

print(
    "Validation F1:",
    f"{selected_model_row['best_f1']:.6f}"
)


# ============================================================
# TEST EVALUATION USING VALIDATION THRESHOLD
# ============================================================

print("\nEvaluating selected model on untouched test data...")


if selected_model == "Logistic Regression":

    test_probabilities = (
        logistic_model
        .predict_proba(X_test_scaled)[:, 1]
    )

elif selected_model == "Random Forest":

    test_probabilities = (
        random_forest_model
        .predict_proba(X_test)[:, 1]
    )

else:

    test_probabilities = (
        xgboost_model
        .predict_proba(X_test)[:, 1]
    )


test_predictions = (
    test_probabilities >= selected_threshold
).astype(int)


test_precision = precision_score(
    y_test,
    test_predictions,
    zero_division=0
)

test_recall = recall_score(
    y_test,
    test_predictions,
    zero_division=0
)

test_f1 = f1_score(
    y_test,
    test_predictions,
    zero_division=0
)

test_pr_auc = average_precision_score(
    y_test,
    test_probabilities
)

test_roc_auc = roc_auc_score(
    y_test,
    test_probabilities
)

tn, fp, fn, tp = confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1]
).ravel()


# ============================================================
# TEST RESULTS
# ============================================================

print("\n" + "-" * 70)
print("SELECTED MODEL — TEST PERFORMANCE")
print("-" * 70)

print(
    "Model     :",
    selected_model
)

print(
    "Threshold :",
    f"{selected_threshold:.3f}"
)

print(
    "Precision :",
    f"{test_precision:.6f}"
)

print(
    "Recall    :",
    f"{test_recall:.6f}"
)

print(
    "F1        :",
    f"{test_f1:.6f}"
)

print(
    "ROC-AUC   :",
    f"{test_roc_auc:.6f}"
)

print(
    "PR-AUC    :",
    f"{test_pr_auc:.6f}"
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        test_predictions
    )
)

print("\nOperational statistics:")

print(
    "Flood events:",
    int(y_test.sum())
)

print(
    "Alerts generated:",
    int(test_predictions.sum())
)

print(
    "True positives:",
    int(tp)
)

print(
    "False positives:",
    int(fp)
)

print(
    "False negatives:",
    int(fn)
)


# ============================================================
# SAVE THRESHOLD RESULTS
# ============================================================

print("\n[8/8] Saving results...")


results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


selected_info = pd.DataFrame([{

    "selected_model": selected_model,

    "selected_threshold":
        selected_threshold,

    "validation_pr_auc":
        selected_model_row["pr_auc"],

    "validation_roc_auc":
        selected_model_row["roc_auc"],

    "validation_precision":
        selected_model_row["best_precision"],

    "validation_recall":
        selected_model_row["best_recall"],

    "validation_f1":
        selected_model_row["best_f1"],

    "test_pr_auc":
        test_pr_auc,

    "test_roc_auc":
        test_roc_auc,

    "test_precision":
        test_precision,

    "test_recall":
        test_recall,

    "test_f1":
        test_f1,

    "test_true_positives":
        tp,

    "test_false_positives":
        fp,

    "test_false_negatives":
        fn,

    "test_true_negatives":
        tn,

    "test_alerts":
        int(test_predictions.sum())
}])


selected_info.to_csv(
    SELECTED_MODEL_FILE,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PHASE 4C COMPLETE")
print("=" * 70)

print("\nThreshold optimization completed.")

print(
    "\nSelected model:",
    selected_model
)

print(
    "Selected threshold:",
    f"{selected_threshold:.3f}"
)

print(
    "\nTest PR-AUC:",
    f"{test_pr_auc:.6f}"
)

print(
    "Test Recall:",
    f"{test_recall:.6f}"
)

print(
    "Test Precision:",
    f"{test_precision:.6f}"
)

print(
    "Test F1:",
    f"{test_f1:.6f}"
)

print("\nResults saved:")

print(
    "✓",
    OUTPUT_FILE
)

print(
    "✓",
    SELECTED_MODEL_FILE
)

print("\n✓ Phase 4C finished successfully.")

print("=" * 70)