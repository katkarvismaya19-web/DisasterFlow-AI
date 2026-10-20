# ============================================================
# DISASTERFLOW AI — PHASE 4B
# BASELINE FLOOD PREDICTION MODELS
# ============================================================

from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")


print("=" * 70)
print("DISASTERFLOW AI — PHASE 4B")
print("Baseline Flood Prediction Models")
print("=" * 70)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = (
    BASE_DIR
    / "data"
    / "processed"
)

MODEL_DIR = (
    BASE_DIR
    / "models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


TRAIN_FILE = DATA_DIR / "ml_train.csv"
VAL_FILE = DATA_DIR / "ml_validation.csv"
TEST_FILE = DATA_DIR / "ml_test.csv"


# ============================================================
# TARGET
# ============================================================

TARGET = "future_flood_event_6h"


# ============================================================
# CHECK FILES
# ============================================================

print("\n[1/9] Checking datasets...")

for file in [
    TRAIN_FILE,
    VAL_FILE,
    TEST_FILE
]:

    if not file.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{file}"
        )

    print(
        "✓",
        file.name
    )


# ============================================================
# LOAD DATA
# ============================================================

print("\n[2/9] Loading datasets...")


train = pd.read_csv(
    TRAIN_FILE
)

val = pd.read_csv(
    VAL_FILE
)

test = pd.read_csv(
    TEST_FILE
)


print(
    "Train:",
    train.shape
)

print(
    "Validation:",
    val.shape
)

print(
    "Test:",
    test.shape
)


# ============================================================
# FEATURE COLUMNS
# ============================================================

META_COLUMNS = [
    "grid_id",
    "time"
]


feature_columns = [
    col
    for col in train.columns
    if col not in META_COLUMNS
    and col != TARGET
]


print(
    "\nFeature count:",
    len(feature_columns)
)

print("\nFeatures:")

for i, feature in enumerate(
    feature_columns,
    start=1
):

    print(
        f"{i:02d}. {feature}"
    )


# ============================================================
# BUILD X / y
# ============================================================

print("\n[3/9] Preparing X and y...")


X_train = train[
    feature_columns
].astype(np.float32)

y_train = train[
    TARGET
].astype(int)


X_val = val[
    feature_columns
].astype(np.float32)

y_val = val[
    TARGET
].astype(int)


X_test = test[
    feature_columns
].astype(np.float32)

y_test = test[
    TARGET
].astype(int)


print(
    "\nX_train:",
    X_train.shape
)

print(
    "X_val:",
    X_val.shape
)

print(
    "X_test:",
    X_test.shape
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\nTarget distribution:")

print(
    "\nTrain:"
)

print(
    y_train.value_counts()
)


print(
    "\nValidation:"
)

print(
    y_val.value_counts()
)


print(
    "\nTest:"
)

print(
    y_test.value_counts()
)


# ============================================================
# CLASS WEIGHT
# ============================================================

negative_count = (
    y_train == 0
).sum()

positive_count = (
    y_train == 1
).sum()


if positive_count == 0:

    raise ValueError(
        "Training dataset contains no positive flood events."
    )


scale_pos_weight = (
    negative_count
    /
    positive_count
)


print(
    "\nClass imbalance:"
)

print(
    "Negative samples:",
    negative_count
)

print(
    "Positive samples:",
    positive_count
)

print(
    "scale_pos_weight:",
    round(
        scale_pos_weight,
        2
    )
)


# ============================================================
# STANDARDIZATION FOR LOGISTIC REGRESSION
# ============================================================

print(
    "\n[4/9] Preparing standardized data..."
)


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_val_scaled = scaler.transform(
    X_val
)

X_test_scaled = scaler.transform(
    X_test
)


print(
    "✓ Standardization complete"
)


# ============================================================
# METRIC FUNCTION
# ============================================================

def evaluate_model(
    model_name,
    y_true,
    probabilities
):

    predictions = (
        probabilities >= 0.5
    ).astype(int)


    accuracy = accuracy_score(
        y_true,
        predictions
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_true,
        probabilities
    )

    pr_auc = average_precision_score(
        y_true,
        probabilities
    )


    cm = confusion_matrix(
        y_true,
        predictions
    )


    print(
        "\n" + "-" * 65
    )

    print(
        model_name
    )

    print(
        "-" * 65
    )

    print(
        f"Accuracy : {accuracy:.6f}"
    )

    print(
        f"Precision: {precision:.6f}"
    )

    print(
        f"Recall   : {recall:.6f}"
    )

    print(
        f"F1 Score : {f1:.6f}"
    )

    print(
        f"ROC-AUC  : {roc_auc:.6f}"
    )

    print(
        f"PR-AUC   : {pr_auc:.6f}"
    )


    print(
        "\nConfusion Matrix:"
    )

    print(cm)


    return {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc
    }


# ============================================================
# RESULTS
# ============================================================

results = []


# ============================================================
# MODEL 1 — LOGISTIC REGRESSION
# ============================================================

print(
    "\n[5/9] Training Logistic Regression..."
)


logistic_model = LogisticRegression(
    max_iter=500,
    class_weight="balanced",
    solver="lbfgs"
)


logistic_model.fit(
    X_train_scaled,
    y_train
)


logistic_val_prob = (
    logistic_model
    .predict_proba(X_val_scaled)[:, 1]
)


logistic_test_prob = (
    logistic_model
    .predict_proba(X_test_scaled)[:, 1]
)


print(
    "\nLogistic Regression — Validation"
)

logistic_val_metrics = evaluate_model(
    "Logistic Regression — Validation",
    y_val,
    logistic_val_prob
)


print(
    "\nLogistic Regression — Test"
)

logistic_test_metrics = evaluate_model(
    "Logistic Regression — Test",
    y_test,
    logistic_test_prob
)


results.append(
    {
        "model": "Logistic Regression",
        **{
            key: logistic_test_metrics[key]
            for key in [
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "pr_auc"
            ]
        }
    }
)


joblib.dump(
    logistic_model,
    MODEL_DIR / "logistic_flood_model.joblib"
)

joblib.dump(
    scaler,
    MODEL_DIR / "logistic_scaler.joblib"
)


# ============================================================
# MODEL 2 — RANDOM FOREST
# ============================================================

print(
    "\n[6/9] Training Random Forest..."
)


random_forest = RandomForestClassifier(
    n_estimators=200,
    max_depth=18,
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    n_jobs=-1,
    random_state=42
)


random_forest.fit(
    X_train,
    y_train
)


rf_val_prob = (
    random_forest
    .predict_proba(X_val)[:, 1]
)


rf_test_prob = (
    random_forest
    .predict_proba(X_test)[:, 1]
)


print(
    "\nRandom Forest — Validation"
)

rf_val_metrics = evaluate_model(
    "Random Forest — Validation",
    y_val,
    rf_val_prob
)


print(
    "\nRandom Forest — Test"
)

rf_test_metrics = evaluate_model(
    "Random Forest — Test",
    y_test,
    rf_test_prob
)


results.append(
    {
        "model": "Random Forest",
        **{
            key: rf_test_metrics[key]
            for key in [
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "pr_auc"
            ]
        }
    }
)


joblib.dump(
    random_forest,
    MODEL_DIR / "random_forest_flood_model.joblib"
)


# ============================================================
# MODEL 3 — XGBOOST
# ============================================================

print(
    "\n[7/9] Training XGBoost..."
)


try:

    from xgboost import XGBClassifier

except ImportError:

    print(
        "\n❌ XGBoost is not installed."
    )

    print(
        "Install it using:"
    )

    print(
        "pip install xgboost"
    )

    raise


xgb_model = XGBClassifier(

    n_estimators=300,

    max_depth=8,

    learning_rate=0.08,

    subsample=0.8,

    colsample_bytree=0.8,

    min_child_weight=3,

    gamma=0.1,

    reg_alpha=0.1,

    reg_lambda=1.0,

    objective="binary:logistic",

    eval_metric="aucpr",

    scale_pos_weight=scale_pos_weight,

    tree_method="hist",

    n_jobs=-1,

    random_state=42
)


xgb_model.fit(
    X_train,
    y_train
)


xgb_val_prob = (
    xgb_model
    .predict_proba(X_val)[:, 1]
)


xgb_test_prob = (
    xgb_model
    .predict_proba(X_test)[:, 1]
)


print(
    "\nXGBoost — Validation"
)

xgb_val_metrics = evaluate_model(
    "XGBoost — Validation",
    y_val,
    xgb_val_prob
)


print(
    "\nXGBoost — Test"
)

xgb_test_metrics = evaluate_model(
    "XGBoost — Test",
    y_test,
    xgb_test_prob
)


results.append(
    {
        "model": "XGBoost",
        **{
            key: xgb_test_metrics[key]
            for key in [
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "pr_auc"
            ]
        }
    }
)


joblib.dump(
    xgb_model,
    MODEL_DIR / "xgboost_flood_model.joblib"
)


# ============================================================
# MODEL COMPARISON
# ============================================================

print(
    "\n[8/9] Comparing models..."
)


results_df = pd.DataFrame(
    results
)


results_df = results_df.sort_values(
    "pr_auc",
    ascending=False
)


print(
    "\n" + "=" * 70
)

print(
    "MODEL PERFORMANCE COMPARISON"
)

print(
    "=" * 70
)


print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_file = (
    MODEL_DIR
    / "baseline_model_results.csv"
)


results_df.to_csv(
    results_file,
    index=False
)


# ============================================================
# BEST MODEL
# ============================================================

best_model_name = (
    results_df
    .iloc[0]["model"]
)


print(
    "\nBest model based on PR-AUC:",
    best_model_name
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print(
    "\n[9/9] Extracting feature importance..."
)


if best_model_name == "XGBoost":

    importance = (
        xgb_model.feature_importances_
    )

elif best_model_name == "Random Forest":

    importance = (
        random_forest.feature_importances_
    )

else:

    importance = (
        np.abs(
            logistic_model.coef_[0]
        )
    )


importance_df = pd.DataFrame({

    "feature":
        feature_columns,

    "importance":
        importance

})


importance_df = (
    importance_df
    .sort_values(
        "importance",
        ascending=False
    )
)


importance_file = (
    MODEL_DIR
    / "feature_importance.csv"
)


importance_df.to_csv(
    importance_file,
    index=False
)


print(
    "\nTop 15 features:"
)

print(
    importance_df
    .head(15)
    .to_string(index=False)
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "PHASE 4B COMPLETE"
)

print(
    "=" * 70
)


print(
    "\nModels trained:"
)

print(
    "  ✓ Logistic Regression"
)

print(
    "  ✓ Random Forest"
)

print(
    "  ✓ XGBoost"
)


print(
    "\nBest model:",
    best_model_name
)


print(
    "\nResults saved:"
)

print(
    "  ✓",
    results_file
)


print(
    "  ✓",
    importance_file
)


print(
    "\nModels saved in:"
)

print(
    "  ",
    MODEL_DIR
)


print(
    "\n✓ Phase 4B finished successfully."
)

print(
    "=" * 70
)