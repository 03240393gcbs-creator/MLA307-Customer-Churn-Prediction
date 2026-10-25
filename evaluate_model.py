import os
import json
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorflow import keras
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "member2_outputs")

DATA_PATH = os.path.join(OUT, "sequence_data_v2.npz")
WEIGHTS_PATH = os.path.join(OUT, "class_weights_v2.json")
MODEL_PATH = os.path.join(OUT, "final_model.keras")

os.makedirs(OUT, exist_ok=True)

# --------------------------------------------------
# 1. Load prepared data and saved model
# --------------------------------------------------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Missing {DATA_PATH}. Check that sequence_data_v2.npz exists."
    )

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Missing {MODEL_PATH}. Run model.py first to train and save the model."
    )

data = np.load(DATA_PATH, allow_pickle=True)

required = [
    "X_train", "y_train",
    "X_val", "y_val",
    "X_test", "y_test",
    "test_user_ids",
]

missing = [key for key in required if key not in data.files]
if missing:
    raise KeyError(f"Missing arrays in sequence data: {missing}")

X_train = data["X_train"].astype("float32")
y_train = data["y_train"].astype("int32").ravel()

X_val = data["X_val"].astype("float32")
y_val = data["y_val"].astype("int32").ravel()

X_test = data["X_test"].astype("float32")
y_test = data["y_test"].astype("int32").ravel()

test_ids = data["test_user_ids"]

if len(np.unique(y_train)) < 2:
    raise ValueError("Training data must contain both churn classes.")

if len(np.unique(y_val)) < 2 or len(np.unique(y_test)) < 2:
    raise ValueError(
        "Validation and test sets must each contain both classes "
        "to calculate ROC-AUC reliably."
    )

with open(WEIGHTS_PATH, "r") as f:
    class_weights = {
        int(k): float(v)
        for k, v in json.load(f).items()
    }

print("Training shape:", X_train.shape)
print("Validation shape:", X_val.shape)
print("Test shape:", X_test.shape)

model = keras.models.load_model(MODEL_PATH)

# --------------------------------------------------
# 2. Train a logistic regression baseline
# --------------------------------------------------
# Flatten each customer's time sequence for the baseline.
X_train_flat = X_train.reshape(len(X_train), -1)
X_val_flat = X_val.reshape(len(X_val), -1)
X_test_flat = X_test.reshape(len(X_test), -1)

baseline = LogisticRegression(
    class_weight=class_weights,
    max_iter=2000,
    C=0.1,
    random_state=42,
)

baseline.fit(X_train_flat, y_train)

baseline_val_prob = baseline.predict_proba(X_val_flat)[:, 1]
baseline_test_prob = baseline.predict_proba(X_test_flat)[:, 1]

# --------------------------------------------------
# 3. Predict with the saved sequence model
# --------------------------------------------------
sequence_val_prob = model.predict(X_val, verbose=0).ravel()
sequence_test_prob = model.predict(X_test, verbose=0).ravel()

# --------------------------------------------------
# 4. Choose the sequence-model threshold on validation
# --------------------------------------------------
# Select the threshold with the best validation F1.
# The test set is NOT used to select this threshold.
thresholds = np.arange(0.05, 0.951, 0.01)

threshold_results = []

for threshold in thresholds:
    pred = (sequence_val_prob >= threshold).astype(int)

    threshold_results.append({
        "threshold": float(threshold),
        "precision": precision_score(
            y_val, pred, zero_division=0
        ),
        "recall": recall_score(
            y_val, pred, zero_division=0
        ),
        "f1": f1_score(
            y_val, pred, zero_division=0
        ),
    })

threshold_table = pd.DataFrame(threshold_results)

# Prefer the threshold with the best F1.
# If tied, prefer the threshold with higher recall.
best_row = threshold_table.sort_values(
    ["f1", "recall", "threshold"],
    ascending=[False, False, False],
).iloc[0]

best_threshold = float(best_row["threshold"])

threshold_table.to_csv(
    os.path.join(OUT, "validation_threshold_results.csv"),
    index=False,
)

print(f"\nSelected validation threshold: {best_threshold:.2f}")
print(
    f"Validation precision: {best_row['precision']:.3f}, "
    f"recall: {best_row['recall']:.3f}, "
    f"F1: {best_row['f1']:.3f}"
)

# --------------------------------------------------
# 5. Evaluate both models on the test set
# --------------------------------------------------
def evaluate_model(name, y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    return {
        "model": name,
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "f1_score": f1_score(
            y_true, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "true_negatives": int(
            confusion_matrix(y_true, predictions, labels=[0, 1])[0, 0]
        ),
        "false_positives": int(
            confusion_matrix(y_true, predictions, labels=[0, 1])[0, 1]
        ),
        "false_negatives": int(
            confusion_matrix(y_true, predictions, labels=[0, 1])[1, 0]
        ),
        "true_positives": int(
            confusion_matrix(y_true, predictions, labels=[0, 1])[1, 1]
        ),
    }

# Baseline uses the standard 0.50 threshold.
baseline_result = evaluate_model(
    "Logistic Regression baseline",
    y_test,
    baseline_test_prob,
    0.50,
)

sequence_result = evaluate_model(
    "Saved GRU/LSTM sequence model",
    y_test,
    sequence_test_prob,
    best_threshold,
)

results = pd.DataFrame([baseline_result, sequence_result])
results.to_csv(
    os.path.join(OUT, "test_metrics.csv"),
    index=False,
)

print("\nTEST-SET MODEL COMPARISON")
print(results.round(3).to_string(index=False))

# Save a readable classification report for the sequence model.
sequence_predictions = (
    sequence_test_prob >= best_threshold
).astype(int)

report = classification_report(
    y_test,
    sequence_predictions,
    labels=[0, 1],
    target_names=["Not churned", "Churned"],
    zero_division=0,
)

with open(os.path.join(OUT, "classification_report.txt"), "w") as f:
    f.write("Saved GRU/LSTM sequence model\n")
    f.write(f"Threshold selected on validation: {best_threshold:.2f}\n\n")
    f.write(report)

print("\nSEQUENCE MODEL CLASSIFICATION REPORT")
print(report)

# --------------------------------------------------
# 6. Save confusion matrix
# --------------------------------------------------
cm = confusion_matrix(
    y_test,
    sequence_predictions,
    labels=[0, 1],
)

pd.DataFrame(
    cm,
    index=["Actual not churned", "Actual churned"],
    columns=["Predicted not churned", "Predicted churned"],
).to_csv(os.path.join(OUT, "confusion_matrix.csv"))

# --------------------------------------------------
# 7. Plot ROC curves
# --------------------------------------------------
plt.figure(figsize=(6, 5))

for name, probabilities in [
    ("Logistic Regression", baseline_test_prob),
    ("GRU/LSTM", sequence_test_prob),
]:
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    auc_value = roc_auc_score(y_test, probabilities)

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC = {auc_value:.3f})",
    )

plt.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Test-Set ROC Curves")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "roc_curves.png"), dpi=150)
plt.close()

# --------------------------------------------------
# 8. Save test predictions and at-risk customer list
# --------------------------------------------------
if len(test_ids) == len(y_test):
    customer_ids = test_ids
else:
    customer_ids = np.arange(len(y_test))

predictions_df = pd.DataFrame({
    "customer_id": customer_ids,
    "actual_churn": y_test,
    "churn_probability": sequence_test_prob,
    "predicted_churn": sequence_predictions,
    "threshold": best_threshold,
})

predictions_df.to_csv(
    os.path.join(OUT, "test_churn_predictions.csv"),
    index=False,
)

# Top 20 customers with the highest predicted churn probabilities.
at_risk = predictions_df.sort_values(
    "churn_probability",
    ascending=False,
).head(20)

at_risk.to_csv(
    os.path.join(OUT, "at_risk_customers_top20.csv"),
    index=False,
)

print("\nTop at-risk customers:")
print(at_risk.to_string(index=False))

print("\nEvaluation completed.")
print("Saved outputs to:", OUT)
