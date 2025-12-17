import json
import joblib
import os
import pandas as pd
import mlflow
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--train_ready", type=str, required=True)
parser.add_argument("--test_ready", type=str, required=True)
parser.add_argument("--model_output", type=str, required=True)
parser.add_argument("--metrics_output", type=str, required=True)
parser.add_argument("--target_col", type=str, default="house_affiliation")
parser.add_argument("--max_depth", type=int, default=5)
parser.add_argument("--min_samples_split", type=int, default=2)
parser.add_argument("--min_samples_leaf", type=int, default=1)
parser.add_argument("--random_state", type=int, default=42)

args = parser.parse_args()

# Load data
X_train_path = os.path.join(args.train_ready, "X_train.csv")
y_train_path = os.path.join(args.train_ready, "y_train.csv")
X_test_path  = os.path.join(args.test_ready, "X_test.csv")
y_test_path  = os.path.join(args.test_ready, "y_test.csv")

for p in [X_train_path, y_train_path, X_test_path, y_test_path]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing required file: {p}")

X_train = pd.read_csv(X_train_path)
y_train = pd.read_csv(y_train_path)[args.target_col].astype(str)

X_test = pd.read_csv(X_test_path)
y_test = pd.read_csv(y_test_path)[args.target_col].astype(str)

# Train model
clf = DecisionTreeClassifier(
    max_depth=args.max_depth,
    min_samples_split=args.min_samples_split,
    min_samples_leaf=args.min_samples_leaf,
    random_state=args.random_state,
)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
acc = float(accuracy_score(y_test, y_pred))

report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

print(f"Accuracy: {acc}")
print("Classification Report:", report)

# Start MLflow run (it will attach to the current Azure ML run)
mlflow.start_run()

# Log params
mlflow.log_param("max_depth", args.max_depth)
mlflow.log_param("min_samples_split", args.min_samples_split)
mlflow.log_param("min_samples_leaf", args.min_samples_leaf)
mlflow.log_param("random_state", args.random_state)

# Log accuracy
mlflow.log_metric("accuracy", acc)

# Log classification report metrics
for label, metrics in report.items():
    if isinstance(metrics, dict):
        for metric_name, value in metrics.items():
            mlflow.log_metric(f"{label}_{metric_name}", value)
    else:
        mlflow.log_metric(label, metrics)

# Save model
os.makedirs(args.model_output, exist_ok=True)
model_path = os.path.join(args.model_output, "model.joblib")
joblib.dump(clf, model_path)

# End run
mlflow.end_run()

# Save metrics
os.makedirs(args.metrics_output, exist_ok=True)
metrics_path = os.path.join(args.metrics_output, "metrics.json")
with open(metrics_path, "w") as f:
    json.dump({"accuracy": acc, "classification_report": report}, f, indent=4)
