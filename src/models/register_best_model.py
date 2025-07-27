import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn


# Load processed data

X_train = pd.read_csv("../data/data/processed/X_train.csv")
y_train = pd.read_csv("../data/data/processed/y_train.csv")

best_params = {
    "n_estimators": 150,
    "max_depth": None,
    "min_samples_split": 2
}

# Start MLflow run to save the final model

with mlflow.start_run(run_name="final_model_training"):
    clf = RandomForestClassifier(**best_params, random_state=42)
    clf.fit(X_train, y_train.values.ravel())

    # Log model to MLflow as artifact
    mlflow.sklearn.log_model(clf, "model")

    # Log parameters
    mlflow.log_params(best_params)
    print("Final model saved to MLflow!")