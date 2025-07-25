import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,f1_score
import mlflow
import mlflow.sklearn


# Load preprocessed data

X_train = pd.read_csv("../data/data/processed/X_train.csv")
X_test = pd.read_csv("../data/data/processed/X_test.csv")
y_train = pd.read_csv("../data/data/processed/y_train.csv")
y_test = pd.read_csv("../data/data/processed/y_test.csv")

# Enable MLflow auto-logging

mlflow.sklearn.autolog()

with mlflow.start_run():
    
    # Train model
    
    clf = RandomForestClassifier(n_estimators=100,random_state=42)
    clf.fit(X_train,y_train.values.ravel())
    
    # Predict and evaluate
    
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred,pos_label="b'1'")
    
    print(f"Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")