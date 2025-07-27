import mlflow
import pandas as pd
import os


data_path = "../data/data/processed/eeg_data.csv"
df = pd.read_csv(data_path)


X = df

model_name = "EEG-RF-Model"
stage = "Staging"
mlflow.set_tracking_uri("src/models/mlruns")
model_uri = f"models:/{model_name}/{stage}"

model = mlflow.sklearn.load_model(model_uri)


predictions = model.predict(X)


print("Predictions:")
print(predictions[:10])
