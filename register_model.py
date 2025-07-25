import mlflow
from mlflow import sklearn
import os



mlflow.set_tracking_uri(os.path.join(os.path.dirname(__file__), "src", "models", "mlruns"))
mlflow.set_tracking_uri("src/models/mlruns")
# Replace this with your actual Run ID

RUN_ID = "f64a4d0a0e59452d961b958ccbdf57c3"

# Path to the model inside the run

MODEL_PATH = f"runs:/{RUN_ID}/model"

# Register the model

mlflow.register_model(
    model_uri=MODEL_PATH,
    name="EEG-RF-Model"
)

print(f"Model from run {RUN_ID} successfully registered as 'EEG-RF-Model'")
