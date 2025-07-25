import mlflow
from mlflow.exceptions import RestException

model_name = "EEG-RF-Model"
model_stage = "Staging"

# Get latest version of the model
mlflow.set_tracking_uri("src/models/mlruns")
client = mlflow.tracking.MlflowClient()
try:
    
    versions = client.search_model_versions(f"name='{model_name}'")
    
    if not versions:
        print(f"No versions found for model '{model_name}'. Please register the model first.")
    else:
        
        latest_version = max(versions, key=lambda x: int(x.version))
        
        print(f"Latest version found: {latest_version.version} (current stage: {latest_version.current_stage})")
        
        
        client.transition_model_version_stage(
            name=model_name,
            version=latest_version.version,
            stage=model_stage
        )
        
        print(f"Successfully transitioned version {latest_version.version} to {model_stage}")

except RestException as e:
    print(f"Error accessing model registry: {e}")
