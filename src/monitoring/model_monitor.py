from typing import Dict, Any
import numpy as np
from datetime import datetime
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelMonitor:
    def __init__(self, metrics_path: str = "metrics"):
        self.metrics_path = Path(metrics_path)
        self.metrics_path.mkdir(parents=True, exist_ok=True)
        
    def log_prediction(self, input_data: Dict[str, Any], prediction: Any, 
                      prediction_time: float, model_version: str):
        """Log prediction details for monitoring."""
        try:
            timestamp = datetime.now().isoformat()
            log_entry = {
                "timestamp": timestamp,
                "input_features": input_data,
                "prediction": prediction,
                "prediction_time_ms": prediction_time * 1000,
                "model_version": model_version
            }
            
            # Save to daily log file
            date_str = datetime.now().strftime("%Y-%m-%d")
            log_file = self.metrics_path / f"predictions_{date_str}.jsonl"
            
            with open(log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
                
        except Exception as e:
            logger.error(f"Error logging prediction: {str(e)}")
            
    def calculate_metrics(self, timeframe: str = "daily"):
        """Calculate monitoring metrics for the specified timeframe."""
        try:
            metrics = {
                "total_predictions": 0,
                "avg_prediction_time": 0.0,
                "feature_statistics": {}
            }
            
            # Load relevant log files based on timeframe
            date_str = datetime.now().strftime("%Y-%m-%d")
            log_file = self.metrics_path / f"predictions_{date_str}.jsonl"
            
            if not log_file.exists():
                return metrics
                
            prediction_times = []
            feature_values = {}
            
            with open(log_file, "r") as f:
                for line in f:
                    entry = json.loads(line)
                    prediction_times.append(entry["prediction_time_ms"])
                    
                    # Collect feature statistics
                    for feature, value in entry["input_features"].items():
                        if feature not in feature_values:
                            feature_values[feature] = []
                        feature_values[feature].append(value)
            
            # Calculate metrics
            metrics["total_predictions"] = len(prediction_times)
            metrics["avg_prediction_time"] = np.mean(prediction_times)
            
            # Calculate feature statistics
            for feature, values in feature_values.items():
                metrics["feature_statistics"][feature] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values)
                }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return None
