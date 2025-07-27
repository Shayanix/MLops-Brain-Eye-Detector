# MLops-Brain-Eye-Detector 🧠👁️

An end-to-end MLOps project for classifying eye state (open or closed) using EEG brainwave data. This project demonstrates a complete machine learning pipeline with MLflow for experiment tracking, model versioning, and deployment.

[![CI/CD Pipeline](https://github.com/Shayanix/MLops-Brain-Eye-Detector/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/Shayanix/MLops-Brain-Eye-Detector/actions/workflows/ci-cd.yml)

## Project Overview 📊

This project implements a machine learning pipeline to detect eye states (open/closed) using EEG (Electroencephalography) data. It showcases MLOps best practices including:

- Data processing and validation
- Model training with hyperparameter optimization
- Experiment tracking using MLflow
- Model versioning and staging
- API deployment with FastAPI
- Containerization with Docker
- Continuous Integration/Deployment (CI/CD)
- Model monitoring and drift detection

## Table of Contents 📑

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Model Training](#model-training)
- [MLflow Integration](#mlflow-integration)
- [Monitoring](#monitoring)
- [Testing](#testing)
- [Docker Deployment](#docker-deployment)
- [Contributing](#contributing)

## Project Structure 🗂️

```
mlops-brain-eye-detector/
├── config.yaml           # Configuration settings
├── Dockerfile           # Container definition
├── requirements.txt     # Project dependencies
├── README.md           # Project documentation
├── tests/              # Test files
│   └── test_api.py     # API tests
├── data/               # Data files
│   ├── raw/           # Original dataset
│   └── processed/     # Processed data
├── notebooks/          # Jupyter notebooks
│   └── eda_initial_exploration.ipynb
└── src/
    ├── data/          # Data processing
    │   └── preprocess.py
    ├── models/        # Model training
    │   ├── train_model.py
    │   ├── hyperopt_train.py
    │   └── register_best_model.py
    ├── predict/       # API service
    │   └── app.py
    └── monitoring/    # Model monitoring
        └── model_monitor.py
```

## Installation 🛠️

1. Clone the repository:
\`\`\`bash
git clone https://github.com/Shayanix/MLops-Brain-Eye-Detector.git
cd MLops-Brain-Eye-Detector
\`\`\`

2. Create and activate a virtual environment:
\`\`\`bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
\`\`\`

3. Install dependencies:
\`\`\`bash
pip install -r requirements.txt
\`\`\`

## Usage 🚀

### Data Processing

Process the raw EEG data:
\`\`\`bash
python src/data/preprocess.py
\`\`\`

### Model Training

Train the Random Forest model:
\`\`\`bash
python src/models/train_model.py
\`\`\`

For hyperparameter optimization:
\`\`\`bash
python src/models/hyperopt_train.py
\`\`\`

### Model Registration

Register the best model to MLflow:
\`\`\`bash
python src/models/register_best_model.py
\`\`\`

### Start the API

Run the FastAPI service:
\`\`\`bash
uvicorn src.predict.app:app --reload
\`\`\`

## API Documentation 📚

### Endpoints

- `POST /predict`
  - Input: EEG readings from 14 electrodes
  - Output: Eye state prediction (Open/Closed)
  
- `GET /health`
  - Check API and model health status
  
- `GET /metrics`
  - Get model monitoring metrics

### Example Request

\`\`\`python
import requests

data = {
    "AF3": 4329.23,
    "F7": 4009.23,
    "F3": 4289.23,
    "FC5": 4148.21,
    "T7": 4350.26,
    "P7": 4586.15,
    "O1": 4096.00,
    "O2": 4129.23,
    "P8": 4356.41,
    "T8": 4216.41,
    "FC6": 4088.97,
    "F4": 4273.85,
    "F8": 4148.72,
    "AF4": 4163.08
}

response = requests.post("http://localhost:8000/predict", json=data)
print(response.json())
\`\`\`

## MLflow Integration 📈

The project uses MLflow for:
- Experiment tracking
- Model versioning
- Model registry
- Model staging (Development/Staging/Production)

Access MLflow UI:
\`\`\`bash
mlflow ui --backend-store-uri src/models/mlruns
\`\`\`

## Monitoring 📊

The monitoring system tracks:
- Model performance metrics
- Prediction latency
- Feature statistics
- Data drift indicators

Access monitoring metrics via the `/metrics` endpoint.

## Testing 🧪

Run the test suite:
\`\`\`bash
pytest tests/ --cov=src
\`\`\`

## Docker Deployment 🐳

Build the Docker image:
\`\`\`bash
docker build -t eeg-eye-detector .
\`\`\`

Run the container:
\`\`\`bash
docker run -p 8000:8000 eeg-eye-detector
\`\`\`

## Contributing 🤝

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License 📄

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments 🙏

- EEG Eye State Dataset from UCI Machine Learning Repository
- MLflow for experiment tracking
- FastAPI for API development
- scikit-learn for machine learning
