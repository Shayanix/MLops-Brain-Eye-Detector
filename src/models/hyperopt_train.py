import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK



# Load preprocessed data

X_train = pd.read_csv("../data/data/processed/X_train.csv")
X_test = pd.read_csv("../data/data/processed/X_test.csv")
y_train = pd.read_csv("../data/data/processed/y_train.csv")
y_test = pd.read_csv("../data/data/processed/y_test.csv")


def objective(params):
    with mlflow.start_run(nested=True):
        clf = RandomForestClassifier(**params,random_state=42)
        clf.fit(X_train, y_train.values.ravel())
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        mlflow.log_params(params)
        mlflow.log_metric("accuracy",acc)
        
        return {"loss":-acc,"status": STATUS_OK}
    
# Define Search Space

search_space ={
    "n_estimators":hp.choice("n_estimators",[50,100,150]),
    "max_depth": hp.choice("max_depth",[5,10,15,20,None]),
    "min_samples_split": hp.choice("min_samples_split",[2,5,10]),
    
}
# MLflow parent run

with mlflow.start_run(run_name="hyperopt_rf_search"):
    trials = Trials()
    best_result = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=20,
        trials=trials
    )

    print("Best Hyperparameters found:", best_result)
    