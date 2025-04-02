import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import joblib
import pickle

# Load Preprocessed Data
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv').values.ravel()
y_test = pd.read_csv('y_test.csv').values.ravel()

# Define Models and Hyperparameter Grid
models = {
    'LinearRegression': (LinearRegression(), {}),
    'Ridge': (Ridge(), {'alpha': [0.1, 1.0, 10.0]}),
    'Lasso': (Lasso(), {'alpha': [0.1, 1.0, 10.0]}),
    'RandomForest': (RandomForestRegressor(), {'n_estimators': [50, 100, 200]})
}

best_model_instance = None
best_score = float('-inf')
best_model_name = ''

mlflow.set_experiment('kubeflow_experiment')

# Train and Evaluate Models
for model_name, (model, param_grid) in models.items():
    with mlflow.start_run(run_name=model_name):
        if param_grid:
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2')
            grid_search.fit(X_train, y_train)
            model_instance = grid_search.best_estimator_
            best_params = grid_search.best_params_
        else:
            model_instance = model.fit(X_train, y_train)
            best_params = {}
        
        predictions = model_instance.predict(X_test)
        r2 = r2_score(y_test, predictions)
        
        mlflow.log_params(best_params)
        mlflow.log_metric('R2 Score', r2)
        mlflow.sklearn.log_model(model_instance, model_name)
        
        if r2 > best_score:
                best_score = r2
                best_model_instance = model_instance
                best_model_name = model_name
                
# Save Model and Artifacts Separately
model_path = "models/fuel.pkl"
pickle.dump(best_model_name, open(model_path, 'wb'))
mlflow.sklearn.log_model(best_model_name, "fuel")
mlflow.log_artifact(model_path, artifact_path="artifacts")
        
# ✅ Register Model in MLflow Model Registry
model_uri = f"runs:/{mlflow.active_run().info.run_id}/fuel_model"
mlflow.register_model(model_uri, "FuelEfficiencyModel")