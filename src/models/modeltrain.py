import mlflow
import mlflow.sklearn
from config.config import (
    LG_C, LG_MAXITER, MLFLOW_TRACKING_URI, ARTIFACT_PIPELINE, 
    ARTIFACT_PIPELINE_DT, ARTIFACT_PIPELINE_RF, DT_MAX_DEP, 
    DT_MIN_SAMP_LEAF, DT_MIN_SAMP_SPLIT, RF_EST, RF_MAX_DEP, RF_MIN_SAMP
)
from src.utils.io import save_artifact

class ModelTrainer:
    def __init__(self, experiment_name="UAS--Predictive--Pipeline"):
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(self.experiment_name)

    def _prepare_data(self, train_prepro):
        X = train_prepro.drop("Credit_Score", axis=1)
        y = train_prepro["Credit_Score"]
        return X, y

    def train_logreg(self, pipeline, train_prepro):
        X, y = self._prepare_data(train_prepro)
        with mlflow.start_run() as run:
            mlflow.log_params({"max_iter": LG_MAXITER, "C": LG_C})
            pipeline.fit(X, y)
            mlflow.sklearn.log_model(pipeline, "model_logreg", registered_model_name="uas_model1")
            save_artifact(pipeline, ARTIFACT_PIPELINE)
            print(f"Logistic Regression trained. Run ID: {run.info.run_id}")
            return run.info.run_id

    def train_rf(self, pipeline, train_prepro):
        X, y = self._prepare_data(train_prepro)
        with mlflow.start_run() as run:
            mlflow.log_params({"max_depth": RF_MAX_DEP, "n_estimator": RF_EST, "min_sample_split": RF_MIN_SAMP})
            pipeline.fit(X, y)
            mlflow.sklearn.log_model(pipeline, "model_rf", registered_model_name="uas_model2")
            save_artifact(pipeline, ARTIFACT_PIPELINE_RF)
            print(f"Random Forest trained. Run ID: {run.info.run_id}")
            return run.info.run_id

    def train_dt(self, pipeline, train_prepro):
        X, y = self._prepare_data(train_prepro)
        with mlflow.start_run() as run:
            mlflow.log_params({"max_depth": DT_MAX_DEP, "min_sample_split": DT_MIN_SAMP_SPLIT, "min_sample_leaf": DT_MIN_SAMP_LEAF})
            pipeline.fit(X, y)
            mlflow.sklearn.log_model(pipeline, "model_dt", registered_model_name="uas_model3")
            save_artifact(pipeline, ARTIFACT_PIPELINE_DT)
            print(f"Decision Tree trained. Run ID: {run.info.run_id}")
            return run.info.run_id