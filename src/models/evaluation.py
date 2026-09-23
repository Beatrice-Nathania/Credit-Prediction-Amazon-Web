import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
from config.config import MLFLOW_TRACKING_URI

class ModelEvaluator:
    def __init__(self, tracking_uri=MLFLOW_TRACKING_URI):
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(self.tracking_uri)

    def evaluate(self, valid_prepro, run_id, artifact_name):
        """
        Melakukan evaluasi model dan mencatat metrik ke MLflow.
        """
        # Memisahkan fitur dan target
        x_test = valid_prepro.drop("Credit_Score", axis=1)
        y_test = valid_prepro["Credit_Score"]

        # Memuat model dari MLflow
        model = mlflow.sklearn.load_model(f"runs:/{run_id}/{artifact_name}")

        # Prediksi dan penghitungan metrik
        preds = model.predict(x_test)
        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, average="macro"),
            "recall": recall_score(y_test, preds, average="macro")
        }

        # Log metrik ke MLflow
        with mlflow.start_run(run_id=run_id):
            for name, value in metrics.items():
                mlflow.log_metric(name, value)

        print(f"Evaluation | Accuracy={metrics['accuracy']:.3f} | "
              f"Precision={metrics['precision']:.3f} | Recall={metrics['recall']:.3f}")
        
        return metrics['accuracy'], metrics['precision'], metrics['recall']

# Contoh Penggunaan:
if __name__ == "__main__":
    # Inisialisasi evaluator
    evaluator = ModelEvaluator()
    
    # Panggil fungsi evaluasi
    # acc, prec, rec = evaluator.evaluate(df_test, "run_id_anda", "model_rf")