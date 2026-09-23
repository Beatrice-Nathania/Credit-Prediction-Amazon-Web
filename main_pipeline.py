from src.data.data_ingestion import ingest_data, split_data
from src.pipelines.sklearn_pipeline import model_pipeline_lg, model_pipeline_dt, model_pipeline_rf
from src.models.modeltrain import ModelTrainer
from src.models.evaluation import ModelEvaluator
from src.feature_engineering.feat_eng import feature_engineering
from src.features.pre_processing import preprocess, DataPreprocessor
from config.config import DATA_ING_DIR, ARTIFACTS_DIR, ACCURACY_THRESHOLD, MLFLOW_TRACKING_URI
import pandas as pd
import tarfile
import os
from pathlib import Path

MODEL_FILENAME = "model_rf.joblib"
SCALER_NAME = "scaler.joblib"
TARBALL_PATH = os.path.join(ARTIFACTS_DIR, "model.tar.gz")

def main():
    print("=" * 50)
    print("Approach B – sklearn Pipeline")
    print("=" * 50)

    print("\nStep 1: Data Ingestion")
    ingest_data()

    print("\nStep 2: Load and Feature Engineering")
    df = pd.read_csv(DATA_ING_DIR / "data_D.csv")

    df = feature_engineering(df)

    print("\nStep 3: Split the data")
    train_scaled, test_scaled = split_data(df)
    train_scaled = preprocess(train_scaled, True)
    test_scaled = preprocess(test_scaled, False)
    print("Data Setelah Cleaning & Encoding:")
    print(df.head(5))

    print("\nStep 4: Build and Train Pipeline")
    pipeline_lg = model_pipeline_lg(train_scaled)
    pipeline_rf = model_pipeline_rf(train_scaled)
    pipeline_dt = model_pipeline_dt(train_scaled)
    
    trainer = ModelTrainer()

    run_id1 = trainer.train_logreg(pipeline_lg, train_scaled)
    run_id2 = trainer.train_rf(pipeline_rf, train_scaled)
    run_id3 = trainer.train_dt(pipeline_dt, train_scaled)

    print("\nStep 5: Evaluation")
    Evaluator = ModelEvaluator(tracking_uri=MLFLOW_TRACKING_URI)
    accuracy1, precision, recall = Evaluator.evaluate(
    valid_prepro=test_scaled, 
    run_id=run_id1, 
    artifact_name="model_logreg"
    )

    accuracy2, precision, recall = Evaluator.evaluate(
    valid_prepro=test_scaled, 
    run_id=run_id2, 
    artifact_name="model_rf"
    )

    accuracy3, precision, recall = Evaluator.evaluate(
    valid_prepro=test_scaled, 
    run_id=run_id3, 
    artifact_name="model_dt"
    )

    if accuracy1 >= accuracy2 and accuracy1 >= accuracy3 and accuracy1 >= ACCURACY_THRESHOLD:
        best_model = "Logistic Regression"
        best_accuracy = accuracy1
    elif accuracy2 >= accuracy1 and accuracy2 >= accuracy3 and accuracy2 >= ACCURACY_THRESHOLD:
        best_model = "Random Forest"
        best_accuracy = accuracy2
    else:
        best_model = "Decision Tree"
        best_accuracy = accuracy3
    
    print("\n" + "=" * 50)
    print(f"Model {best_model} APPROVED (accuracy={best_accuracy:.3f})")

    model_path = os.path.join(ARTIFACTS_DIR, MODEL_FILENAME)
    scaler_path = os.path.join(ARTIFACTS_DIR, SCALER_NAME)
    with tarfile.open(TARBALL_PATH, "w:gz") as tar:
        tar.add(model_path, arcname=MODEL_FILENAME)
        tar.add(scaler_path, arcname=SCALER_NAME)
        if os.path.exists("src"):
            tar.add("src", arcname="code")
    print(f"Packaged: {TARBALL_PATH}")

    print("Isi arsip saat ini:")
    with tarfile.open(TARBALL_PATH, "r:gz") as tar:
        for name in tar.getnames():
            print(f"- {name}")

    print("\nNext steps:")
    print("  Make the s3 bucket with:\n\n  aws s3 mb s3://your-bucket-name --region us-east-1\n\n  and upload with:")
    print(f"\n  aws s3 cp {TARBALL_PATH} s3://your-bucket-name/credit/model.tar.gz\n")

    


if __name__ == "__main__":
    main()
