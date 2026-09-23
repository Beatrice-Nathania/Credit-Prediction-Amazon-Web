from pathlib import Path

BASE_DIR      = Path(__file__).resolve().parent.parent
DATA_RAW_DIR  = BASE_DIR / "data" / "raw"
DATA_ING_DIR  = BASE_DIR / "data" / "ingested"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

ARTIFACT_PIPELINE     = ARTIFACTS_DIR / "model_logreg.joblib"
ARTIFACT_PIPELINE_RF     = ARTIFACTS_DIR / "model_rf.joblib"
ARTIFACT_PIPELINE_DT     = ARTIFACTS_DIR / "model_dt.joblib"

LG_C = 0.316735
LG_MAXITER = 200
RF_EST = 100
RF_MAX_DEP = 15
RF_MIN_SAMP = 5
DT_MAX_DEP = 10
DT_MIN_SAMP_SPLIT = 5
DT_MIN_SAMP_LEAF = 2
RANDOM_STATE = 42
TEST_SIZE    = 0.2

MLFLOW_TRACKING_URI = f"sqlite:///{BASE_DIR.parent / 'mlflow.db'}"
MLFLOW_EXP_PIPELINE = "Credit_Score Prediction"

ACCURACY_THRESHOLD = 0.7
