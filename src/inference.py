import io
import json
import os
import numpy as np
import pandas as pd
import joblib
from features.pre_processing import preprocess
from feature_engineering.feat_eng import feature_engineering
from pathlib import Path

JSON_CONTENT_TYPE = "application/json"
CSV_CONTENT_TYPE = "text/csv"

CLASS_NAMES = ["Poor", "Standard", "Good"]

FEATURE_NAMES = [
    "Unnamed: 0",
    "ID",
    "Customer_ID",
    "Month",
    "Name",
    "Age",
    "SSN",
    "Occupation",
    "Annual_Income",
    "Monthly_Inhand_Salary",
    "Num_Bank_Accounts",
    "Num_Credit_Card",
    "Interest_Rate",
    "Num_of_Loan",
    "Type_of_Loan",
    "Delay_from_due_date",
    "Num_of_Delayed_Payment",
    "Changed_Credit_Limit",
    "Num_Credit_Inquiries",
    "Credit_Mix",
    "Outstanding_Debt",
    "Credit_Utilization_Ratio",
    "Credit_History_Age",
    "Payment_of_Min_Amount",
    "Total_EMI_per_month",
    "Amount_invested_monthly",
    "Payment_Behaviour",
    "Monthly_Balance"
]

def model_fn(model_dir: str):
    """
    Memuat model dan memastikan scaler tersedia.
    SageMaker akan mencari file di model_dir (S3 bucket path).
    """
    model_path = os.path.join(model_dir, "model_rf.joblib")
    scaler_path = os.path.join(model_dir, "scaler.joblib")
    
    # Load model
    model = joblib.load(model_path)
    
    if not os.path.exists(scaler_path):
        raise FileNotFoundError("Scaler artifact (scaler.joblib) tidak ditemukan di container!")
        
    return model

def input_fn(request_body, request_content_type: str) -> pd.DataFrame:
    if request_content_type == JSON_CONTENT_TYPE:
        payload = json.loads(request_body)
        instances = payload["instances"]
        df = pd.DataFrame(instances, columns=FEATURE_NAMES)
    elif request_content_type == CSV_CONTENT_TYPE:
        if isinstance(request_body, (bytes, bytearray)):
            request_body = request_body.decode("utf-8")
        df = pd.read_csv(io.StringIO(request_body.strip()), header=None, names=FEATURE_NAMES)
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")
    float_cols = [
        "Monthly_Inhand_Salary", "Num_Bank_Accounts", "Num_Credit_Card", 
        "Interest_Rate", "Delay_from_due_date", "Num_Credit_Inquiries", 
        "Credit_Utilization_Ratio", "Total_EMI_per_month"
    ]
    string_cols = [
        "ID", "Customer_ID", "Month", "Name", "Age", "SSN", "Occupation", 
        "Annual_Income", "Num_of_Loan", "Type_of_Loan", "Num_of_Delayed_Payment", 
        "Changed_Credit_Limit", "Credit_Mix", "Outstanding_Debt", 
        "Credit_History_Age", "Payment_of_Min_Amount", "Amount_invested_monthly", 
        "Payment_Behaviour", "Monthly_Balance"
    ]

    # Konversi tipe data awal secara aman
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str).replace("", "Unknown")

    return df

def predict_fn(input_data: pd.DataFrame, model) -> dict:
    feat = feature_engineering(input_data)
    feat_processed = preprocess(feat, is_train=False)
    probs = model.predict_proba(feat_processed)
    class_ids = np.argmax(probs, axis=1)
    labels = [str(CLASS_NAMES[int(i)]) for i in class_ids]
    
    return {
        "probabilities": probs.tolist(),
        "predictions": class_ids.tolist(),
        "labels": labels,
    }

def output_fn(prediction: dict, accept_content_type: str):
    if accept_content_type == JSON_CONTENT_TYPE or accept_content_type == "*/*":
        return json.dumps(prediction), JSON_CONTENT_TYPE
    raise ValueError(f"Unsupported accept type: {accept_content_type}")