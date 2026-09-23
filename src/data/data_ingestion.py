"""
Session 04 – Step 1: Data Ingestion
Reads raw IRIS.csv and saves it to the ingested/ folder.
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from config.config import (
    DATA_RAW_DIR, DATA_ING_DIR
)

def ingest_data():
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    DATA_ING_DIR.mkdir(parents=True, exist_ok=True)
    
    raw = DATA_RAW_DIR / "data_D.csv"
    if not raw.exists():
        print("data_D.csv not found – importing data...")
        base = Path(__file__).resolve().parent
        df = pd.read_csv(base / "data_D.csv")
        df.to_csv(raw, sep = ",", index=False)
        print(f"Generated data_D.csv at {raw}")

    df = pd.read_csv(raw)
    assert not df.empty, "Dataset is empty"

    out = DATA_ING_DIR / "data_D.csv"
    df.to_csv(out,sep =",", index=False)
    print(f"Data ingested: {raw} → {out}")

def split_data(df):
    y = df['Credit_Score']
    x = df.drop(["Credit_Score"], axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    train_scaled = pd.concat([X_train, pd.Series(y_train, name = "Credit_Score")], axis=1)
    test_scaled = pd.concat([X_test, pd.Series(y_test, name = "Credit_Score")], axis=1)
    print("Splitting done")
    return train_scaled, test_scaled