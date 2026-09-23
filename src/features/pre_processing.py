"""
Session 04 – Step 2: Preprocessing
Reads ingested data, splits, scales, and saves the preprocessor artifact.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from config.config import ARTIFACTS_DIR
import joblib

class DataPreprocessor:
    def __init__(self):
        self.column_rem = ["Unnamed: 0", 'ID', 'Customer_ID', "Name", "SSN", "Type_of_Loan"]
        self.cols_to_strip = ['Month', 'Occupation', 'Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour', 'Credit_Score']
        
        # Mapping tetap sama...
        self.month_mapping = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 'July': 7, 'August': 8}
        self.occupation_mapping = {'unknown': -1, 'Accountant': 0, 'Architect': 1, 'Developer': 2, 'Doctor': 3, 'Engineer': 4, 'Entrepreneur': 5, 'Journalist': 6, 'Lawyer': 7, 'Manager': 8, 'Mechanic': 9, 'MediaManager': 10, 'Musician': 11, 'Scientist': 12, 'Teacher': 13, 'Writer': 14}
        self.credit_mix_mapping = {'Bad': 0, 'Standard': 1, 'Good': 2, 'unknown': -1}
        self.payment_min_mapping = {'No': 0, 'Yes': 1, 'unknown': -1}
        self.payment_behaviour_mapping = {'LowspentSmallvaluepayments': 0, 'LowspentMediumvaluepayments': 1, 'LowspentLargevaluepayments': 2, 'HighspentSmallvaluepayments': 3, 'HighspentMediumvaluepayments': 4, 'HighspentLargevaluepayments': 5, 'unknown': -1}
        self.credit_score_mapping = {'Poor': 0, 'Standard': 1, 'Good': 2}
        
        self.scaler = StandardScaler()
        self.winsor_limits_ = {} # <--- Untuk menyimpan batas percentile dari data training
        model_dir = os.getenv("SM_MODEL_DIR", ARTIFACTS_DIR)
        self.scaler_path = os.path.join(model_dir, "scaler.joblib")

    def save_scaler(self):
        # Menyimpan scaler dan batas winsorizing bersamaan ke dalam satu file artifact
        joblib.dump({'scaler': self.scaler, 'winsor_limits': self.winsor_limits_}, self.scaler_path)

    def load_scaler(self):
        if os.path.exists(self.scaler_path):
            payload = joblib.load(self.scaler_path)
            # Mengecek apakah formatnya dictionary (baru) atau objek scaler tunggal (lama)
            if isinstance(payload, dict) and 'scaler' in payload:
                self.scaler = payload['scaler']
                self.winsor_limits_ = payload['winsor_limits']
            else:
                self.scaler = payload
                self.winsor_limits_ = {}
        else:
            raise FileNotFoundError(f"Scaler tidak ditemukan di {self.scaler_path}. Pastikan sudah melakukan training!")

    @staticmethod
    def _count_loans(text):
        if pd.isna(text) or str(text).strip().lower() in ['unknown', '', 'nan']:
            return 0
        return len([item for item in str(text).split(',') if item.strip()])

    def clean_and_encode(self, df):
        df_cleaned = df.copy()

        if 'Num_of_Loan' in df_cleaned.columns:
            if df_cleaned['Num_of_Loan'].dtype == 'object':
                df_cleaned['Num_of_Loan'] = df_cleaned['Num_of_Loan'].astype(str).str.replace(r'[^0-9.-]', '', regex=True)
            df_cleaned['Num_of_Loan'] = pd.to_numeric(df_cleaned['Num_of_Loan'], errors='coerce')
            
            if 'Type_of_Loan' in df_cleaned.columns:
                calculated_loans = df_cleaned['Type_of_Loan'].apply(self._count_loans)
                df_cleaned['Num_of_Loan'] = df_cleaned['Num_of_Loan'].fillna(calculated_loans)
            
            median_val = df_cleaned['Num_of_Loan'].median()
            df_cleaned['Num_of_Loan'] = df_cleaned['Num_of_Loan'].fillna(median_val if not pd.isna(median_val) else 0)

        # Drop kolom tidak terpakai
        existing_rem_cols = [col for col in self.column_rem if col in df_cleaned.columns]
        df_cleaned = df_cleaned.drop(columns=existing_rem_cols)

        numeric = df_cleaned.select_dtypes(exclude="object").columns
        categoric = df_cleaned.select_dtypes(include="object").columns

        for col in self.cols_to_strip:
            if col in df_cleaned.columns:
                df_cleaned[col] = df_cleaned[col].astype(str).str.replace(' ', '').str.replace('_', '').str.strip()
        
        # Jalankan Mapping Encoding
        if 'Month' in df_cleaned.columns:                  df_cleaned['Month'] = df_cleaned['Month'].map(self.month_mapping)
        if 'Occupation' in df_cleaned.columns:             df_cleaned['Occupation'] = df_cleaned['Occupation'].map(self.occupation_mapping)
        if 'Credit_Mix' in df_cleaned.columns:             df_cleaned['Credit_Mix'] = df_cleaned['Credit_Mix'].map(self.credit_mix_mapping)
        if 'Payment_of_Min_Amount' in df_cleaned.columns:  df_cleaned['Payment_of_Min_Amount'] = df_cleaned['Payment_of_Min_Amount'].map(self.payment_min_mapping)
        if 'Payment_Behaviour' in df_cleaned.columns:      df_cleaned['Payment_Behaviour'] = df_cleaned['Payment_Behaviour'].map(self.payment_behaviour_mapping)
        if 'Credit_Score' in df_cleaned.columns:           df_cleaned['Credit_Score'] = df_cleaned['Credit_Score'].map(self.credit_score_mapping)
   
        # Untuk kolom hasil mapping kategorikal yang gagal cocok, kita kembalikan ke angka -1 (unknown)
        for col in self.cols_to_strip:
            if col in df_cleaned.columns and col != 'Credit_Score':
                df_cleaned[col] = df_cleaned[col].fillna(-1)

        # Bersihkan string kotor pada sisa nilai numerik asli lainnya (seperti Age, Annual_Income)
        for col in df_cleaned.columns:
            if df_cleaned[col].dtype == 'object' and col != 'Credit_Score':
                df_cleaned[col] = df_cleaned[col].astype(str).str.replace(r'[^0-9.-]', '', regex=True)
                df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
                # Isi nilai numerik kosong dengan median kolom tersebut agar Scaler tidak crash
                median_num = df_cleaned[col].median()
                df_cleaned[col] = df_cleaned[col].fillna(median_num if not pd.isna(median_num) else 0)

        return df_cleaned, categoric, numeric

    def scale_features(self, df, num, is_train=True):
        df_scaled = df.copy()
        features_to_scale = [col for col in num if col != 'Credit_Score']
        
        # ── INTEGRASI LOGIKA 1: HANDLING NEGATIVE VALUES ──────────────────────
        col_neg = ['Age', 'Num_Bank_Accounts', 'Num_of_Loan', 'Num_of_Delayed_Payment']

        for col in col_neg:
            if col in df_scaled.columns:
                df_scaled[col] = df_scaled[col].clip(lower=0)

        print("✅ Nilai negatif pada kolom target berhasil")

        # ── INTEGRASI LOGIKA 2: WINSORIZING OUTLIERS ──────────────────────────
        numerical_cols = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['Unnamed: 0']
        numerical_cols = [col for col in numerical_cols if col not in exclude_cols]

        print(colom_yang_diproses := f"Memproses {len(numerical_cols)} kolom numerik...")

        if is_train:
            # Mode Training: Hitung batas baru berdasarkan data training aktual
            for col in numerical_cols:
                lower_limit = df_scaled[col].quantile(0.05)  # Batas bawah percentile ke-5
                upper_limit = df_scaled[col].quantile(0.95)  # Batas atas percentile ke-95
                
                # Simpan limit ke attribute class agar ikut terbawa saat di-save
                self.winsor_limits_[col] = (lower_limit, upper_limit)
                
                df_scaled[col] = df_scaled[col].clip(lower_limit, upper_limit)

            print("✅ Proses Winsorizing selesai\n")
            
            df_scaled[features_to_scale] = self.scaler.fit_transform(df_scaled[features_to_scale])
            self.save_scaler() # Simpan state setelah fit (termasuk isi winsor_limits_)
        else:
            # Mode Testing / SageMaker Inference Endpoint: Gunakan batas training lama
            self.load_scaler() 
            for col in numerical_cols:
                if col in self.winsor_limits_:
                    lower_limit, upper_limit = self.winsor_limits_[col]
                    df_scaled[col] = df_scaled[col].clip(lower_limit, upper_limit)
            
            print("✅ Proses Winsorizing selesai untuk semua kolom numerik (Inference Mode)!\n")
            df_scaled[features_to_scale] = self.scaler.transform(df_scaled[features_to_scale])
            
        return df_scaled

preprocessor = DataPreprocessor()

def preprocess(df_model, is_train):
    """
    Fungsi utama pipeline preprocessing dengan pemisahan logika scaling
    antara data train dan data testing.
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    
    df_processed, categoric, numeric = preprocessor.clean_and_encode(df_model)
    
    df_final = preprocessor.scale_features(df_processed, numeric, is_train=is_train)

    return df_final