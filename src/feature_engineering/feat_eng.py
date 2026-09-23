import pandas as pd
import numpy as np

def feature_engineering(df_model):
    """Apply comprehensive feature engineering"""
    # 1. Buat salinan data agar aman dan tidak merusak DataFrame asli
    df_result = df_model.copy()

    # =========================================================================
    # DAFTAR KOLOM YANG WAJIB DIUBAH MENJADI ANGKA DULUAN
    # Karena kolom-kolom ini akan langsung dipakai dalam rumus matematika di bawah!
    # =========================================================================
    cols_to_clean = [
        "Outstanding_Debt", "Annual_Income", "Monthly_Inhand_Salary", 
        "Total_EMI_per_month", "Amount_invested_monthly", 
        "Num_of_Delayed_Payment", "Num_of_Loan", "Num_Credit_Inquiries"
    ]
    
    for col in cols_to_clean:
        if col in df_result.columns:
            # Langkah A: Paksa jadi string lalu hapus semua karakter kotor KECUALI angka dan titik desimal
            df_result[col] = df_result[col].astype(str).str.replace(r'[^0-9.-]', '', regex=True)
            
            # Langkah B: Ubah tipe datanya dari string menjadi numerik murni (float/int)
            df_result[col] = pd.to_numeric(df_result[col], errors="coerce")
            
            # Langkah C: Beri jaring pengaman median agar tidak ada nilai kosong (NaN) yang merusak rumus matematika
            median_val = df_result[col].median()
            df_result[col] = df_result[col].fillna(median_val if not pd.isna(median_val) else 0)

    # =========================================================================
    # 2. Proses Ekstraksi Teks Credit_History_Age (Mencari total bulan)
    # =========================================================================
    if 'Credit_History_Age' in df_result.columns:
        df_result['Credit_History_Age'] = df_result['Credit_History_Age'].astype(str)
        years = df_result["Credit_History_Age"].str.extract(r"(\d+)\s+Years")[0].astype(float)
        months = df_result["Credit_History_Age"].str.extract(r"(\d+)\s+Months")[0].astype(float)
        
        df_result["Credit_History_Age_Months"] = (years.fillna(0) * 12) + months.fillna(0)
        df_result = df_result.drop(columns=["Credit_History_Age"])

    # Nilai epsilon untuk menghindari pembagian dengan angka nol
    eps = 1e-6

    # =========================================================================
    # 3. Perhitungan Fitur Rasio Finansial (100% AMAN DARI ERROR TYPEERROR)
    # =========================================================================
    # Debt to Income Ratio
    df_result["Debt_to_Income_Ratio"] = (
        df_result["Outstanding_Debt"] / (df_result["Annual_Income"] + eps)
    )

    # Income to EMI Ratio
    df_result["Income_to_EMI_Ratio"] = (
        df_result["Monthly_Inhand_Salary"] / (df_result["Total_EMI_per_month"] + eps)
    )

    # Savings Ratio
    df_result["Savings_Ratio"] = (
        df_result["Amount_invested_monthly"] / (df_result["Monthly_Inhand_Salary"] + eps)
    )

    # Remaining money after EMI
    df_result["Remaining_Money"] = (
        df_result["Monthly_Inhand_Salary"] - df_result["Total_EMI_per_month"]
    )

    # Delayed payments per loan
    df_result["Delayed_Payment_Ratio"] = (
        df_result["Num_of_Delayed_Payment"] / (df_result["Num_of_Loan"] + eps)
    )

    # Credit inquiries per loan
    df_result["Inquiry_per_Loan"] = (
        df_result["Num_Credit_Inquiries"] / (df_result["Num_of_Loan"] + eps)
    )
    
    print("Feature Engineering is done. :D")

    return df_result