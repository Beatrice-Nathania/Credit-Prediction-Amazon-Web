import json
from inference import model_fn, input_fn, predict_fn, output_fn

MODEL_DIR = "./artifacts"  # Sesuaikan dengan folder lokal kamu

print("--- Menyiapkan Model ---")
context = model_fn(MODEL_DIR)
print("Model dan Scaler berhasil dimuat!\n")

mock_instances = [
    # Baris 1: Profil Risiko Tinggi (Sangat "Poor")
    [0, "0x45c7", "CUS_0x3969", "February", "Petunia Olaps", "30", "316-61-XXX", "Scientist", 
     "16154.25", 1000.0, 8, 8, 29, "5", "Credit-Builder Loan, Auto Loan, and Credit-Builder Loan", 31, 
     "20", "4.8", 10.0, "Bad", "1426.76", 35.0, "15 Years and 11 Months", "Yes", 30.0, 
     "116.134", "Low_spent_Large_value_payments", "247.416"],
     
    # Baris 2: Profil Standar (Kemungkinan "Standard")
    [1, "ID_002", "CUST_002", "January", "Jane Doe", "34", "123-46-XXX", "Teacher", 
     "50000", 3500.0, 5, 6, 12.0, 3, "Personal Loan", 10.0, 
     "4", "10.2", 4.0, "Standard", 1500.0, 50.0, "5 Years", "NM", 200.0, "150", 
     "Medium_spent_Medium_value_payments", "300.0"],
     
    # Baris 3: Profil Bersih/Ideal (Kemungkinan "Good")
    [2, "ID_003", "CUST_003", "January", "Alice Smith", "29", "123-47-XXX", "Engineer", 
     "120000", 8500.0, 2, 3, 5.0, 1, "Home Equity Loan", 1.0, 
     "0", "2.1", 1.0, "Good", 200.0, 15.2, "12 Years", "No", 100.0, "500", 
     "High_spent_Medium_value_payments", "1500.0"]
]

payload = {"instances": mock_instances}
request_body = json.dumps(payload)

print("--- Simulasi Request JSON Masuk ---")

df_input = input_fn(request_body, "application/json")
prediction_output = predict_fn(df_input, context)
final_response, content_type = output_fn(prediction_output, "application/json")

parsed_response = json.loads(final_response)
print(json.dumps(parsed_response, indent=4))

print("\n--- Ringkasan Hasil Test Case ---")
for idx, label in enumerate(parsed_response["labels"]):
    print(f"Data Kasus ke-{idx+1} menghasilkan Prediksi Kelas: {label}")