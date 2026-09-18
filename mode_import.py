import pandas as pd
import joblib
model = joblib.load(r"C:\python\hospital_readmission_model.pkl")    

new_patient = pd.DataFrame({
    "age": ["70-80"],
    "diag_1": ["428"],
    "diag_2": ["250"],
    "diag_3": ["401"],
    "number_outpatient": [2],
    "number_emergency": [10],
    "number_inpatient": [7]
})

probability = model.predict_proba(new_patient)[0, 1]

print("30-day readmission probability:", f"{probability:.2%}")