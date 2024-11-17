import streamlit as st
import numpy as np
import joblib

# Load saved models
model_paths = {
    "Logistic Regression": "Training\Models\Logistic_Regression.joblib",
    "Random Forest": "Training\Models\Random_Forest.joblib",
    "XGBoost": "Training\Models\XGBoost.joblib"
}
models = {name: joblib.load(path) for name, path in model_paths.items()}

# Streamlit app
st.title("Heart Attack Prediction App")
st.write("Enter the patient details to predict the risk of a heart attack using three different models.")

# Input fields for user data
age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1)
sex = st.selectbox("Sex (1 = Male, 0 = Female)", [1, 0])
cp = st.selectbox("Chest Pain Type (0 = Typical Angina, 1 = Atypical Angina, 2 = Non-anginal Pain, 3 = Asymptomatic)", [0, 1, 2, 3])
trtbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=300, value=120, step=1)
chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200, step=1)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)", [1, 0])
restecg = st.selectbox("Resting ECG (0 = Normal, 1 = ST-T wave abnormality, 2 = Left ventricular hypertrophy)", [0, 1, 2])
thalachh = st.number_input("Maximum Heart Rate Achieved", min_value=50, max_value=250, value=140, step=1)
exng = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [1, 0])
oldpeak = st.number_input("Oldpeak (ST depression induced by exercise)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
slp = st.selectbox("Slope of Peak Exercise ST Segment (0 = Upsloping, 1 = Flat, 2 = Downsloping)", [0, 1, 2])
caa = st.number_input("Number of Major Vessels (0-3) Colored by Fluoroscopy", min_value=0, max_value=3, value=0, step=1)
thall = st.selectbox("Thalassemia (0 = Null, 1 = Fixed Defect, 2 = Normal, 3 = Reversible Defect)", [0, 1, 2, 3])

# Prediction button
if st.button("Predict"):
    # Prepare the input data for the model
    input_data = np.array([[age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall]])
    
    # Make predictions using all three models
    st.subheader("Prediction Results")
    for name, model in models.items():
        prediction = model.predict(input_data)[0]
        result = "Heart Attack Risk" if prediction == 1 else "No Heart Attack Risk"
        st.write(f"{name}: **{result}**")
