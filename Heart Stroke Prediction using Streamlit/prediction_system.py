import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE


def stroke_prediction_app(df):

    st.title("🩺 Heart Stroke Prediction App")

    # **Data Preprocessing**
    df = df.drop(columns=['id'])  # Remove ID column
    df = df[df['gender'] != "Other"]  # Remove 'Other' gender
    df['bmi'].fillna(df['bmi'].mean(), inplace=True)  # Fill missing BMI values

    # Label Encoding
    label_encoder = LabelEncoder()
    df['work_type'] = label_encoder.fit_transform(df['work_type'])
    df['Residence_type'] = label_encoder.fit_transform(df['Residence_type'])
    df['smoking_status'] = label_encoder.fit_transform(df['smoking_status'])
    df['ever_married'] = label_encoder.fit_transform(df['ever_married'])
    df['gender'] = label_encoder.fit_transform(df['gender'])

    # Splitting Features & Target
    X = df.drop(columns=['stroke'])
    y = df['stroke']

    # Handling Imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Splitting Data
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=3)

    # Feature Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train Model
    logistic_model = LogisticRegression(random_state=3, max_iter=1000, class_weight='balanced')
    logistic_model.fit(X_train, y_train)

    # Save Model & Scaler
    joblib.dump(logistic_model, "stroke_model.pkl")
    joblib.dump(scaler, "scaler.pkl")

    # **User Input for Prediction**
    st.header("📝 Enter User Data")
    age = st.slider("Age", 1, 100, 50)
    hypertension = st.selectbox("Hypertension", [0, 1])
    heart_disease = st.selectbox("Heart Disease", [0, 1])
    avg_glucose_level = st.number_input("Average Glucose Level", value=120.0)
    bmi = st.number_input("BMI", value=25.0)
    gender = st.selectbox("Gender", ["Male", "Female"])
    ever_married = st.selectbox("Ever Married", ["Yes", "No"])
    work_type = st.selectbox("Work Type",  ["Private","Self-employed","Govt_job","Children","Never_worked"])
    residence_type = st.selectbox("Residence Type",["Urban","Rural"])
    smoking_status = st.selectbox("Smoking Status", ["Never smoked","Formerly smoked","Smokes","Unknown"])

    # Encoding User Input
    gender_encoded= 1 if gender == "Male" else 0
    ever_married_encoded = 1 if ever_married == "Yes" else 0

    # Encoding work_type
    if work_type == "Private":
        work_type_encoded = 0
    elif work_type == "Self-employed":
        work_type_encoded = 1
    elif work_type == "Govt_job":
        work_type_encoded = 2
    elif work_type == "Children":
        work_type_encoded = 3
    else:  # "Never_worked"
        work_type_encoded = 4

    # Encoding residence_type
    residence_type_encoded = 1 if residence_type == "Urban" else 0

    # Encoding smoking_status
    if smoking_status == "Never smoked":
        smoking_status_encoded = 0
    elif smoking_status == "Formerly smoked":
        smoking_status_encoded = 1
    elif smoking_status == "Smokes":
        smoking_status_encoded = 2
    else:  # "Unknown"
        smoking_status_encoded = 3


    # Create user input array
    user_data = np.array([gender_encoded, age, hypertension, heart_disease, ever_married_encoded, work_type_encoded, residence_type_encoded, avg_glucose_level, bmi, smoking_status_encoded]).reshape(1, -1)

    if st.button("Predict Stroke Risk"):
        if os.path.exists("stroke_model.pkl") and os.path.exists("scaler.pkl"):
            loaded_model = joblib.load("stroke_model.pkl")
            loaded_scaler = joblib.load("scaler.pkl")

            # Scale Input
            user_input_scaled = loaded_scaler.transform(user_data)

            # Make Prediction
            probability = loaded_model.predict_proba(user_input_scaled)[0][1]
            prediction = (probability >= 0.5).astype(int)  # Custom threshold at 0.5

            # Display Results
            st.subheader("Prediction Results:")
            if prediction == 1:
                st.error(f"High Risk of Stroke! Probability: {probability:.2f}")
            else:
                st.success(f"Low Risk of Stroke. Probability: {probability:.2f}")
        else:
            st.warning("Model and Scaler files not found! Train the model first.")
