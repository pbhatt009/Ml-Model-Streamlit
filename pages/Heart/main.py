import streamlit as st
import pickle
import pandas as pd
def Heart():
    @st.cache_data()
    def load_model():
        with open("pages/Heart/model.pkl", "rb") as f:
            return pickle.load(f)

    model = load_model()

    def get_user_input():
        st.title("🫀 Heart Disease Risk Predictor")

        st.subheader("Enter Patient Details")

        age = st.number_input("Age", 20, 100, 30)
        sex = st.selectbox("Sex", ["Male", "Female"])

        cp_display = ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic']
        cp = st.selectbox("Chest Pain Type", cp_display)
        cp_map = {name: i for i, name in enumerate(cp_display)}

        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
        chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 400, 200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
        restecg_display = ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"]
        restecg = st.selectbox("Resting ECG Results", restecg_display)
        restecg_map = {name: i for i, name in enumerate(restecg_display)}

        thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
        exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
        oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.5, 1.0, 0.1)

        slope_display = ["Upsloping", "Flat", "Downsloping"]
        slope = st.selectbox("Slope of ST Segment", slope_display)
        slope_map = {name: i for i, name in enumerate(slope_display)}

        ca = st.selectbox("Number of Major Vessels Colored", [0, 1, 2, 3, 4])

        thal_display = ["Normal", "Fixed Defect", "Reversible Defect"]
        thal = st.selectbox("Thalassemia", thal_display)
        thal_map = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}

        # Convert all to numerical values
        input_df = pd.DataFrame([{
            "age": age,
            "sex": 1 if sex == "Male" else 0,
            "cp": cp_map[cp],
            "trestbps": trestbps,
            "chol": chol,
            "fbs": 1 if fbs == "Yes" else 0,
            "restecg": restecg_map[restecg],
            "thalach": thalach,
            "exang": 1 if exang == "Yes" else 0,
            "oldpeak": oldpeak,
            "slope": slope_map[slope],
            "ca": ca,
            "thal": thal_map[thal]
        }])

        return input_df

    # Predict and display result
    user_input = get_user_input()

    if st.button("Predict"):
        prediction = model.predict(user_input)[0]
        proba = model.predict_proba(user_input)[0] if hasattr(model, "predict_proba") else None

      
        st.subheader("🩺 Prediction Result:")
        
        if proba is not None:
            risk_score = proba[1]  # probability for class 1
            st.write(f"🧮 **Probability of Heart Disease:** `{risk_score:.2f}`")

            if risk_score < 0.3:
                st.success("🟢 **Low Risk** — No major concern, stay healthy!")
            elif risk_score < 0.6:
                st.warning("🟠 **Moderate Risk** — Consider regular checkups and monitoring.")
            else:
                st.error("🔴 **High Risk** — Immediate medical attention is recommended!")

        
        else:
            if prediction == 1:
                st.error("⚠️ High Risk of Heart Disease Detected!")
            else:
                st.success("✅ Low Risk — No Immediate Concern")
        

    
