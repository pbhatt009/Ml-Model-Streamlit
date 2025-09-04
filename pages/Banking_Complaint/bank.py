import joblib
import streamlit as st
import numpy as np
import pandas as pd
import pathlib

# Allow Windows to load Linux paths
pathlib.PosixPath = pathlib.WindowsPath  

def bank_fn():  
    @st.cache_resource()
    def load_model_6():
        try:
            return joblib.load("pages/Banking_Complaint/model2.pkl")
        except FileNotFoundError:
            st.error("❌ 6-Class model file not found.")
            return None

    @st.cache_resource()
    def load_model_3():
        try:
            return joblib.load("pages/Banking_Complaint/model1.pkl")
        except FileNotFoundError:
            st.error("❌ 3-Class model file not found.")
            return None

    model_6 = load_model_6()
    model_3 = load_model_3()

    classes_6 = ['Banking and Payments', 'Credit Card', 'Credit Reporting', 'Debt collection', 'Loan', 'Mortgage']
    classes_3 = ["Banking & Payments", "Credit & Debt", "Loans & Mortgages"]

    st.title("🏦 Banking Complaint Classifier")

    # Layout: Radio + Notebook Buttons in same row
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        model_choice = st.radio("Select Model:", ("6-Class Model", "3-Class Model"))
    with col2:
        st.markdown("[📓 Notebook - 6 classes model](https://www.kaggle.com/code/pbhatt009/downsampling-the-bank-data-model-2)")
    with col3:
        st.markdown("[📓 Notebook - 3 classes model](https://www.kaggle.com/code/pbhatt009/downsampling-the-bank-data)")

    # Show classes according to model selection
    st.subheader("📌 Classes for Selected Model")
    if model_choice == "6-Class Model":
        st.write(", ".join(classes_6))
    else:
        st.write(", ".join(classes_3))

    complaint_text = st.text_area("✍️ Enter the complaint text here:")
    
    if st.button("🚀 Classify"):
        if complaint_text.strip() == "":
            st.warning("⚠️ Please enter a complaint before classification.")
        else:
            model = model_6 if model_choice == "6-Class Model" else model_3
            class_names = classes_6 if model_choice == "6-Class Model" else classes_3

            if model is None:
                st.error("🚨 Model not loaded yet.")
            else:
                probs = model.predict_proba([complaint_text])[0]
                pred_class = class_names[np.argmax(probs)]
              
                st.success(f"**Predicted Class:** {pred_class}")
                df = pd.DataFrame({"Class": class_names, "Confidence (%)": probs * 100})
                st.bar_chart(df.set_index("Class"))

                for cls, p in zip(class_names, probs):
                    st.write(f"{cls}: {p*100:.2f}%")
