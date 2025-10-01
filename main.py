import streamlit as st
import joblib
import numpy as np
import pandas as pd
import pathlib
import sys
from Text_preprocess import TextPreprocessor

# Map __main__.TextPreprocessor to avoid joblib errors
sys.modules['__main__'].TextPreprocessor = TextPreprocessor

# Allow Windows to load Linux-path-based joblib models
pathlib.PosixPath = pathlib.WindowsPath

st.set_page_config(page_title="Banking Complaint Classifier", layout="wide")

# -----------------------------
# Load Models with Caching
# -----------------------------
@st.cache_resource
def load_model(path):
    try:
        return joblib.load(path)
    except FileNotFoundError:
        st.error(f"❌ Model file not found: {path}")
        return None

model_6 = load_model("model2.pkl")
model_3 = load_model("model1.pkl")

classes_6 = ['Banking and Payments', 'Credit Card', 'Credit Reporting', 'Debt collection', 'Loan', 'Mortgage']
classes_3 = ["Banking & Payments", "Credit & Debt", "Loans & Mortgages"]

# -----------------------------
# Sidebar: Model Selection
# -----------------------------
st.sidebar.title("🏦 Banking Complaint Classifier")
model_choice = st.sidebar.radio("Select Model:", ("6-Class Model", "3-Class Model"))

st.sidebar.markdown("**Notebook Links:**")
st.sidebar.markdown("[📓 6-Class Model Notebook](https://www.kaggle.com/code/pbhatt009/downsampling-the-bank-data-model-2)")
st.sidebar.markdown("[📓 3-Class Model Notebook](https://www.kaggle.com/code/pbhatt009/downsampling-the-bank-data)")

# Display classes
st.subheader("📌 Classes for Selected Model")
st.write(", ".join(classes_6 if model_choice == "6-Class Model" else classes_3))

# -----------------------------
# User Input
# -----------------------------
complaint_text = st.text_area("✍️ Enter the complaint text here:")

# -----------------------------
# Classification Logic
# -----------------------------
if st.button("🚀 Classify"):
    if not complaint_text.strip():
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

            # Prepare DataFrame for visualization
            df = pd.DataFrame({"Class": class_names, "Confidence": probs * 100})

            # Display Altair bar chart
            import altair as alt
            chart = alt.Chart(df).mark_bar().encode(
                x=alt.X("Class", sort=None),
                y=alt.Y("Confidence", title="Confidence (%)"),
                tooltip=["Class", alt.Tooltip("Confidence", format=".2f")]
            ).properties(width=700, height=400)
            st.altair_chart(chart, use_container_width=True)

            # Show detailed probabilities
            st.subheader("🔹 Detailed Probabilities")
            for cls, p in zip(class_names, probs):
                st.write(f"{cls}: {p*100:.2f}%")
