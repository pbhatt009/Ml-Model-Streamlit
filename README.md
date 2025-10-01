# Banking Complaint Classifier - Streamlit App


A **Streamlit web application** for classifying banking complaints using **two pre-trained ML models**. The app provides an **interactive interface**, **visualizations**, and **quick predictions**.  

## Live: https://my-ml-model-009.streamlit.app/
---

## 📌 Project Overview

This app classifies banking complaints into **six or three categories** depending on the selected model.  

**Key Features:**  
- ✅ Two ML Models:
  - **6-class model:** `['Banking and Payments', 'Credit Card', 'Credit Reporting', 'Debt collection', 'Loan', 'Mortgage']`  
  - **3-class model:** `["Banking & Payments", "Credit & Debt", "Loans & Mortgages"]`  
- ✅ **Variance-aware downsampling** for handling large datasets  
- ✅ **Grouped similar classes** to simplify predictions  
- ✅ **Bar plot visualization** to enhance understanding of class distribution  

---

## 🗂 Project Structure


### Ml-Model-Streamlit/
### │
### ├─ .devcontainer # Dev container configuration
### ├─ pycache/ # Python cache files
### ├─ Text_preprocess.py # Text preprocessing functions
### ├─ main.py # Streamlit app entry point 
### ├─ model1.pkl # Pre-trained 6-class model
### ├─ model2.pkl # Pre-trained 3-class model
### ├─ requirements.txt # Python dependencies
### └─ README.md # Project documentation



---

## 🚀 Setup Instructions ###

 ### 1. Clone the Repository ###


git clone https://github.com/pbhatt009/Ml-Model-Streamlit.git
cd Ml-Model-Streamlit

### 2. Install Dependencies
pip install -r requirements.txt

### 3. Run the Application
streamlit run main.py


 # 🧪 Usage

1.Select a Model: Choose either the 6-class or 3-class classifier.

2.Input Complaint Text: Type or paste a banking complaint.

3.Predict: Click Predict to see the predicted category.

4.Visualize Data: Bar plots show complaint distributions for better understanding.



# 📊 Sample UI Screenshots


## Home Screen:
<img width="500" height="400" alt="Screenshot 2025-10-01 124247" src="https://github.com/user-attachments/assets/3e9d7aa7-e35b-483f-9639-5c04d0b1cad4" />

## Input and Prediction:

<img width="400" height="400" alt="Screenshot 2025-10-01 124403" src="https://github.com/user-attachments/assets/906a9bb5-a870-43f6-9215-b56e2615d2a5" />

## Bar Plot Visualization:

<img width="700" height="500" alt="Screenshot 2025-10-01 124425" src="https://github.com/user-attachments/assets/52d2e217-4045-45ed-830c-4520008f2dc0" />

<img width="300" height="300" alt="Screenshot 2025-10-01 124445" src="https://github.com/user-attachments/assets/f19397e6-64e8-4e42-8905-fd74eca8f72c" />

# 🔧 Preprocessing and Data Handling

1.Variance-aware downsampling to reduce large dataset size

2.Grouped similar classes to generate 3-class predictions

3.Text preprocessing done using Text_preprocess.py



# 📄 License

This project is licensed under the MIT License


