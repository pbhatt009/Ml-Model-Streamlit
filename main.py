import streamlit as st
from pages.iris.iris import show_iris
from pages.Heart.main import Heart

st.set_page_config(page_title="ML App", layout="wide")

# Initialize page state
if "page" not in st.session_state:
    st.session_state.page = "home"

# Navigation logic
def go_to(page_name):
    st.session_state.page = page_name

def back_home():
    st.session_state.page = "home"

def go_to_notebook(url):
    st.markdown(f"""
        <a href="{url}" target="_blank">
            <button style='padding: 0.5em 1em; font-size: 16px; border: 0.01rem solid; background: none;'>📘 Open Notebook</button>
        </a>
    """, unsafe_allow_html=True)


    

# --- Page: Home ---
if st.session_state.page == "home":
    st.title("🧠 Welcome to ML Model Showcase")

    st.markdown("### Choose a model to explore:")

    col1,col2= st.columns(2)

###############  IRIS DATSET #################

    with col1:
        st.button("🌸 Iris Classifier", on_click=go_to, args=("iris",))
        st.image("iris.jpeg", width=300)

    
############# Heart Deases Prediction ##########
    with col2:
        st.button("🫀 Heart Deases Prediction", on_click=go_to, args=("heart",))
        go_to_notebook("https://github.com/pbhatt009/ml-project-notebooks/blob/main/Heart-Disease/main.ipynb")
        st.image("heart.jpg", width=300)




######             Naviagtion   #############
elif st.session_state.page == "iris":
    st.button("🔙 Back", on_click=back_home)
    show_iris()

elif st.session_state.page == "heart":
    st.button("🔙 Back", on_click=back_home)
    Heart()

