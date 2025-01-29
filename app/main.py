# app/main.py
import streamlit as st
from app.pages import home, disease_predictor

PAGES = {
    "Home": home,
    "Disease Predictor": disease_predictor
}

def main():
    st.set_page_config(page_title="Disease Prediction System", layout="wide")
    
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))
    
    page = PAGES[selection]
    page.show()

if __name__ == "__main__":
    main()