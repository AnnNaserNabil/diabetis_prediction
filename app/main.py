import streamlit as st
from app.pages import home, data_analysis, disease_predictor, visualization

PAGES = {
    "Home": home,
    "Data Analysis": data_analysis,
    "Disease Predictor": disease_predictor,
    "Visualization": visualization
}

def main():
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))
    page = PAGES[selection]
    page.app()

if __name__ == "__main__":
    main()