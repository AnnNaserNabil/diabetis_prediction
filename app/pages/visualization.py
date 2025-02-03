import streamlit as st
from src.data_visualization import visualizer

def app():
    st.title("Data Visualization")
    st.write("This page will contain data visualizations.")

    # Example functionality to visualize data
    data_file = st.file_uploader("Upload your data file", type=["csv"])
    if data_file is not None:
        data = visualizer.load_data(data_file)
        visualizer.visualize_data(data)