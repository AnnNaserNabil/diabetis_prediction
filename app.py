import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load the data using the updated caching function
@st.cache_data
def load_data():
    try:
        train_data = pd.read_csv('data/Training.csv')
        test_data = pd.read_csv('data/Testing.csv')
        return train_data, test_data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

train_data, test_data = load_data()

# Check if data is loaded successfully
if train_data is None or test_data is None:
    st.error("Failed to load data. Please check the file paths and ensure the CSV files are present.")
else:
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Data Analysis", "Data Visualization", "Model Training"])

    if page == "Data Analysis":
        st.title("Data Analysis")

        st.write("### Training Data")
        st.write(train_data.head())

        st.write("### Testing Data")
        st.write(test_data.head())

        st.write("### Summary Statistics")
        st.write(train_data.describe())

    elif page == "Data Visualization":
        st.title("Data Visualization")

        st.write("### Correlation Matrix")
        # Select only numeric columns for correlation matrix
        numeric_columns = train_data.select_dtypes(include=['number']).columns
        corr_matrix = train_data[numeric_columns].corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
        st.pyplot(fig)

        st.write("### Pairplot")
        selected_columns = st.multiselect("Select columns for pairplot", train_data.columns)
        if selected_columns:
            pairplot_fig = sns.pairplot(train_data[selected_columns])
            st.pyplot(pairplot_fig)

    elif page == "Model Training":
        st.title("Model Training")

        st.write("### Select Features and Target")
        features = st.multiselect("Select features", train_data.columns, default=train_data.columns[:-1])
        target = st.selectbox("Select target", train_data.columns, index=len(train_data.columns)-1)

        if st.button("Train Model"):
            X = train_data[features]
            y = train_data[target]
            
            # Encode target labels
            le = LabelEncoder()
            y = le.fit_transform(y)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            st.write("### Model Accuracy")
            st.write(f"Accuracy: {accuracy:.2f}")

            st.write("### Feature Importances")
            feature_importances = pd.Series(model.feature_importances_, index=features)
            st.bar_chart(feature_importances)