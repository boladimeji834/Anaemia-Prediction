import joblib
import streamlit as st
import pandas as pd

# to load the randomforest pipeline with all the preprocessing steps 
model = joblib.load('../models/anaemia_pipeline.joblib')  


# performance metrics
model_performance = {
    'Accuracy': 0.99,  
    'Precision': 0.97,
    'Recall': 1.0,
    'F1 Score': 0.99,
}

st.title("Anaemia Prediction App")

# My little sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ("Home", "Model Performance", "Make a Prediction"))

if page == "Home":
    st.header("Welcome to the Anaemia Prediction App")
    st.write("Use the sidebar to navigate through the app")
    st.write("Select 'Model Performance' to view the performance metrics of the model")
    st.write("Select 'Make a Prediction' to make a prediction for anaemia")
    st.success("Made by Oladimeji")


elif page == "Model Performance":
    st.header("Model Performance Metrics")
    # displaying the performance metrics 
    for metric, value in model_performance.items():
        st.write(f"**{metric}:** {value}")

elif page == "Make a Prediction":
    st.header("Make a Prediction")

    # input fields for prediction
    sex = st.selectbox("Sex", options=["Male", "Female"])
    red_pixel = st.number_input("Red Pixel (%)", min_value=0.0)
    green_pixel = st.number_input("Green Pixel (%)", min_value=0.0)
    blue_pixel = st.number_input("Blue Pixel (%)", min_value=0.0)
    hb = st.number_input("Hemoglobin (Hb)", min_value=0.0)

    if st.button("Predict"):
        # this logic preprocess the data as needed
        sex_value = "M" if sex == 'Male' else "F"  # encoding for the sex input

        # Dataframe for prediction
        features = pd.DataFrame([{
            '%Red Pixel': red_pixel,
            '%Green pixel': green_pixel,
            '%Blue pixel': blue_pixel,
            'Hb': hb,
            'Sex': sex_value
        }])

        
        features['%Red Pixel'] = features['%Red Pixel'].astype(float)
        features['%Green pixel'] = features['%Green pixel'].astype(float)
        features['%Blue pixel'] = features['%Blue pixel'].astype(float)
        features['Hb'] = features['Hb'].astype(float)
        features['Sex'] = features['Sex'].astype(str)

        # check for NaN values in input from frontend
        if features.isnull().values.any():
            st.error("Input data contains NaN values. Please check your inputs.")
        else:
            # make prediction
            prediction = model.predict(features)
            
            # display prediction result
            prediction_str = "Anaemic" if prediction[0] == 1 else "Not Anaemic"
            st.success(f"Prediction: {prediction_str}")



