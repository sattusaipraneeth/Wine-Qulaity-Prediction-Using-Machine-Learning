import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Write the title and description
st.write("""
# Wine Quality Prediction App

This app predicts the quality of wine based on various parameters.
""")

# Sidebar for user input
st.sidebar.header('User Input Parameters')
st.sidebar.markdown("Please adjust the sliders to input the desired values.")

def user_input_features():
    fixed_acidity = st.sidebar.slider('Fixed Acidity', 4.6, 15.9, 8.31)
    volatile_acidity = st.sidebar.slider('Volatile Acidity', 0.12, 1.58, 0.52)
    citric_acid = st.sidebar.slider('Citric Acid', 0.0, 1.0, 0.5)
    residual_sugar = st.sidebar.slider('Residual Sugar', 0.9, 32.8, 6.4)
    chlorides = st.sidebar.slider('Chlorides', 0.01, 0.6, 0.08)
    free_sulfur_dioxide = st.sidebar.slider('Free Sulfur Dioxide', 1.0, 72.0, 16.0)
    total_sulfur_dioxide = st.sidebar.slider('Total Sulfur Dioxide', 6.0, 289.0, 46.0)
    density = st.sidebar.slider('Density', 0.99, 1.00, 0.99)
    pH = st.sidebar.slider('pH', 2.72, 4.0, 3.2)
    sulphates = st.sidebar.slider('Sulphates', 0.33, 2.0, 0.65)
    alcohol = st.sidebar.slider('Alcohol', 8.4, 14.9, 10.4)
    best_quality = st.sidebar.slider('Best Quality', 0, 10, 5)  # Adjust range as needed

    # Create a DataFrame with the correct feature names
    data = {
        'fixed acidity': fixed_acidity,
        'volatile acidity': volatile_acidity,
        'citric acid': citric_acid,
        'residual sugar': residual_sugar,
        'chlorides': chlorides,
        'free sulfur dioxide': free_sulfur_dioxide,
        'total sulfur dioxide': total_sulfur_dioxide,
        'density': density,
        'pH': pH,
        'sulphates': sulphates,
        'alcohol': alcohol,
        'best quality': best_quality
    }

    features = pd.DataFrame(data, index=[0])
    return features

# Get user input features
df = user_input_features()

# Display user input
st.subheader('User Input Parameters')
st.write(df)

# Load the pickled model
with open(r"C:\Users\saipr\OneDrive\Desktop\SAIPRANEETH S\Wine Quality prediction\Model\wine_quality_model.pkl", 'rb') as f:
    pickled_model = pickle.load(f)

# Prediction
prediction = pickled_model.predict(df)
prediction_proba = pickled_model.predict_proba(df)

# Show class labels and their corresponding index numbers
st.subheader('Class Labels and Their Corresponding Index Number')
st.write(pd.DataFrame({'Wine Quality': [3, 4, 5, 6, 7, 8]}))

# Display the prediction
st.subheader('Prediction')
st.write(f"Predicted Wine Quality: **{prediction[0]}**")

# Display the prediction probabilities
class_labels = pd.DataFrame({'Quality': [3, 4, 5, 6, 7, 8]})
class_probabilities = pd.DataFrame(prediction_proba[0].T, columns=['Probability'])
st.subheader('Prediction Probability')
st.write(pd.concat([class_labels, class_probabilities], axis=1))