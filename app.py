import streamlit as st
import tensorflow as tf
import numpy as np

# Load the trained model
@st.cache_resource  # Caches the model to speed up loading
def load_model():
    return tf.keras.models.load_model("fire_area_model.keras")

# Initialize the model
model = load_model()

# App Title
st.title("Fire Area Prediction")
st.write("Enter the feature values to predict the fire-affected area.")

# Input fields for features
X = st.number_input("X Coordinate", min_value=0.0, step=0.1)
Y = st.number_input("Y Coordinate", min_value=0.0, step=0.1)
temp = st.number_input("Temperature (°C)", min_value=0.0, step=0.1)
RH = st.number_input("Relative Humidity (%)", min_value=0.0, step=0.1)
wind = st.number_input("Wind Speed (km/h)", min_value=0.0, step=0.1)
rain = st.number_input("Rainfall (mm)", min_value=0.0, step=0.1)

# Predict Button
if st.button("Predict"):
    # Prepare input data
    features = np.array([[X,Y,temp, RH, wind, rain]])  # Include all input features
    prediction = model.predict(features)[0][0]  # Make prediction

    # Display the prediction result
    st.subheader(f"Predicted Fire Area: {prediction:.2f} hectares")

