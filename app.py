import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("model.pkl")

st.title("🚖 NYC Taxi Fare Prediction")

st.write("Enter trip details to predict fare")

passenger_count = st.slider("Passenger Count",1,6,1)
trip_distance = st.number_input("Trip Distance (miles)",0.1,30.0,1.0)
pickup_hour = st.slider("Pickup Hour",0,23,12)
pickup_day = st.slider("Pickup Day (0=Mon,6=Sun)",0,6,2)
trip_duration = st.number_input("Trip Duration (minutes)",1,180,10)

if st.button("Predict Fare"):
    
    features = np.array([[passenger_count,
                          trip_distance,
                          pickup_hour,
                          pickup_day,
                          trip_duration]])
    
    prediction = model.predict(features)

    st.success(f"Estimated Fare: ${prediction[0]:.2f}")
