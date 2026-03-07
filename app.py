import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("taxi_fare_model.pkl")

st.set_page_config(page_title="NYC Taxi Fare Predictor", layout="centered")

st.title("🚕 NYC Taxi Fare Prediction App")
st.write("Predict taxi fare using trip metadata and weather conditions")

st.divider()

# -------------------------
# Trip Information
# -------------------------

st.subheader("Trip Information")

col1, col2 = st.columns(2)

with col1:
    hour = st.slider("Transaction Hour", 0, 23, 12)
    weekday = st.selectbox(
        "Day of Week",
        ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    )
    month = st.slider("Month", 1, 12, 6)

with col2:
    weekend = st.selectbox("Weekend", [0,1])
    holiday = st.selectbox("Holiday", [0,1])
    borough = st.selectbox(
        "Borough",
        ["Manhattan","Brooklyn","Queens","Bronx","Staten Island"]
    )

st.divider()

# -------------------------
# Weather Information
# -------------------------

st.subheader("Weather Conditions")

col3, col4 = st.columns(2)

with col3:
    tavg = st.number_input("Average Temperature (°F)", value=60.0)
    precipitation = st.number_input("Precipitation", value=0.0)

with col4:
    snow_depth = st.number_input("Snow Depth", value=0.0)
    new_snow = st.number_input("New Snow", value=0.0)

st.divider()

# Convert weekday to numeric
weekday_map = {
    "Monday":0,"Tuesday":1,"Wednesday":2,
    "Thursday":3,"Friday":4,"Saturday":5,"Sunday":6
}

weekday_num = weekday_map[weekday]

# Create dataframe
input_data = pd.DataFrame({
    "transaction_hour":[hour],
    "transaction_week_day":[weekday_num],
    "transaction_month":[month],
    "weekend":[weekend],
    "is_holiday":[holiday],
    "tavg":[tavg],
    "precipitation":[precipitation],
    "snow_depth":[snow_depth],
    "new_snow":[new_snow]
})

# -------------------------
# Prediction
# -------------------------

if st.button("Predict Taxi Fare"):
    prediction = model.predict(input_data)

    st.success(f"💰 Estimated Taxi Fare: **${prediction[0]:.2f}**")

    st.info("Prediction generated using trained Machine Learning model.")
