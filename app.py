import os
import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.title("🚗 Syarah - Used Car Price Predictor")


# load data
@st.cache_data  
def load_data():
    csv_path = os.path.join(os.path.dirname(__file__), 'UsedCarsSaudiCleaned.csv')
    return pd.read_csv(csv_path)

df = load_data()

def sorted_unique(col):
    """Return sorted unique non-null values from a column as a list."""
    return sorted(df[col].dropna().unique().tolist())

# load model
try:
    model_path = os.path.join(os.path.dirname(__file__), 'Used_cars_XGB.sav')
    model = joblib.load(model_path)
    st.success("Model loaded!")
except Exception as e:
    st.error(f"Cannot load model: {e}")
    st.stop()

# Input form

# select make untuk filter type berdasarkan make yang dipilih
make = st.selectbox("Car Make", sorted_unique("Make"))

types_for_make = sorted(df[df["Make"] == make]["Type"].dropna().unique().tolist())
car_type = st.selectbox("Car Type", types_for_make)

region      = st.selectbox("Region",    sorted_unique("Region"))
origin      = st.selectbox("Origin",    sorted_unique("Origin"))
color       = st.selectbox("Color",     sorted_unique("Color"))
car_options = st.selectbox("Options",   sorted_unique("Options"))
fuel_type   = st.selectbox("Fuel Type", sorted_unique("Fuel_Type"))
gear_type   = st.selectbox("Gear Type", sorted_unique("Gear_Type"))

year = st.slider(
    "Year",
    int(df["Year"].min()),
    int(df["Year"].max()),
    2018
)

mileage = st.number_input(
    "Mileage (km)",
    min_value=int(df["Mileage"].min()),
    max_value=int(df["Mileage"].max()),
    value=80000,
    step=5000
)

engine_size = st.number_input(
    "Engine Size (L)",
    min_value=float(df["Engine_Size"].min()),
    max_value=float(df["Engine_Size"].max()),
    value=2.0,
    step=0.1
)

# Predict
if st.button("Predict Price"):
    car_age = pd.Timestamp.now().year - year

    input_df = pd.DataFrame([{
        'Make'       : make,
        'Type'       : car_type,
        'Engine_Size': engine_size,
        'Mileage'    : mileage,
        'Car_Age'    : car_age,
        'Gear_Type'  : gear_type,
        'Fuel_Type'  : fuel_type,
        'Options'    : car_options,
        'Region'     : region,
        'Origin'     : origin,
        'Color'      : color
    }])

    price = model.predict(input_df)[0]

    lower = price * (1 - 0.1696)
    upper = price * (1 + 0.1696)

    st.markdown(f"## 💰 Estimated Price: SAR {price:,.0f}")
    st.markdown(f"#### 📊 Likely Range: SAR {lower:,.0f} — SAR {upper:,.0f}")
    st.caption("Range based on model MAPE of 16.96%")