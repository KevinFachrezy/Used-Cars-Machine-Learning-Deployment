import os
import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.title("🚗 Syarah - Used Car Price Predictor")

# Patch for pandas StringDtype version mismatch
import pandas.core.arrays.string_ as _str

_original_new  = _str.StringDtype.__new__
_original_init = _str.StringDtype.__init__

def _patched_new(cls, *args, **kwargs):
    return _original_new(cls)

def _patched_init(self, *args, **kwargs):
    try:
        _original_init(self)
    except Exception:
        pass

def _patched_setstate(self, state):
    pass

_str.StringDtype.__new__      = _patched_new
_str.StringDtype.__init__     = _patched_init
_str.StringDtype.__setstate__ = _patched_setstate

# Load model — looks in the same folder as this script
try:
    model_path = os.path.join(os.path.dirname(__file__), 'Used_cars_XGB.sav')
    model = joblib.load(model_path)
    st.success("Model loaded!")
except Exception as e:
    st.error(f"Cannot load model: {e}")
    st.stop()

# ── Inputs ──────────────────────────────────────────────────────────────────
make = st.selectbox("Car Make", [
    "Toyota","Honda","BMW","Mercedes","Lexus","Land Rover","GMC","Jeep",
    "Nissan","Hyundai","Chevrolet","Ford","Kia","Dodge","Mazda",
    "Mitsubishi","Audi","Porsche","Cadillac","Infiniti"
])

car_type = st.text_input("Car Type (e.g. Camry, Patrol, X5)", "Camry")

region = st.selectbox("Region", [
    "Riyadh","Jeddah","Dammam","Al-Medina","Qassim","Makkah","Jazan",
    "Tabouk","Aseer","Al-Ahsa","Taef","Sabya","Al-Baha","Khobar","Yanbu",
    "Hail","Al-Namas","Jubail","Al-Jouf","Abha","Hafar Al-Batin","Najran",
    "Arar","Besha","Qurayyat","Wadi Dawasir","Sakaka"
])

origin      = st.selectbox("Origin", ["Saudi", "Gulf Arabic", "Other"])
year        = st.slider("Year", 2000, 2025, 2018)
mileage     = st.number_input("Mileage (km)", 0, 600000, 80000, 5000)
engine_size = st.number_input("Engine Size (L)", 1.0, 9.0, 2.0, 0.5)
gear_type   = st.selectbox("Gear Type", ["Automatic", "Manual"])
fuel_type   = st.selectbox("Fuel Type", ["Gas", "Diesel", "Hybrid"])
car_options = st.selectbox("Car Options", ["Full", "Standard", "Semi Full"])
color       = st.selectbox("Color", [
    "Black","Silver","Grey","Navy","White","Bronze","Another Color",
    "Golden","Brown","Blue","Red","Oily","Green","Orange","Yellow"
])

# ── Predict ─────────────────────────────────────────────────────────────────
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

    # Confidence range based on model MAPE of 16.96%
    lower = price * (1 - 0.1696)
    upper = price * (1 + 0.1696)

    st.markdown(f"## 💰 Estimated Price: SAR {price:,.0f}")
    st.markdown(f"#### 📊 Likely Range: SAR {lower:,.0f} — SAR {upper:,.0f}")
    st.caption("Range based on model MAPE of 16.96%")