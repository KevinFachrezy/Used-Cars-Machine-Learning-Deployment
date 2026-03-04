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
    
# Make Type Mapping
MAKE_TYPE = {
    "Aston Martin": ["DB9", "Vanquish", "Vantage"],
    "Audi": ["A3", "A4", "A5", "A6", "A7", "A8", "Q5", "Q7", "S5", "S8"],
    "BMW": ["The 3", "The 4", "The 5", "The 6", "The 7", "The M", "X", "Z"],
    "BYD": ["F3"],
    "Bentley": ["Arnage", "Bentayga", "Flying Spur"],
    "Cadillac": ["ATS", "CT-S", "CT4", "CT5", "CT6", "DTS", "Escalade", "Fleetwood", "XT5"],
    "Changan": ["CS35", "CS35 Plus", "CS75", "CS85", "CS95", "Eado", "Seven"],
    "Chery": ["QQ", "Tiggo"],
    "Chevrolet": ["Abeka", "Aveo", "Blazer", "Camaro", "Caprice", "Colorado", "Cruze",
                  "Impala", "Kaptiva", "Lumina", "Malibu", "Optra", "Pick up",
                  "Silverado", "Spark", "Suburban", "Tahoe", "Trailblazer", "Traverse"],
    "Chrysler": ["300", "C200", "C300", "S300", "SRT"],
    "Daihatsu": ["Delta", "Gran Max", "Terios", "Terios Ground"],
    "Dodge": ["Challenger", "Charger", "Durango", "Nitro", "Ram"],
    "FAW": ["B50", "T77", "X40"],
    "Ferrari": ["GTB 599 Fiorano"],
    "Fiat": ["500", "Doblo"],
    "Ford": ["Echo Sport", "Edge", "Expedition", "Explorer", "F150", "Flex",
             "Focus", "Fusion", "Marquis", "Mustang", "Ranger", "Taurus",
             "Van", "Vego", "Victoria"],
    "Foton": ["Mini Van", "Suvana"],
    "GAC": ["GS3", "GS8"],
    "GMC": ["Acadia", "Behbehani", "Envoy", "Safari", "Savana", "Sierra",
            "Suburban", "Terrain", "Yukon"],
    "Geely": ["Azkarra", "Coolray", "EC7", "EC8", "Emgrand", "GC7", "GS", "X7"],
    "Genesis": ["Coupe", "G330", "G70", "G80", "Platinum", "Prestige", "Prestige Plus", "Royal"],
    "Great Wall": ["Power"],
    "HAVAL": ["H2", "H6", "H9"],
    "Honda": ["Accord", "CRV", "City", "Civic", "Crosstour", "HRV", "Odyssey", "Pilot"],
    "Hummer": ["H-2", "H3"],
    "Hyundai": ["Accent", "Avante", "Azera", "Bus County", "Coupe", "Coupe S",
                "Creta", "Elantra", "Genesis", "H1", "Kona", "Senta fe",
                "Sonata", "Tucson", "Tuscani", "Veloster", "Veracruz", "i40"],
    "INFINITI": ["FX", "M", "Q", "QX"],
    "Isuzu": ["D-MAX", "Dyna"],
    "Iveco": ["Daily"],
    "Jaguar": ["F Type", "F-Pace", "XF", "XJ"],
    "Jeep": ["Cherokee", "Compass", "Grand Cherokee", "Liberty", "Patriot", "Wrangler"],
    "Kia": ["Cadenza", "Carens", "Carenz", "Carnival", "Cerato", "Cores", "K5",
            "Mohave", "Opirus", "Optima", "Pegas", "Picanto", "Rio", "Sedona",
            "Seltos", "Sorento", "Soul", "Sportage", "Stinger"],
    "Land Rover": ["Defender", "Discovery", "Range Rover"],
    "Lexus": ["ES", "GS", "GX", "IS", "LS", "LX", "NX", "RC", "RX", "UX"],
    "Lifan": ["LF X60"],
    "Lincoln": ["Continental GT", "MKS", "MKX", "MKZ", "Navigator"],
    "MG": ["3", "360", "5", "6", "GS", "HS", "RX5", "RX8", "ZS"],
    "MINI": ["Copper", "Countryman"],
    "Maserati": ["Gamble", "Levante", "Quattroporte"],
    "Mazda": ["2", "3", "6", "CX3", "CX5", "CX7", "CX9"],
    "Mercedes": ["A", "C", "CL", "CLA", "CLS", "E", "G", "GL", "GLC", "GLE",
                 "ML", "POS24", "S", "SEL", "SL", "Viano"],
    "Mercury": ["Grand Marquis", "Milan", "Montero2"],
    "Mitsubishi": ["ASX", "Attrage", "Galant", "L200", "L300", "Lancer",
                   "Montero", "Nativa", "Outlander", "Pajero"],
    "Nissan": ["Altima", "Armada", "Bus Urvan", "Datsun", "Gloria", "Juke",
               "KICKS", "Land Cruiser Pickup", "Maxima", "Murano", "Navara",
               "Pathfinder", "Patrol", "Sentra", "Sunny", "Sylvian Bus",
               "VTC", "X-Terra", "X-Trail", "Z350", "Z370"],
    "Peugeot": ["3008", "301", "307", "5008", "Boxer", "Partner"],
    "Porsche": ["911", "Cayenne", "Cayenne S", "Cayenne Turbo",
                "Cayenne Turbo GTS", "Cayman", "Macan", "Panamera"],
    "Renault": ["Capture", "Dokker", "Duster", "Fluence", "Koleos",
                "Logan", "Megane", "Safrane", "Symbol", "Talisman"],
    "Rolls-Royce": ["Camargue", "Ghost"],
    "Subaru": ["Forester"],
    "Suzuki": ["APV", "D'max", "Dzire", "Ertiga", "Grand Vitara", "Jimny", "SX4", "Vitara"],
    "Toyota": ["4Runner", "Aurion", "Avalon", "Avanza", "C-HR", "Camry",
               "Ciocca", "Coaster", "Corolla", "Cressida", "Echo", "FJ",
               "Furniture", "Hiace", "Hilux", "Innova", "Land Cruiser",
               "Land Cruiser 70", "Land Cruiser Pickup", "Prado", "Previa",
               "Prius", "Rav4", "Rush", "Yaris"],
    "Victory Auto": ["Van R"],
    "Volkswagen": ["Beetle", "CC", "Golf", "Jetta", "Passat", "Tiguan", "Touareg"],
    "Zhengzhou": ["Pick up"],
    "Škoda": ["Fabia", "Superb"],
}

# ── Inputs ──────────────────────────────────────────────────────────────────
make = st.selectbox("Brand (Make)", options=sorted(MAKE_TYPE.keys()))

car_type = st.text_input("Car Type (e.g. Camry, Patrol, X5)", "Camry")

region = st.selectbox("Model (Type)", options=MAKE_TYPE.get(make, []))

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