import streamlit as st
import pandas as pd
import numpy as np
import pickle                                           # ← pickle only, no joblib
from datetime import datetime

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Saudi Used Car Price Predictor",
    page_icon="🚗",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .stApp { background: linear-gradient(135deg, #0f0f0f 0%, #1a1a2e 100%); }

    h1, h2, h3 {
        font-family: 'Syne', sans-serif !important;
        font-weight: 800 !important;
    }

    .hero-title {
        font-family: 'Syne', sans-serif;
        font-size: 3rem;
        font-weight: 800;
        color: #ffffff;
        line-height: 1.1;
        margin-bottom: 0.3rem;
    }

    .hero-subtitle {
        font-family: 'DM Sans', sans-serif;
        font-size: 1rem;
        color: #888888;
        margin-bottom: 2rem;
    }

    .accent { color: #e8c547; }

    .price-card {
        background: linear-gradient(135deg, #1e1e1e, #252525);
        border: 1px solid #e8c547;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin-top: 1.5rem;
        box-shadow: 0 0 40px rgba(232, 197, 71, 0.15);
    }

    .price-label {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.85rem;
        color: #888888;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 0.5rem;
    }

    .price-value {
        font-family: 'Syne', sans-serif;
        font-size: 3.2rem;
        font-weight: 800;
        color: #e8c547;
        line-height: 1;
    }

    .price-currency {
        font-size: 1.2rem;
        color: #aaaaaa;
        margin-top: 0.3rem;
        font-family: 'DM Sans', sans-serif;
    }

    .range-row {
        display: flex;
        justify-content: space-between;
        margin-top: 1.2rem;
        padding-top: 1.2rem;
        border-top: 1px solid #333;
    }

    .range-item { text-align: center; flex: 1; }

    .range-label {
        font-size: 0.75rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }

    .range-val {
        font-size: 1.1rem;
        font-weight: 600;
        color: #cccccc;
        font-family: 'Syne', sans-serif;
    }

    .section-header {
        font-family: 'Syne', sans-serif;
        font-size: 0.75rem;
        font-weight: 700;
        color: #e8c547;
        text-transform: uppercase;
        letter-spacing: 3px;
        margin-top: 2rem;
        margin-bottom: 0.8rem;
    }

    div[data-testid="stSelectbox"] label,
    div[data-testid="stNumberInput"] label,
    div[data-testid="stSlider"] label {
        color: #aaaaaa !important;
        font-size: 0.85rem !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    div[data-testid="stSelectbox"] > div > div,
    div[data-testid="stNumberInput"] input {
        background-color: #1e1e1e !important;
        border-color: #333 !important;
        color: #ffffff !important;
        border-radius: 8px !important;
    }

    div.stButton > button {
        background: #e8c547;
        color: #0f0f0f;
        font-family: 'Syne', sans-serif;
        font-weight: 700;
        font-size: 1rem;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        width: 100%;
        margin-top: 1rem;
        transition: all 0.2s;
        cursor: pointer;
    }

    div.stButton > button:hover {
        background: #f5d565;
        transform: translateY(-1px);
        box-shadow: 0 4px 20px rgba(232, 197, 71, 0.4);
    }

    .disclaimer {
        font-size: 0.75rem;
        color: #555;
        text-align: center;
        margin-top: 1rem;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)

# ── Make-Type Mapping ─────────────────────────────────────────────────────────
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

# ── Load Model ────────────────────────────────────────────────────────────────
with open("model/Used_cars_XGB.pkl", "rb") as f:

    model = pickle.load(f)
    

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">Saudi Used Car<br><span class="accent">Price Estimator</span></div>',
            unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Enter your car details below to get an instant market price estimate powered by machine learning.</div>',
            unsafe_allow_html=True)

if model is None:
    st.error("⚠️ Model file `Used_cars_XGB.pkl` not found. Place the file in the same directory as this app.")
    st.stop()

# ── Form ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Vehicle Identity</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    make = st.selectbox("Brand (Make)", options=sorted(MAKE_TYPE.keys()))
with col2:
    car_type = st.selectbox("Model (Type)", options=MAKE_TYPE.get(make, []))

col3, col4 = st.columns(2)
with col3:
    year = st.selectbox("Year", options=list(range(2021, 1963, -1)))
with col4:
    engine_size = st.number_input("Engine Size (L)", min_value=1.0, max_value=9.0,
                                   value=2.0, step=0.1, format="%.1f")

st.markdown('<div class="section-header">Condition & Usage</div>', unsafe_allow_html=True)

col5, col6 = st.columns(2)
with col5:
    mileage = st.number_input("Mileage (km)", min_value=0, max_value=500000,
                               value=50000, step=1000)
with col6:
    gear_type = st.selectbox("Transmission", options=["Automatic", "Manual"])

col7, col8 = st.columns(2)
with col7:
    fuel_type = st.selectbox("Fuel Type", options=["Gas", "Diesel", "Hybrid"])
with col8:
    options = st.selectbox("Options Level", options=["Full", "Semi Full", "Standard"])

st.markdown('<div class="section-header">Details</div>', unsafe_allow_html=True)

col9, col10 = st.columns(2)
with col9:
    origin = st.selectbox("Origin", options=["Saudi", "Gulf Arabic", "Other"])
with col10:
    color = st.selectbox("Color", options=["Black", "White", "Silver", "Grey",
                                            "Blue", "Red", "Brown", "Golden",
                                            "Navy", "Green", "Bronze", "Yellow",
                                            "Orange", "Oily", "Another Color"])

region = st.selectbox("Region", options=sorted([
    "Riyadh", "Jeddah", "Dammam", "Khobar", "Al-Medina", "Makkah",
    "Qassim", "Taef", "Hail", "Tabouk", "Al-Ahsa", "Jubail",
    "Abha", "Aseer", "Yanbu", "Al-Baha", "Najran", "Jazan",
    "Al-Jouf", "Arar", "Hafar Al-Batin", "Sakaka", "Al-Namas",
    "Besha", "Qurayyat", "Sabya", "Wadi Dawasir"
]))

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("Estimate Price"):
    car_age = datetime.now().year - year

    input_data = pd.DataFrame([{
        "Make"       : make,
        "Type"       : car_type,
        "Origin"     : origin,
        "Color"      : color,
        "Options"    : options,
        "Engine_Size": engine_size,
        "Fuel_Type"  : fuel_type,
        "Gear_Type"  : gear_type,
        "Mileage"    : mileage,
        "Region"     : region,
        "Car_Age"    : car_age,
    }])

    try:
        prediction = model.predict(input_data)[0]
        low        = prediction * 0.80
        high       = prediction * 1.20

        st.markdown(f"""
        <div class="price-card">
            <div class="price-label">Estimated Market Price</div>
            <div class="price-value">SAR {prediction:,.0f}</div>
            <div class="price-currency">Saudi Riyal</div>
            <div class="range-row">
                <div class="range-item">
                    <div class="range-label">Lower Estimate</div>
                    <div class="range-val">SAR {low:,.0f}</div>
                </div>
                <div class="range-item">
                    <div class="range-label">Model MAPE</div>
                    <div class="range-val">~20%</div>
                </div>
                <div class="range-item">
                    <div class="range-label">Upper Estimate</div>
                    <div class="range-val">SAR {high:,.0f}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="disclaimer">
            Estimated for a {year} {make} {car_type} with {mileage:,} km · {car_age} years old<br>
            Price range based on ±20% model error margin (MAPE).
            Actual prices may vary based on negotiation and market conditions.
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        st.info("Make sure the model file was saved with the same preprocessing pipeline used during training.")