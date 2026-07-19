"""
=====================================================================
CROP YIELD PREDICTOR -- Streamlit Web App
=====================================================================
This is the app that gets DEPLOYED. It loads the model we exported in
Phase 4 (crop_yield_model.pkl) and gives anyone a simple web form to
get a yield prediction -- no coding required on their end.

HOW STREAMLIT WORKS (if you've never used it):
Streamlit turns a plain Python script into a website. Every time a
user changes an input (like moving a slider), Streamlit re-runs this
whole script top-to-bottom and redraws the page with the new result.
You don't write any HTML/CSS/JavaScript -- just Python.
"""

import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------------------
# Load the trained model and dropdown options (saved in Phase 4)
# ---------------------------------------------------------------
# st.cache_resource makes Streamlit load these files ONCE and reuse
# them for every user, instead of reloading on every single click
# (which would be slow).

@st.cache_resource
def load_model():
    model = joblib.load("crop_yield_model.pkl")
    options = joblib.load("dropdown_options.pkl")
    return model, options

model, options = load_model()

# ---------------------------------------------------------------
# Page layout and title
# ---------------------------------------------------------------
st.set_page_config(page_title="Crop Yield Predictor", page_icon="🌾")
st.title("🌾 Crop Yield Predictor")
st.write(
    "Predict expected crop yield (hg/ha) using historical patterns "
    "in country, crop type, rainfall, temperature, and pesticide use."
)

# ---------------------------------------------------------------
# Input form -- this is what the user actually interacts with
# ---------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    area = st.selectbox("Country", options['areas'])
    item = st.selectbox("Crop", options['items'])
    year = st.number_input("Year", min_value=1990, max_value=2035, value=2024)

with col2:
    rainfall = st.number_input("Average rainfall (mm/year)", min_value=0.0, value=1000.0)
    pesticides = st.number_input("Pesticide use (tonnes)", min_value=0.0, value=5000.0)
    avg_temp = st.number_input("Average temperature (°C)", min_value=-10.0, max_value=40.0, value=20.0)

# ---------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------
if st.button("Predict Yield"):
    # Build a single-row table matching the exact column names/order
    # the model was trained on.
    input_df = pd.DataFrame([{
        'Area': area,
        'Item': item,
        'Year': year,
        'average_rain_fall_mm_per_year': rainfall,
        'pesticides_tonnes': pesticides,
        'avg_temp': avg_temp
    }])

    prediction = model.predict(input_df)[0]

    st.success(f"### Predicted Yield: {prediction:,.0f} hg/ha")
    st.caption(f"That's approximately {prediction/10000:,.2f} tonnes per hectare.")

st.divider()
st.caption(
    "Model: Random Forest Regressor | Test R² = 0.98 | "
    "Trained on FAO crop data, 1990-2013."
)
