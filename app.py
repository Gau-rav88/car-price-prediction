import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# =========================
# Load trained objects
# =========================
model = joblib.load("car_price_model.pkl")
brand_map = joblib.load("brand_map.pkl")
features = joblib.load("features.pkl")

# =========================
# App title
# =========================
st.title("🚗 Car Price Prediction App")
st.write("Enter car details to predict the selling price")

# =========================
# User inputs
# =========================
year = st.number_input(
    "Year of Manufacture",
    min_value=1990,
    max_value=datetime.now().year,
    value=2015
)

kms_driven = st.number_input("Kilometers Driven", min_value=0)

user_selected_brand = st.selectbox(
    "Car Brand",
    list(brand_map.keys())
)

fuel_type = st.selectbox(
    "Fuel Type",
    ["Petrol", "Diesel", "CNG", "LPG", "Electric"]
)

seller_type = st.selectbox(
    "Seller Type",
    ["Individual", "Dealer", "Trustmark Dealer"]
)

transmission = st.selectbox(
    "Transmission",
    ["Manual", "Automatic"]
)

owner = st.selectbox(
    "Owner",
    [
        "First Owner",
        "Second Owner",
        "Third Owner",
        "Fourth & Above Owner",
        "Test Drive Car"
    ]
)

# =========================
# Manual encoding maps
# =========================
fuel_map = {
    "Petrol": 0,
    "Diesel": 1,
    "CNG": 2,
    "LPG": 3,
    "Electric": 4
}

seller_map = {
    "Individual": 0,
    "Dealer": 1,
    "Trustmark Dealer": 2
}

transmission_map = {
    "Manual": 0,
    "Automatic": 1
}

owner_map = {
    "First Owner": 0,
    "Second Owner": 1,
    "Third Owner": 2,
    "Fourth & Above Owner": 3,
    "Test Drive Car": 4
}

# =========================
# Prediction
# =========================
if st.button("Predict Price"):

    # Safety check (optional but good)
    if user_selected_brand not in brand_map:
        st.error("Selected brand was not seen during training.")
        st.stop()

    car_age = datetime.now().year - year

    # Encode categorical values
    fuel_encoded = fuel_map[fuel_type]
    seller_encoded = seller_map[seller_type]
    transmission_encoded = transmission_map[transmission]
    owner_encoded = owner_map[owner]

    # Base input dataframe (without brand dummies)
    input_df = pd.DataFrame([{
        "km_driven": kms_driven,
        "fuel": fuel_encoded,
        "seller_type": seller_encoded,
        "transmission": transmission_encoded,
        "owner": owner_encoded,
        "car_age": car_age
    }])

    # Add all missing feature column column
    for col in features:
        if col not in input_df.columns:
            input_df[col] = 0

    # One-hot encode selected brand
    brand_encoded = brand_map[user_selected_brand]
    brand_col = f"brand_{brand_encoded}"

    if brand_col in input_df.columns:
        input_df[brand_col] = 1

    # Reorder columns exactly like training
    input_df = input_df[features]

    # Predict
    prediction = model.predict(input_df)[0]

    st.success(f"💰 Estimated Price: ₹{int(prediction):,}")
