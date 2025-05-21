import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open("xgb_model.pkl", "rb"))

st.title(" Cryptocurrency Liquidity Predictor")

# Input fields
price = st.number_input("Current Price", value=1.0)
price_change_score = st.number_input("Price Change Score", value=0.0)
volume_to_marketcap = st.number_input("Volume-to-Market Cap Ratio", value=0.1)
mkt_cap = st.number_input("Market Capitalization", value=1.0)

# Predict
if st.button("Predict Liquidity (24h Volume)"):
    features = np.array([[price, price_change_score, volume_to_marketcap, mkt_cap]])
    prediction = model.predict(features)
    st.success(f"Predicted 24h Volume: {prediction[0]:.2f}")
