import streamlit as st
import joblib
import numpy as np
import pandas as pd

bundle = joblib.load("knn_model_bundle.pkl")

model = bundle["model"]
scaler = bundle["scaler"]
label_encoder = bundle["label_encoder"]
best_cols = bundle["best_cols"]

st.title("Flower type with KNN model")

# inputlar (ustun soniga qarab moslang!)
inputs = []
for col in best_cols:
    val = st.number_input(col, value=0.0)
    inputs.append(val)

if st.button("Predict"):
    X = np.array(inputs).reshape(1, -1)
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)
    result = label_encoder.inverse_transform(pred)

    st.success(f"🌸 Bashorat: {result[0]}")
