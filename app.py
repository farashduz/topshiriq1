import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import os

st.title("🌸 Iris Flower Type Prediction with KNN (.pkl)")

PKL_FILE = "knn_model_bundle.pkl"

# =========================
# Agar .pkl mavjud bo'lmasa, uni yarating
# =========================
if not os.path.exists(PKL_FILE):
    df = pd.read_csv("iris_synthetic_data.csv")

    # Label encode
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['label'])

    X = df.drop(columns=["label"])
    y = df["label"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KNeighborsClassifier(n_neighbors=9)
    model.fit(X_scaled, y)

    best_cols = X.columns.tolist()

    bundle = {
        "model": model,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "best_cols": best_cols
    }

    joblib.dump(bundle, PKL_FILE)
    st.info(".pkl fayl yaratildi!")

# =========================
# Bundle ni yuklash
# =========================
bundle = joblib.load(PKL_FILE)

model = bundle["model"]
scaler = bundle["scaler"]
label_encoder = bundle["label_encoder"]
best_cols = bundle["best_cols"]

# =========================
# Inputlar
# =========================
st.sidebar.header("🔢 Kirish qiymatlari")
inputs = []
for col in best_cols:
    val = st.sidebar.number_input(col, value=0.0)
    inputs.append(val)

if st.button("Predict"):
    X_input = np.array(inputs).reshape(1, -1)
    X_scaled = scaler.transform(X_input)
    pred = model.predict(X_scaled)
    result = label_encoder.inverse_transform(pred)

    st.success(f"🌸 Bashorat qilingan gul turi: **{result[0]}**")
