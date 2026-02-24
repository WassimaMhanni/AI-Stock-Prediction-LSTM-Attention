import streamlit as st
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

model = load_model("hybrid_model.keras", compile=False)

st.title("Prédiction Boursière avec LSTM + Attention")

stock = st.text_input("Entrer le symbole de l'action :", "AAPL")

if st.button("Prédire"):

    data = yf.download(stock, start="2018-01-01", end="2024-01-01")
    data = data[['Close']]

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    seq = scaled[-60:]
    seq = seq.reshape(1,60,1)

    prediction = model.predict(seq)

    st.write("Prix prédit :", prediction[0][0])