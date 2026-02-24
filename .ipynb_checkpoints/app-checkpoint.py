import streamlit as st
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Attention
import matplotlib.pyplot as plt

st.set_page_config(page_title="Stock AI Predictor", layout="wide")

@st.cache_resource
def load_my_model():
    return load_model(
        "hybrid_model.keras",
        compile=False,
        custom_objects={'Attention': Attention}
    )

model = load_my_model()

st.markdown("<h1 style='text-align: center;'>📊 AI Stock Prediction System</h1>", unsafe_allow_html=True)
st.markdown("---")

col1, col2 = st.columns([1,2])

with col1:
    st.subheader("🔍 Select Stock")
    stock = st.text_input("Stock Symbol:", "AAPL")
    predict = st.button("🚀 Predict")

with col2:
    st.subheader("📈 Stock Trend")

if predict:

    data = yf.download(stock, start="2018-01-01", end="2024-01-01")
    data = data[['Close']]

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    seq = scaled[-60:]
    seq = seq.reshape(1,60,1)

    prediction = model.predict(seq)

    pred_price = prediction[0][0]

    st.markdown("### 📊 Prediction Result")
    st.success(f"Predicted Next Price: {round(pred_price,2)}")

    fig, ax = plt.subplots()
    ax.plot(data[-100:], label="Historical Prices")
    ax.set_title("Last 100 Days Trend")
    ax.legend()

    st.pyplot(fig)