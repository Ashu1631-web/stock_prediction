import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf

from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split

import plotly.express as px
import plotly.graph_objects as go

# ------------------------------
# Streamlit Page Config
# ------------------------------
st.set_page_config(page_title="Stock Prediction Dashboard", layout="wide")

st.title("📈 Stock Price Prediction using LSTM")
st.markdown("Professional Stock Forecasting Dashboard with KPI + Animated Graphs")

# ------------------------------
# Stock Input
# ------------------------------
ticker = st.text_input("Enter Stock Symbol (Example: AAPL, TSLA, INFY)", "AAPL")

if st.button("Download Data & Train Model"):

    # ------------------------------
    # Download Data
    # ------------------------------
    data = yf.download(ticker, period="5y")
    data.reset_index(inplace=True)

    st.success("✅ Stock Data Downloaded Successfully!")

    # ------------------------------
    # KPI Section
    # ------------------------------
    latest_close = data["Close"].iloc[-1]
    highest_price = data["High"].max()
    lowest_price = data["Low"].min()
    avg_volume = data["Volume"].mean()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("📌 Latest Close Price", f"${latest_close:.2f}")
    col2.metric("🚀 Highest Price", f"${highest_price:.2f}")
    col3.metric("📉 Lowest Price", f"${lowest_price:.2f}")
    col4.metric("📊 Avg Volume", f"{avg_volume:,.0f}")

    st.divider()

    # ------------------------------
    # Stock Price Line Chart
    # ------------------------------
    st.subheader("📉 Closing Price Trend")

    fig_line = px.line(
        data,
        x="Date",
        y="Close",
        title=f"{ticker} Closing Price Over Time",
    )
    st.plotly_chart(fig_line, use_container_width=True)

    # ------------------------------
    # Animated Bar Chart (Volume)
    # ------------------------------
    st.subheader("📊 Animated Trading Volume Bar Chart")

    data["Year"] = data["Date"].dt.year

    volume_year = data.groupby("Year")["Volume"].mean().reset_index()

    fig_bar = px.bar(
        volume_year,
        x="Year",
        y="Volume",
        title="📌 Average Volume per Year (Animated)",
        text_auto=True,
        animation_frame="Year",
    )

    st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    # ------------------------------
    # Prepare Data for LSTM
    # ------------------------------
    st.subheader("🤖 Training LSTM Model")

    x = data[["Open", "High", "Low", "Volume"]].to_numpy()
    y = data["Close"].to_numpy().reshape(-1, 1)

    xtrain, xtest, ytrain, ytest = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    # Reshape for LSTM
    xtrain = xtrain.reshape((xtrain.shape[0], xtrain.shape[1], 1))
    xtest = xtest.reshape((xtest.shape[0], xtest.shape[1], 1))

    # ------------------------------
    # LSTM Model Architecture
    # ------------------------------
    model = Sequential()
    model.add(LSTM(128, return_sequences=True,
                   input_shape=(xtrain.shape[1], 1)))
    model.add(LSTM(64))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mean_squared_error")

    with st.spinner("⏳ Training Model... Please wait"):
        model.fit(xtrain, ytrain, epochs=5, batch_size=1, verbose=0)

    st.success("✅ Model Trained Successfully!")

    # ------------------------------
    # Prediction Section
    # ------------------------------
    st.subheader("🔮 Predicted Stock Close Price")

    prediction = model.predict(xtest[:1])

    st.info(f"Predicted Close Price = ${prediction[0][0]:.2f}")

    # ------------------------------
    # Prediction vs Actual Graph
    # ------------------------------
    st.subheader("📌 Actual vs Predicted Graph")

    predicted_prices = model.predict(xtest[:50])
    actual_prices = ytest[:50]

    compare_df = pd.DataFrame({
        "Actual Price": actual_prices.flatten(),
        "Predicted Price": predicted_prices.flatten()
    })

    fig_compare = px.line(
        compare_df,
        title="Actual vs Predicted Stock Prices",
    )

    st.plotly_chart(fig_compare, use_container_width=True)
