import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from transformers import pipeline

# Load Hugging Face model untuk sentimen
sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=-1
)

def run_stock_prediction():
    st.header("📈 Stock Prediction AI")

    # Input fields
    ticker = st.text_input("Masukkan kode saham (contoh: AAPL, BBRI.JK):")
    horizon = st.number_input("Masukkan prediksi horizon (hari):", min_value=1, max_value=30, value=7)
    market_index = st.text_input("Masukkan indeks pasar (contoh: IHSG, S&P500, Nasdaq):")

    # Ambil hasil sentimen dari CompanyRecordAnalyzer jika ada
    sentiment = st.session_state.get("sentiment_text", "")
    if sentiment:
        st.info(f"Sentimen otomatis terisi dari Company Record Analyzer ✅")
    else:
        st.warning("Belum ada hasil dari Company Record Analyzer (opsional).")

    if st.button("Prediksi Harga Saham"):
        if ticker:
            st.info(f"Mengambil data historis saham **{ticker}** ...")
            data = yf.download(ticker, period="6mo", interval="1d")

            if data.empty:
                st.error("Data saham tidak ditemukan. Coba kode ticker lain.")
                return

            # Ambil harga penutupan terakhir
            closing_prices = data["Close"].values[-30:]  # 30 hari terakhir
            days_hist = np.arange(len(closing_prices))

            # Dummy prediksi harga
            pred_prices = []
            last_price = closing_prices[-1]
            for i in range(horizon):
                last_price += np.random.uniform(-2, 2)
                pred_prices.append(last_price)
            days_pred = np.arange(len(closing_prices), len(closing_prices) + horizon)

            # Analisis sentimen jika ada hasil dari CompanyRecordAnalyzer
            if sentiment:
                senti_result = sentiment_model(sentiment)[0]
                st.success(f"Sentimen terdeteksi: {senti_result['label']} (score={senti_result['score']:.2f})")
                if senti_result["label"] == "POSITIVE":
                    pred_prices = [p * 1.02 for p in pred_prices]
                elif senti_result["label"] == "NEGATIVE":
                    pred_prices = [p * 0.98 for p in pred_prices]

            # Plot grafik
            fig, ax = plt.subplots()
            ax.plot(days_hist, closing_prices, label="Harga Historis", marker="o")
            ax.plot(days_pred, pred_prices, label="Prediksi AI", marker="x", linestyle="--")
            ax.set_title(f"Prediksi Harga Saham {ticker} ({market_index})")
            ax.set_xlabel("Hari ke-")
            ax.set_ylabel("Harga")
            ax.legend()
            st.pyplot(fig)

            # Tabel prediksi
            st.subheader("📊 Data Prediksi")
            for i, p in enumerate(pred_prices, 1):
                st.write(f"Hari +{i}: {p:.2f}")
        else:
            st.error("Silakan masukkan kode saham terlebih dahulu.")
