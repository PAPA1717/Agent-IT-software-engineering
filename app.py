import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import os
from dotenv import load_dotenv
from groq import Groq

# Load API key securely
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("\U0001F6A8 API Key is missing! Set it in Streamlit Secrets or a .env file.")
    st.stop()

# Streamlit UI Setup
st.set_page_config(page_title="Revenue Forecasting Agent", page_icon="📊", layout="wide")
st.title("📊 AI Revenue Forecasting with Prophet")

# File Upload
uploaded_file = st.file_uploader("Upload your Excel file with Date and Revenue columns", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    
    if 'Date' not in df.columns or 'Revenue' not in df.columns:
        st.error("The file must contain 'Date' and 'Revenue' columns.")
    else:
        # Prepare data for Prophet
        df = df[['Date', 'Revenue']].rename(columns={'Date': 'ds', 'Revenue': 'y'})
        df['ds'] = pd.to_datetime(df['ds'])

        # Fit Prophet model
        model = Prophet()
        model.fit(df)

        # Forecasting future periods
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        # Plot Forecast
        fig1 = model.plot(forecast)
        st.subheader("🔢 Forecast Plot")
        st.pyplot(fig1)

        # Plot forecast components
        fig2 = model.plot_components(forecast)
        st.subheader("📊 Forecast Components")
        st.pyplot(fig2)

        # AI-generated insights using Groq
        st.subheader("🧠 AI-Generated Forecast Insights")
        client = Groq(api_key=GROQ_API_KEY)

        prompt = f"""
        You are an expert FP&A analyst.
        Analyze the revenue forecast below. Highlight trends, seasonality, anomalies,
        and generate a short executive summary for the CFO.

        Forecast data:
        {forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30).to_json(orient='records')}
        """

        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an expert financial forecaster and FP&A advisor."},
                {"role": "user", "content": prompt}
            ],
            model="llama3-8b-8192",
        )

        ai_commentary = response.choices[0].message.content

        st.markdown('<div class="analysis-container">', unsafe_allow_html=True)
        st.write(ai_commentary)
        st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("Please upload an Excel file with 'Date' and 'Revenue' columns to begin.")
