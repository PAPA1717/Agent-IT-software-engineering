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

# Streamlit UI configuration
st.set_page_config(page_title="AI Revenue Forecasting with Prophet", page_icon="📊", layout="wide")
st.title("📊 AI-Powered Revenue Forecasting with Prophet")

# File upload
uploaded_file = st.file_uploader("Upload Excel File with Date and Revenue columns", type=["xlsx"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        st.subheader("📈 Preview of Uploaded Data")
        st.dataframe(df.head())

        # Ensure required columns exist
        if 'Date' not in df.columns or 'Revenue' not in df.columns:
            st.error("The Excel file must contain 'Date' and 'Revenue' columns.")
            st.stop()

        # Preprocess data
        df = df[['Date', 'Revenue']].rename(columns={'Date': 'ds', 'Revenue': 'y'})
        df['ds'] = pd.to_datetime(df['ds'])

        # Fit Prophet model
        model = Prophet()
        model.fit(df)

        # Create future dataframe and forecast
        future = model.make_future_dataframe(periods=12, freq='M')
        forecast = model.predict(future)

        # Plot forecast
        st.subheader("🌍 Forecast Plot")
        fig1 = model.plot(forecast)
        st.pyplot(fig1)

        # Display components
        st.subheader("🔄 Forecast Components")
        fig2 = model.plot_components(forecast)
        st.pyplot(fig2)

        # AI commentary using Groq
        client = Groq(api_key=GROQ_API_KEY)

        forecast_json = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(12).to_json(orient='records')
        prompt = f"""
        You are an FP&A expert. Analyze the following Prophet revenue forecast data in JSON format:
        {forecast_json}

        Provide:
        - Forecast confidence insights
        - Seasonality or trends observed
        - Key risks or assumptions
        - Actionable insights for CFOs
        """

        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a senior FP&A specialist."},
                {"role": "user", "content": prompt}
            ],
            model="llama3-8b-8192",
        )

        ai_commentary = response.choices[0].message.content

        # Display AI-generated commentary
        st.markdown('<div class="analysis-container">', unsafe_allow_html=True)
        st.subheader("📖 AI Forecast Commentary")
        st.write(ai_commentary)
        st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error processing file: {e}")
