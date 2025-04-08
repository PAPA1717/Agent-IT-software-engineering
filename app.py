import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
import os
from groq import Groq
from dotenv import load_dotenv

# Load API key securely
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("\U0001F6A8 API Key is missing! Set it in Streamlit Secrets or a .env file.")
    st.stop()

# **🎨 Streamlit UI Styling**
st.set_page_config(page_title="AI Forecasting Agent", page_icon="⏳", layout="wide")
st.title("📈 Revenue Forecasting with Prophet")

# **📤 File Upload**
st.sidebar.header("Upload Excel File")
uploaded_file = st.sidebar.file_uploader("Choose a file", type=["xlsx"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        df.columns = [col.strip().lower() for col in df.columns]

        if 'date' not in df.columns or 'revenue' not in df.columns:
            st.error("The Excel file must contain 'Date' and 'Revenue' columns.")
            st.stop()

        df['date'] = pd.to_datetime(df['date'])
        df = df.rename(columns={"date": "ds", "revenue": "y"})

        # **📊 Display Data Preview**
        st.subheader("📋 Raw Data Preview")
        st.dataframe(df.head())

        # **🔮 Forecasting with Prophet**
        st.subheader("🔮 Prophet Forecast")
        model = Prophet()
        model.fit(df)

        future = model.make_future_dataframe(periods=12, freq='M')
        forecast = model.predict(future)

        fig1 = model.plot(forecast)
        st.pyplot(fig1)

        fig2 = model.plot_components(forecast)
        st.pyplot(fig2)

        # **📌 AI Commentary**
        st.subheader("🤖 AI-Generated Forecast Commentary")

        # Tooltip for Pyramid Principle
        with st.expander("❓ What is the Pyramid Principle?"):
            st.markdown("""
            **The Pyramid Principle** is a communication framework that helps convey ideas clearly and logically:
            
            - **Top**: Start with the key message or conclusion.
            - **Middle**: Follow with 2–4 supporting arguments.
            - **Bottom**: End with detailed data and evidence.
            
            This structure ensures clarity and makes your message easy to follow — especially useful for CFOs and senior decision-makers.
            """)

        client = Groq(api_key=GROQ_API_KEY)

        forecast_summary = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(12).to_json(orient="records")

        prompt = f"""
        You are the Head of FP&A at an IT-software engineering company. Your task is to analyze the forecast results and provide:
        - Key trends and insights for the next 12 months.
        - Any expected seasonality or risks.
        - A CFO-ready summary using the Pyramid Principle.
        - Recommendations for revenue growth.

        Here is the forecast data in JSON format: {forecast_summary}
        """

        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a forecasting and FP&A expert."},
                {"role": "user", "content": prompt}
            ],
            model="llama3-8b-8192",
        )

        ai_commentary = response.choices[0].message.content
        st.markdown('<div class="analysis-container">', unsafe_allow_html=True)
        st.write(ai_commentary)
        st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ An error occurred while processing the file: {str(e)}")
else:
    st.info("Please upload an Excel file to begin.")
