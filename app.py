import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from groq import Groq
from dotenv import load_dotenv
import os
import io

# Load API key securely
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("\U0001F6A8 API Key is missing! Set it in Streamlit Secrets or a .env file.")
    st.stop()

# Streamlit UI
st.set_page_config(page_title="AI Forecasting Agent", page_icon="📈", layout="wide")
st.title("📈 Revenue Forecasting using Prophet")

# File uploader
uploaded_file = st.file_uploader("Upload your Excel file with 'Date' and 'Revenue' columns", type=["xlsx"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        st.write("### Uploaded Data Preview", df.head())

        # Check required columns
        if not {'Date', 'Revenue'}.issubset(df.columns):
            st.error("Uploaded file must contain 'Date' and 'Revenue' columns.")
            st.stop()

        df['Date'] = pd.to_datetime(df['Date'])
        df = df.rename(columns={"Date": "ds", "Revenue": "y"})

        scenarios = {
            "Base Case": 0.00,
            "Best Case": 0.10,
            "Worst Case": -0.10
        }

        forecasts = {}
        last_date = df['ds'].max()

        for name, adjustment in scenarios.items():
            df_adj = df.copy()
            model = Prophet()
            model.fit(df_adj)

            future = model.make_future_dataframe(periods=12, freq='M')
            forecast = model.predict(future)

            # Apply adjustment only to future
            future_mask = forecast['ds'] > last_date
            forecast.loc[future_mask, 'yhat'] *= (1 + adjustment)
            forecast.loc[future_mask, 'yhat_lower'] *= (1 + adjustment)
            forecast.loc[future_mask, 'yhat_upper'] *= (1 + adjustment)

            forecasts[name] = forecast

        # Plotting all scenarios
        st.write("### Scenario Comparison Forecast Plot")
        plt.figure(figsize=(10, 6))

        # Fill past data background
        plt.axvspan(df['ds'].min(), last_date, color='lightgray', alpha=0.4, label='Historical Period')

        for name, forecast in forecasts.items():
            plt.plot(forecast['ds'], forecast['yhat'], label=name)

        plt.axvline(x=last_date, color='gray', linestyle='--', label='Forecast Start')
        plt.xlabel("Date")
        plt.ylabel("Forecasted Revenue")
        plt.title("Revenue Forecast - Multi-Scenario")
        plt.legend()
        plt.grid(True)
        st.pyplot(plt.gcf())

        # Show forecast data from selected scenario
        scenario = st.selectbox("Select Scenario for Details & Commentary", list(scenarios.keys()))
        forecast = forecasts[scenario]
        forecast_future = forecast[forecast['ds'] > last_date][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        st.write(f"### Forecast Data - {scenario}", forecast_future.tail(12))

        # Download button for forecast data
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            forecast_future.to_excel(writer, index=False, sheet_name='Forecast')
            writer.save()
        st.download_button(
            label="📥 Download Forecast as Excel",
            data=buffer,
            file_name=f"forecast_{scenario.lower().replace(' ', '_')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # AI Commentary (only for base case)
        if scenario == "Base Case":
            st.subheader("🤖 AI-Generated Forecast Commentary")
            client = Groq(api_key=GROQ_API_KEY)

            prompt = f"""
            You are the Head of FP&A at an IT-software engineering company. Analyze the following historical data and base case trend from Prophet and provide:
            - Key trends and seasonal patterns in the historical performance.
            - Risks or anomalies in revenue development.
            - CFO-ready summary using the Pyramid Principle.
            - Actionable recommendations to improve revenue forecasting.

            Historical data:
            {df.to_json(orient='records')}

            Forecast trend:
            {forecast_future.to_json(orient='records')}
            """

            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a financial planning and analysis (FP&A) expert, specializing in SaaS companies."},
                    {"role": "user", "content": prompt}
                ],
                model="llama3-8b-8192",
            )

            ai_commentary = response.choices[0].message.content

            st.markdown('<div class="analysis-container">', unsafe_allow_html=True)
            st.subheader("📖 AI-Generated Commentary")
            st.write(ai_commentary)
            st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
