import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("🚨 API Key is missing! Set it in Streamlit Secrets or a .env file.")
    st.stop()

# Streamlit Page Config
st.set_page_config(page_title="📈 AI Revenue Forecasting with Prophet", page_icon="🤖", layout="wide")

# Page Header
st.title("🤖 AI-Powered Revenue Forecasting")
st.markdown("Upload an Excel file with `Date` and `Revenue` columns to forecast future revenues using Prophet.")

# File Upload
uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx", "xls"])

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
        st.success("✅ File uploaded successfully!")

        # Check required columns
        if "Date" not in df.columns or "Revenue" not in df.columns:
            st.error("🚨 Excel file must contain 'Date' and 'Revenue' columns.")
        else:
            st.subheader("📊 Preview of Uploaded Data")
            st.dataframe(df.head())

            df = df[['Date', 'Revenue']].rename(columns={"Date": "ds", "Revenue": "y"})
            df['ds'] = pd.to_datetime(df['ds'])

            # Optional log transformation
            use_log = st.checkbox("Apply log transformation to Revenue", value=False)
            if use_log:
                df['y'] = np.log1p(df['y'])

            # Forecast configuration
            future_periods = st.slider("Select forecast period (months)", 1, 36, 12)

            # Fit Prophet model with enhancements
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.1
            )
            model.fit(df)

            # Forecast
            future = model.make_future_dataframe(periods=future_periods, freq='M')
            forecast = model.predict(future)

            if use_log:
                forecast[['yhat', 'yhat_lower', 'yhat_upper']] = np.expm1(forecast[['yhat', 'yhat_lower', 'yhat_upper']])

            # Show forecast plot
            st.subheader("📈 Forecast Plot")
            fig1 = model.plot(forecast)
            st.pyplot(fig1)

            # Show forecast components
            st.subheader("📊 Forecast Components")
            fig2 = model.plot_components(forecast)
            st.pyplot(fig2)

            # Show forecast table
            st.subheader("📋 Forecast Data Table")
            st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(future_periods))

            # AI Forecast Commentary
            st.subheader("🧠 AI-Generated Forecast Commentary")
            client = Groq(api_key=GROQ_API_KEY)

            ai_prompt = f"""
            You are a financial forecasting expert. Analyze the following Prophet forecast data in JSON format and provide:
            - Confidence insights on the forecast
            - Seasonality or trend observations
            - Risks and assumptions
            - Actionable insights for CFO decision-making

            Forecast data:
            {forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(future_periods).to_json(orient='records')}
            """

            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a senior FP&A specialist with expertise in time series forecasting."},
                    {"role": "user", "content": ai_prompt}
                ],
                model="llama3-8b-8192",
            )

            ai_comment = response.choices[0].message.content

            # Display AI commentary
            st.markdown('<div class="analysis-container">', unsafe_allow_html=True)
            st.subheader("📖 AI Forecast Commentary")
            st.write(ai_comment)
            st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
else:
    st.info("📂 Please upload an Excel file to begin.")
