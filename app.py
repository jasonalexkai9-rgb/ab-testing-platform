import streamlit as st
import pandas as pd
from groq import Groq
from analysis import run_ab_test, get_sample_data

st.set_page_config(page_title="A/B Test Analyzer", page_icon="🧪", layout="wide")
st.title("🧪 A/B Testing Platform")
st.subheader("Automated Analysis & AI Recommendations")

import os
from dotenv import load_dotenv
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

st.sidebar.title("Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
use_sample = st.sidebar.checkbox("Use Sample Data")

df = None
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")
elif use_sample:
    df = get_sample_data()
    st.info("Using sample data!")

if df is not None:
    st.subheader("📊 Raw Data Preview")
    st.dataframe(df.head(10))
    results = run_ab_test(df)

    st.subheader("📈 Test Results")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Version A", results["mean_a"])
    col2.metric("Version B", results["mean_b"])
    col3.metric("Lift", f"{results['lift']}%")
    col4.metric("P-Value", results["p_value"])

    st.subheader("🏆 Verdict")
    st.write(f"Significance: {results['significance']}")
    st.write(f"Winner: Version {results['winner']}")
    st.write(f"Confidence Interval: {results['ci_low']} to {results['ci_high']}")

    st.subheader("🤖 AI Agent Recommendation")
    if st.button("Generate AI Recommendation"):
        with st.spinner("AI Agent is analyzing your results..."):
            prompt = f"""
            You are an expert A/B testing analyst and data scientist. 
            Analyze these A/B test results and give a detailed business recommendation:
            
            - Version A conversion rate: {results['mean_a']}
            - Version B conversion rate: {results['mean_b']}
            - Lift: {results['lift']}%
            - P-value: {results['p_value']}
            - Statistical significance: {results['significance']}
            - Winner: {results['winner']}
            - 95% Confidence Interval: {results['ci_low']} to {results['ci_high']}
            
            Please provide:
            1. A clear winner declaration or reason why there is no winner yet
            2. What the confidence interval means in simple business terms
            3. Estimated business impact if they switch to the winner
            4. Any risks or confounding factors to watch out for
            5. Your final actionable recommendation
            """
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
            )
            st.success(chat_completion.choices[0].message.content)