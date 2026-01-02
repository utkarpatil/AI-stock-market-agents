import streamlit as st
import json
import os
from main import run_analysis_sync

st.set_page_config(page_title="AI Stock Market Analysis", layout="wide")

st.title("📈 AI Stock Market Analysis")
st.caption("Enter company names only — AI does the rest")

st.warning("Educational purpose only. Not financial advice.")

# -------- USER INPUT -------- #

companies_input = st.text_input(
    "Enter Company Names (comma separated)",
    placeholder="NVIDIA, Tesla, Apple"
)

risk_profile = st.selectbox(
    "Risk Profile",
    ["conservative", "moderate", "aggressive"]
)

time_horizon = st.selectbox(
    "Time Horizon",
    ["short-term", "medium-term", "long-term"]
)

# -------- RUN BUTTON -------- #

if st.button("🚀 Run Analysis"):
    if not companies_input.strip():
        st.error("Please enter at least one company name.")
    else:
        companies = [c.strip() for c in companies_input.split(",")]

        try:
            with st.spinner("Running AI agents (this may take 2–3 minutes)..."):
                run_analysis_sync(
                    companies=companies,
                    risk_profile=risk_profile,
                    time_horizon=time_horizon
                )
            st.success("Analysis completed!")

        except Exception as e:
            st.error(str(e))

# -------- SHOW OUTPUT -------- #

if os.path.exists("analysis_output.json"):
    with open("analysis_output.json") as f:
        result = json.load(f)

    if not result.get("success"):
        st.error(result.get("error", "Analysis failed"))
    else:
        summary = result.get("summary", {})

        st.subheader("📊 Overall Summary")
        c1, c2, c3 = st.columns(3)

        c1.metric("Total Opportunities", summary.get("total_opportunities", 0))
        c2.metric("Avg Confidence", f"{summary.get('average_confidence', 0):.1f}%")
        c3.metric("High Confidence Trades", summary.get("high_confidence_trades", 0))

        st.subheader("🏆 Top Recommendation")
        top = summary.get("top_recommendation", {})
        if top:
            st.info(
                f"""
                **Ticker:** {top.get('ticker')}  
                **Action:** {top.get('action')}  
                **Confidence:** {top.get('confidence', 0):.1f}%  
                **Expected ROI:** {top.get('expected_roi', 0):.2f}%
                """
            )
