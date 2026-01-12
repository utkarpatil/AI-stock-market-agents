import streamlit as st
import json
import subprocess
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Stock Market Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.75rem;
        border-radius: 10px;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .recommendation-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("📈 Stock Market Analysis Dashboard")
st.markdown("### AI-Powered Multi-Agent Stock Analysis System")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key input (if needed)
    api_key = st.text_input("OpenAI API Key", type="password", 
                            help="Enter your OpenAI API key")
    
    st.markdown("---")
    st.markdown("### 📊 About")
    st.info("""
    This system uses multiple AI agents to analyze stocks:
    - 📊 Quantitative Analysis
    - 📰 Fundamental Analysis
    - 📈 Technical Analysis
    - 💰 Macro Economics
    - ⚖️ Risk Management
    """)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Enter Stock Tickers")
    tickers_input = st.text_input(
        "Stock Tickers",
        placeholder="AAPL, GOOGL, MSFT, TSLA",
        help="Enter stock ticker symbols separated by commas"
    )

with col2:
    st.subheader("Actions")
    analyze_button = st.button("🚀 Run Analysis", use_container_width=True)

# Analysis section
if analyze_button:
    if not tickers_input:
        st.error("⚠️ Please enter at least one stock ticker")
    else:
        # Parse tickers
        tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
        
        st.success(f"Analyzing: {', '.join(tickers)}")
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Update progress
            status_text.text("🔄 Initializing agents...")
            progress_bar.progress(20)
            
            # Set API key as environment variable if provided
            if api_key:
                os.environ['OPENAI_API_KEY'] = api_key
            
            # Run the analysis
            status_text.text("🔍 Running multi-agent analysis...")
            progress_bar.progress(40)
            
            # Execute your main.py script
            result = subprocess.run(
                ['python', 'main.py'] + tickers,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            progress_bar.progress(80)
            status_text.text("📊 Processing results...")
            
            # Check if analysis was successful
            if os.path.exists('analysis_output.json'):
                with open('analysis_output.json', 'r') as f:
                    analysis_data = json.load(f)
                
                progress_bar.progress(100)
                status_text.text("✅ Analysis Complete!")
                
                # Display results
                st.markdown("---")
                st.header("📊 Analysis Results")
                
                # Overall Statistics
                st.subheader("Overall Statistics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Total Opportunities",
                        analysis_data.get('total_opportunities', 0)
                    )
                
                with col2:
                    st.metric(
                        "High Confidence Trades",
                        analysis_data.get('high_confidence_trades', 0)
                    )
                
                with col3:
                    st.metric(
                        "Average Confidence",
                        f"{analysis_data.get('average_confidence', 0):.1f}%"
                    )
                
                with col4:
                    status = analysis_data.get('status', 'Unknown')
                    st.metric("Status", status)
                
                # Actions Breakdown
                st.markdown("---")
                st.subheader("📋 Actions Breakdown")
                
                actions = analysis_data.get('actions_breakdown', {})
                if actions:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.info(f"🔵 HOLD: {actions.get('HOLD', 0)}")
                    with col2:
                        st.success(f"🟢 BUY: {actions.get('BUY', 0)}")
                    with col3:
                        st.error(f"🔴 SELL: {actions.get('SELL', 0)}")
                
                # Top Recommendation
                st.markdown("---")
                st.subheader("🎯 Top Recommendation")
                
                top_rec = analysis_data.get('top_recommendation', {})
                if top_rec:
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"### {top_rec.get('ticker', 'N/A')}")
                        st.markdown(f"**Action:** {top_rec.get('action', 'N/A')}")
                        st.markdown(f"**Expected ROI:** {top_rec.get('expected_roi', 'N/A')}")
                    
                    with col2:
                        confidence = top_rec.get('confidence', '0%')
                        st.metric("Confidence", confidence)
                
                # All Recommendations
                st.markdown("---")
                st.subheader("📈 All Recommendations")
                
                recommendations = analysis_data.get('recommendations', [])
                
                for i, rec in enumerate(recommendations, 1):
                    with st.expander(f"#{i} - {rec.get('ticker', 'N/A')} - {rec.get('action', 'N/A')}"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown(f"**Priority:** #{rec.get('priority', 'N/A')}")
                            st.markdown(f"**Confidence:** {rec.get('confidence', 'N/A')}")
                        
                        with col2:
                            st.markdown(f"**Expected ROI:** {rec.get('expected_roi', 'N/A')}")
                            st.markdown(f"**Time Horizon:** {rec.get('time_horizon', 'N/A')}")
                        
                        with col3:
                            st.markdown(f"**Entry:** ${rec.get('entry', 'N/A')}")
                            st.markdown(f"**Target:** ${rec.get('target', 'N/A')}")
                            st.markdown(f"**Stop Loss:** ${rec.get('stop_loss', 'N/A')}")
                        
                        st.markdown("**Justification:**")
                        st.info(rec.get('justification', 'No justification provided'))
                
                # Download button for JSON
                st.markdown("---")
                st.download_button(
                    label="📥 Download Full Report (JSON)",
                    data=json.dumps(analysis_data, indent=2),
                    file_name=f"stock_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
                
            else:
                st.error("❌ Analysis output file not found")
                if result.stderr:
                    st.error(f"Error details: {result.stderr}")
        
        except subprocess.TimeoutExpired:
            st.error("⏱️ Analysis timed out. Please try with fewer stocks.")
        
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            if 'result' in locals() and result.stderr:
                st.error(f"Details: {result.stderr}")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>Made with ❤️ using Streamlit | Multi-Agent Stock Analysis System</p>
        <p style='font-size: 0.8em;'>⚠️ This is not financial advice. Always do your own research.</p>
    </div>
""", unsafe_allow_html=True)