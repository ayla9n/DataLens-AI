import streamlit as st
import pandas as pd
import sys
import os
import importlib
import tempfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pipeline.clean as clean_module
import AI_Insights.insights as insights_module
import dashboard.charts as charts_module

importlib.reload(clean_module)
importlib.reload(insights_module)
importlib.reload(charts_module)

from pipeline.clean import clean_data
from AI_Insights.insights import (
    generate_insights,
    generate_dataset_description,
    generate_chart_recommendations,
    DOMAIN_PROMPTS
)
from dashboard.charts import create_all_charts


# PAGE CONFIG
st.set_page_config(
    page_title="DataLens AI",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
    <style>
        .stButton > button {
            background-color: #5DADE2;
            color: white;
            border: none;
            border-radius: 8px;
            font-weight: bold;
        }
        .stButton > button:hover {
            background-color: #2E86C1;
            color: white;
        }
        .stat-box {
            background-color: #EBF5FB;
            border-radius: 10px;
            padding: 12px 16px;
            margin: 4px 0;
            font-size: 0.9rem;
        }
    </style>
""", unsafe_allow_html=True)

# SESSION STATE
defaults = {
    "df": None,
    "stats": None,
    "insights": None,
    "description": None,
    "chart_recommendations": None,
    "domain": None,
    "suggested_domain": None,
    "last_file": None,
}

for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

if "sample" not in st.session_state:
    st.session_state.sample = None


def reset_state():
    """Clears all session state so a new file loads fresh"""
    for key in defaults:
        st.session_state[key] = None
    st.session_state.sample = None  


# SAMPLE DATASET PATHS
SAMPLE_DATASETS = {
    "sales": {
        "path": "./example-datasets/sams_bakery_sales.csv",
        "name": "sams_bakery_sales.csv",
        "domain": "Sales & Revenue"
    },
    "marketing": {
        "path": "./example-datasets/sams_bakery_marketing.csv",
        "name": "sams_bakery_marketing.csv",
        "domain": "Marketing & Campaigns"
    },
    "finance": {
        "path": "./example-datasets/sams_bakery_finance.csv",
        "name": "sams_bakery_finance.csv",
        "domain": "Finance & Accounting"
    }
}


# SIDEBAR
with st.sidebar:
    st.title("📈 DataLens AI")
    st.markdown("*AI-powered data insights, Made for small businesses*")
    st.divider()

    uploaded_file = st.file_uploader(  
        "Upload your CSV",
        type=["csv"],
        help="Upload a sales, marketing or finance CSV to get started"
    )

    # Show sample name in sidebar if a sample is loaded and no file uploaded
    if st.session_state.sample and not uploaded_file:
        sample = SAMPLE_DATASETS[st.session_state.sample]
        st.caption(f"Using sample: **{sample['name']}**")

    # Reset state when a new file is uploaded
    if uploaded_file:
        if st.session_state.last_file != uploaded_file.name:
            reset_state()
            st.session_state.last_file = uploaded_file.name

    if uploaded_file or st.session_state.sample:  

        if st.session_state.suggested_domain:
            suggestion = st.session_state.suggested_domain
            st.caption(f"💡 We think this looks like: **{suggestion['domain']}** — {suggestion['reason']}")

        domain = st.selectbox(
            "What is this data about?",
            list(DOMAIN_PROMPTS.keys()),
            index=list(DOMAIN_PROMPTS.keys()).index(
                st.session_state.suggested_domain["domain"]
            ) if st.session_state.suggested_domain else 0,
            help="Choose the category that best describes your data"
        )
        st.session_state.domain = domain

        st.divider()

        run_button = st.button("✨ Generate Insights", width="stretch")

        if st.button("🔄 Load New Dataset", width="stretch"):
            reset_state()
            st.rerun()

    else:
        run_button = False  

    st.divider()
    st.caption("Built with Google Gemini API + Streamlit")



# LANDING PAGE 
if not uploaded_file and not st.session_state.sample:
    st.title("📈 Welcome to DataLens AI")
    st.markdown(
        "Upload your sales, marketing, or finance CSV "
        "and get instant AI-powered insights."
    )
    st.divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Auto Cleaning**\n\nHandles missing values, duplicates and type detection automatically")
    with col2:
        st.info("**AI Insights**\n\nGet trends, patterns and recommendations tailored to your data")
    with col3:
        st.info("**Smart Charts**\n\nAI picks the best visualizations for your sales, marketing or finance data")

    st.divider()

    st.subheader("🧁 Try a Sample Dataset")
    st.markdown(
    "Not sure where to start? Load one of "
    "<span style='color:#FF6B8A; font-weight:600;'>Sam's Bakery</span>"
    " datasets to explore the app.",
    unsafe_allow_html=True)

    s1, s2, s3 = st.columns(3)

    with s1:
        st.markdown("**Sales & Revenue**")
        st.caption("140 orders across 12 months — products, regions, customers and revenue")
        if st.button("Load Sales Dataset", width="stretch"):
            st.session_state.sample = "sales"
            st.rerun()

    with s2:
        st.markdown("**Marketing & Campaigns**")
        st.caption("20 campaigns across email, social, events and paid search")
        if st.button("Load Marketing Dataset", width="stretch"):
            st.session_state.sample = "marketing"
            st.rerun()

    with s3:
        st.markdown("**Finance & Accounting**")
        st.caption("Monthly P&L with revenue, expenses, profit and capital entries")
        if st.button("Load Finance Dataset", width="stretch"):
            st.session_state.sample = "finance"
            st.rerun()



# MAIN DASHBOARD 
else:
    # HANDLE SAMPLE DATASET LOADING 
    if st.session_state.sample and st.session_state.df is None:
        sample_key = st.session_state.sample
        sample = SAMPLE_DATASETS[sample_key]

        with st.spinner(f"Loading and cleaning {sample['name']}..."):
            df, stats = clean_data(sample["path"])

        st.session_state.df = df
        st.session_state.stats = stats
        st.session_state.domain = sample["domain"]
        st.session_state.last_file = sample["name"]

    # HANDLE UPLOADED FILE LOADING 
    elif uploaded_file and st.session_state.df is None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        with st.spinner("Cleaning your data..."):
            df, stats = clean_data(tmp_path)

        os.unlink(tmp_path)

        st.session_state.df = df
        st.session_state.stats = stats

    #pull from session state
    df = st.session_state.df
    stats = st.session_state.stats
    domain = st.session_state.domain or "Sales & Revenue"

    if df is None:
        st.error("Something went wrong loading your file. Please check it is a valid CSV.")
        st.stop()

    # DESCRIPTION AND DOMAIN 
    if st.session_state.description is None:
        with st.spinner("Analyzing your dataset..."):
            result = generate_dataset_description(df, domain)
            st.session_state.description = result["description"]
            st.session_state.suggested_domain = {
                "domain": result["domain"],
                "reason": result["domain_reason"]
            }
        st.rerun()

    # HEADER 
    st.title("📈 DataLens AI")

    # Show file name — handle both uploaded file and sample
    file_name = uploaded_file.name if uploaded_file else st.session_state.last_file
    st.markdown(f"### {file_name}")
    st.markdown(f"_{st.session_state.description}_")
    st.markdown("""
    > **Auto-cleaned** your data &nbsp;|&nbsp;
    **AI Insights** generated based on your data &nbsp;|&nbsp;
    **Charts** created for your dataset
    """)
    st.divider()

    # DATASET OVERVIEW 
    st.subheader("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Total Rows",
        f"{df.shape[0]:,}",
        delta=f"{stats['cleaned_rows'] - stats['original_rows']:,} from cleaning"
    )
    col2.metric("Total Columns", df.shape[1])
    col3.metric("Numeric Columns", len(df.select_dtypes(include="number").columns))
    col4.metric("Missing Values", df.isnull().sum().sum())

    cl1, cl2, cl3 = st.columns(3)

    with cl1:
        st.markdown(f"""<div class='stat-box'>
            <b>Original rows</b><br>{stats['original_rows']:,}
        </div>""", unsafe_allow_html=True)

    with cl2:
        st.markdown(f"""<div class='stat-box'>
            <b>Duplicates removed</b><br>{stats['duplicates_removed']:,}
        </div>""", unsafe_allow_html=True)

    with cl3:
        st.markdown(f"""<div class='stat-box'>
            <b>Missing values filled</b><br>{stats['missing_filled']:,}
        </div>""", unsafe_allow_html=True)

    st.divider()

    # TABS 
    tab1, tab2, tab3 = st.tabs(["Data Preview", "AI Insights", "📈 Charts"])

    with tab1:
        st.subheader("Cleaned Data Preview")
        st.dataframe(df.head(20), width="stretch")

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Cleaned CSV",
            data=csv,
            file_name=f"cleaned_{file_name}",
            mime="text/csv"
        )

    with tab2:
        if run_button:
            with st.spinner("✨ Generating AI insights..."):
                st.session_state.insights = generate_insights(df, domain)

        if st.session_state.insights:
            st.markdown(st.session_state.insights)
        else:
            st.info("Click **Generate Insights** in the sidebar to get AI analysis")

    with tab3:
        if run_button:
            with st.spinner("Generating chart recommendations..."):
                st.session_state.chart_recommendations = generate_chart_recommendations(df, domain)

        if st.session_state.chart_recommendations:
            create_all_charts(df, st.session_state.chart_recommendations)
        else:
            st.info("Click **Generate Insights** in the sidebar to generate charts")