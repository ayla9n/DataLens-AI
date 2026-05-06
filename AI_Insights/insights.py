import pandas as pd
import numpy as np 
from groq import Groq
import json
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

DOMAIN_PROMPTS = {
    "Sales & Revenue": """
        You are a senior sales analyst. Focus on:
        - Revenue trends over time and across regions/segments
        - Top and bottom performing products or categories
        - Seasonal patterns in sales data
        - Actionable recommendations for improving sales performance
    """,

    "Finance & Accounting": """
        You are a financial analyst. Focus on:
        - Key financial metrics and what they indicate
        - Cost patterns and areas of concern
        - Period over period comparisons
        - Risk indicators or anomalies in the numbers
    """,

    "Marketing & Campaigns": """
        You are a marketing analyst. Focus on:
        - Campaign performance and ROI patterns
        - Audience segments that respond best
        - Conversion and engagement trends over time
        - Recommendations for optimizing future campaigns
    """,

    "Personal Finance": """
        You are a personal finance advisor. Focus on:
        - Total spending and average monthly breakdown
        - Top spending categories and how they trend over time
        - Unusual spikes or patterns worth flagging
        - Practical budget recommendations grounded in the numbers
    """,

    "General / Other": """
        You are a senior data analyst. Focus on:
        - Key patterns and trends in the data
        - Notable statistical findings
        - Relationships between variables
        - Actionable recommendations based on the data
    """
}

def build_data_summary(df):
    '''
    Creates data summary dictonary to feed into the AI inorder to gather dataset insigts
    Returns Dictionary with statistical and numerical summaries
    '''
    summary = {}

    summary["shape"] = {"rows": df.shape[0], "columns": df.shape[1]}

    summary["columns"] = {col: str(dtype) for col, dtype in df.dtypes.items()}

    # Statistical summary for numeric columns
    numeric_summary = df.describe().round(2).to_dict()
    summary["numeric_stats"] = numeric_summary

    # For categorical columns- displays the top 3 most frequent values
    summary["categorical_summary"] = {}
    for col in df.select_dtypes(include=["category", "object"]).columns:
        top_values = df[col].value_counts().head(3).to_dict()
        summary["categorical_summary"][col] = top_values

    # Date range for any datetime columns
    summary["date_ranges"] = {}
    for col in df.select_dtypes(include="datetime").columns:
        summary["date_ranges"][col] = {
            "min": str(df[col].min()),
            "max": str(df[col].max())
        }

    # sample of rows from dataset so AI understands the data structure
    summary["sample_rows"] = df.head(5).to_dict(orient="records")

    return summary


def generate_insights(df, domain="General / Other"):
    '''
    AI generates insights based on sample data and data summary dictionary 
    Returns insights string
    '''
    print("Building data summary...")
    summary = build_data_summary(df)

    # Get the domain specific instructions
    # general/ other is default domain
    domain_context = DOMAIN_PROMPTS.get(domain, DOMAIN_PROMPTS["General / Other"])

    prompt = f"""
    {domain_context}
    You are a senior data analyst presenting findings to a mixed audience 
    of business stakeholders and analysts.
    
    Analyze this dataset summary and provide insights in the following structure:
    
    1. DATASET OVERVIEW
       - What this dataset appears to be about
       - Key dimensions (time period, categories, scale)
    
    2. STATISTICAL HIGHLIGHTS
       - Key statistics worth noting (averages, ranges, distributions)
       - Any statistically interesting patterns in the numbers
    
    3. TRENDS & PATTERNS
       - Notable trends across time, categories, or segments
       - Relationships between variables worth highlighting
    
    4. RECOMMENDATIONS
       - 2-3 actionable suggestions based on the data
       - Keep these practical and grounded in the numbers
    
    Writing style:
    - Mix plain English explanations with specific numbers
    - Bold key findings using **bold**
    - Keep each section concise — 3 to 5 bullet points max
    - Avoid jargon where possible but don't oversimplify
    
    Dataset Summary:
    {json.dumps(summary, indent=2, default=str)}
    """

    try:
        print("Generating insights with AI...")
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3 
            #lowring temp allows model to be more factual 
        )

        insights = response.choices[0].message.content.strip()
        print("Insights generated successfully.")
        return insights

    except Exception as e:
        print(f"Error generating insights: {e}")
        return None


def generate_dataset_description(df, domain="General / Other"):
    '''
    Creates a short summary of the dataset 
    returns AI generated data summary string
    '''
    sample = df.head(3).to_dict(orient="records")
    columns = list(df.columns)

    prompt = f"""
    You are analyzing a {domain} dataset.
    Look at these column names and sample rows.
    Write ONE sentence describing what this dataset is about.
    Be specific — mention the domain, key metrics, and time period if visible.
    No preamble, just the sentence.
    
    Columns: {columns}
    Sample rows: {json.dumps(sample, default=str)}
    """

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Error generating description: {e}")
        return "Dataset loaded successfully."

