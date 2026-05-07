import pandas as pd
import numpy as np
from google import genai
import json
import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()
def get_secret(key):
    try:
        return st.secrets[key]
    except:
        return os.getenv(key)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def call_gemini(prompt, temperature=0.3):
    models = [
        "gemini-3-flash-preview", # primary
        "gemini-2.5-flash",      # fallback
    ]

    for model in models:
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config={"temperature": temperature}
            )
            print(f"Used model: {model}")
            return response.text.strip()

        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                print(f"Rate limit hit on {model}, trying next...")
                continue
            else:
                raise e

    raise Exception("All models rate limited — try again later")

DOMAIN_PROMPTS = {
    "Sales & Revenue": """
        You are a sales analyst. Focus on:
        - Revenue trends over time and across regions or segments
        - Top and bottom performing products or categories
        - Customer purchasing patterns and seasonal trends
        - Actionable recommendations to improve sales performance
    """,

    "Marketing & Campaigns": """
        You are a marketing analyst. Focus on:
        - Campaign performance and return on spend
        - Which channels or audiences are driving the most value
        - Conversion and engagement trends over time
        - Practical recommendations to optimize future campaigns
    """,

    "Finance & Accounting": """
        You are a financial analyst. Focus on:
        - Key financial metrics and what they indicate
        - Expense patterns and areas where costs are growing
        - Period over period comparisons to track financial health
        - Actionable recommendations grounded in the actual numbers
    """
}



def build_data_summary(df):
    '''
    Creates data summary dictionary to feed into the AI
    Returns dictionary with statistical and numerical summaries
    '''
    summary = {}

    summary["shape"] = {"rows": df.shape[0], "columns": df.shape[1]}
    summary["columns"] = {col: str(dtype) for col, dtype in df.dtypes.items()}

    # Statistical summary for numeric columns
    summary["numeric_stats"] = df.describe().round(2).to_dict()

    # Top 3 most frequent values for categorical columns
    summary["categorical_summary"] = {}
    for col in df.select_dtypes(include=["category", "object"]).columns:
        top_values = df[col].value_counts().head(3).to_dict()
        summary["categorical_summary"][col] = top_values

    # Date range for datetime columns
    summary["date_ranges"] = {}
    for col in df.select_dtypes(include="datetime").columns:
        summary["date_ranges"][col] = {
            "min": str(df[col].min()),
            "max": str(df[col].max())
        }

    summary["sample_rows"] = df.head(5).to_dict(orient="records")

    return summary


def generate_dataset_description(df, domain="Sales & Revenue"):
    '''
    Creates a short summary of the dataset and suggests the best domain
    Returns dict with description, domain, and domain_reason
    '''
    sample = df.head(3).to_dict(orient="records")
    columns = list(df.columns)
    domains = list(DOMAIN_PROMPTS.keys())

    prompt = f"""
    You are a data analyst.
    Look at these column names and sample rows from a dataset.
    
    Columns: {columns}
    Sample rows: {json.dumps(sample, default=str)}
    Available domains: {domains}
    
    Return ONLY a valid JSON object, no explanation, no markdown:
    {{
        "description": "One sentence describing what this dataset is about, mentioning key metrics and time period if visible.",
        "domain": "Sales & Revenue",
        "domain_reason": "Brief reason why this domain was selected"
    }}
    """

    try:
        raw = call_gemini(prompt, temperature=0.2)
        raw = raw.replace("```json", "").replace("```", "")
        result = json.loads(raw)


        suggested = result.get("domain", "Sales & Revenue")
        if suggested not in DOMAIN_PROMPTS:
            suggested = "Sales & Revenue"

        return {
            "description": result.get("description", "Dataset loaded successfully."),
            "domain": suggested,
            "domain_reason": result.get("domain_reason", "")
        }

    except Exception as e:
        print(f"Error generating description: {e}")
        return {
            "description": "Dataset loaded successfully.",
            "domain": "Sales & Revenue",
            "domain_reason": ""
        }



def generate_insights(df, domain="Sales & Revenue"):
    '''
    Generates AI insights tailored to the selected domain
    Returns formatted markdown string
    '''
    print("Building data summary...")
    summary = build_data_summary(df)
    domain_context = DOMAIN_PROMPTS.get(domain, DOMAIN_PROMPTS["Sales & Revenue"])

    prompt = f"""
    {domain_context}
    
    Analyze this dataset summary and provide insights in the following structure.
    Use ### for each section heading.
    
    ### Dataset Overview
       - What this dataset is about
       - Key dimensions (time period, categories, scale)
    
    ### Statistical Highlights
       - Key statistics worth noting (averages, ranges, distributions)
       - Interesting patterns in the numbers
    
    ### Trends & Patterns
       - Notable trends across time, categories, or segments
       - Relationships between variables worth highlighting
    
    ### Recommendations
       - 2-3 actionable suggestions based on the data
       - Keep these practical and grounded in the numbers
    
    Writing style:
    - Mix plain English with specific numbers
    - Bold key findings using **bold**
    - Keep each section to 3-5 bullet points max
    - Clear and practical, avoid unnecessary jargon
    
    Dataset Summary:
    {json.dumps(summary, indent=2, default=str)}
    """

    try:
        insights = call_gemini(prompt, temperature=0.3)
        print("Insights generated successfully.")
        return insights

    except Exception as e:
        print(f"Error generating insights: {e}")
        return None



def generate_chart_recommendations(df, domain="Sales & Revenue"):
    '''
    Gets AI chart recommendations and validates column names exist
    Returns filtered dict of valid chart recommendations
    '''
    summary = build_data_summary(df)
    domain_context = DOMAIN_PROMPTS.get(domain, DOMAIN_PROMPTS["Sales & Revenue"])
    column_types = {col: str(dtype) for col, dtype in df.dtypes.items()}
    exact_columns = list(df.columns)

    prompt = f"""
    You are a data visualization expert.
    
    {domain_context}
    
    CRITICAL RULES:
    - You MUST only use column names from this exact list: {exact_columns}
    - Do NOT invent column names like "count of X" or "frequency of Y"
    - If you want to show counts of a category, use bar_chart with x as
      the category column and y as an existing numeric column
    - Only recommend charts that are genuinely useful for this data
    - Quality over quantity — recommend 4 charts MAX
    
    Available chart types:
    - "line_chart": x (datetime col), y (numeric col) — trends over time
    - "bar_chart": x (category col), y (numeric col) — comparing categories
    - "histogram": x (numeric col, no y needed) — distribution of values
    - "scatter_plot": x (numeric col), y (numeric col) — relationships
    - "pie_chart": x (category col), y (numeric col) — % composition
    - "heatmap": no x or y needed — correlations between numeric columns
    
    Exact column names and types you MUST use:
    {json.dumps(column_types, indent=2)}
    
    Dataset summary:
    {json.dumps(summary, indent=2, default=str)}
    
    Return ONLY valid JSON, no explanation, no markdown:
    {{
        "charts": [
            {{
                "type": "bar_chart",
                "x": "channel",
                "y": "revenue_generated",
                "title": "Revenue by Channel",
                "reason": "Compares revenue generated across different channels",
                "takeaway": [
                    "Social Media drives the highest revenue by a significant margin",
                    "Flyer and Paid Search contribute the least"
                ]
            }}
        ]
    }}
    
    For each chart include 1-2 important takeaway points that explain what the chart shows in plain English. Keep each point to one sentence, and make it short and straightfoward. Write for someone who may not be familiar with data analysis.
    """

    try:
        raw = call_gemini(prompt, temperature=0.1)
        raw = raw.replace("```json", "").replace("```", "")
        result = json.loads(raw)

        # Filter out any charts with columns that don't exist
        valid_charts = []
        for chart in result.get("charts", []):
            x = chart.get("x")
            y = chart.get("y")
            chart_type = chart.get("type")

            if chart_type == "heatmap":
                valid_charts.append(chart)
                continue

            if chart_type == "histogram" and x in df.columns:
                valid_charts.append(chart)
                continue

            if x in df.columns and (y is None or y in df.columns):
                valid_charts.append(chart)
            else:
                print(f"Filtered invalid chart: {chart.get('title')}")

        result["charts"] = valid_charts
        print(f"Valid chart recommendations: {result}")
        return result

    except json.JSONDecodeError:
        print("Warning: Could not parse chart recommendations as JSON.")
        return {}

    except Exception as e:
        print(f"Warning: Chart recommendation failed: {e}")
        return {}