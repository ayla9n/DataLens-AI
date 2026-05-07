import pandas as pd
import numpy as np
from google import genai
import json
import os
import streamlit as st
from dotenv import load_dotenv

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
        "gemini-2.0-flash",
        "gemini-2.5-flash"     # fallback
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



def standardize_columns(df):
    '''
    Cleans column names : strips whitespace, lowercases, replaces spaces
    with underscores and removes special characters
    Returns df
    '''
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace(r"[^a-z0-9_]", "", regex=True)
    )
    return df


def detect_column_types(df):
    '''
    Sends column names and sample values to Groq AI
    Returns column classifications and date formats for datetime columns
    '''
    column_info = {}

    for col in df.columns:
        sample_values = df[col].dropna().head(5).tolist()
        column_info[col] = sample_values

    prompt = f"""
    You are a data analyst. Classify each column and detect date formats in one pass.
    
    Classify each column as one of:
    "datetime", "year", "category", "numeric", "text", "identifier", "skip"
    
    Guidelines:
    - "identifier": looks numeric but is actually a code e.g postal_code, zip, phone, row_id
    - "numeric": math makes sense e.g sales, quantity, age
    - "category": repeated finite values e.g region, segment, ship_mode
    - "datetime": date or timestamp columns
    - "year": year-only columns that stay as integers
    - "text": free form text e.g product_name, customer_name
    - "skip": no useful information
    
    For any column classified as "datetime", also detect its format from the samples.
    
    Common date formats:
    - "%d/%m/%Y" for 18/12/2017
    - "%m/%d/%Y" for 12/18/2017
    - "%Y-%m-%d" for 2017-12-18
    - "%d-%m-%Y" for 18-12-2017
    - "%d/%m/%y" for 18/12/17
    - "%m/%d/%y" for 12/18/17
    - "%Y/%m/%d" for 2017/12/18
    - "%d %b %Y" for 18 Dec 2017
    - "%B %d, %Y" for December 18, 2017
    
    Columns and sample values:
    {json.dumps(column_info, indent=2, default=str)}
    
    Return ONLY a valid JSON object in this exact structure, no explanation, no markdown:
    {{
        "column_types": {{
            "column_name": "datetime",
            "another_column": "numeric"
        }},
        "date_formats": {{
            "column_name": "%d/%m/%Y"
        }}
    }}
    
    Only include columns in "date_formats" if they are classified as "datetime".
    """

    try:
        raw = call_gemini(prompt)
        raw = raw.replace("```json", "").replace("```", "")
        result = json.loads(raw)

        column_types = result.get("column_types", {})
        date_formats = result.get("date_formats", {})

        print(f"Column types: {column_types}")
        print(f"Date formats: {date_formats}")

        return column_types, date_formats

    except json.JSONDecodeError:
        print("Warning: AI response could not be parsed as JSON. Skipping AI detection.")
        return {}, {}

    except Exception as e:
        print(f"Warning: AI column detection failed: {e}")
        return {}, {}


def apply_column_types(df, col_types, date_formats={}):
    '''
    Applies AI column classifications to the dataframe
    Uses detected date format for datetime columns
    Returns df
    '''
    for col, dtype in col_types.items():
        if col not in df.columns:
            continue

        try:
            match dtype:
                case "datetime":
                    fmt = date_formats.get(col)
                    if fmt:
                        parsed = pd.to_datetime(df[col], format=fmt, errors="coerce")
                        null_count = parsed.isnull().sum()
                        if null_count > len(df) * 0.1:
                            print(f"  Format {fmt} had {null_count} NaTs, trying auto...")
                            parsed = pd.to_datetime(df[col], errors="coerce")
                        df[col] = parsed
                    else:
                        df[col] = pd.to_datetime(df[col], errors="coerce")

                case "year":
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

                case "identifier":
                    if pd.api.types.is_float_dtype(df[col]):
                        df[col] = df[col].fillna(-1).astype(int).astype(str)
                        df[col] = df[col].replace("-1", "Unknown")
                    else:
                        df[col] = df[col].astype(str)

                case "category":
                    df[col] = df[col].astype("category")

                case "numeric":
                    df[col] = pd.to_numeric(df[col], errors="coerce")

                case "text":
                    df[col] = df[col].astype(str)

                case "skip" | _:
                    pass

        except Exception as e:
            print(f"Could not convert column '{col}': {e}")

    return df


def drop_duplicates(df):
    '''
    Removes exact duplicate rows and prints how many were dropped
    returns df
    '''
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"Removed {before - after} duplicate rows")
    return df

def handle_missing_values(df):
    '''
    Fills missing values based on column type 
    returns df
    '''
    for col in df.columns:
        missing_count = df[col].isnull().sum()

        if missing_count == 0:
            continue

        print(f"Filling {missing_count} missing values in '{col}'")

        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())

        elif isinstance(df[col].dtype, pd.CategoricalDtype):
            df[col] = df[col].fillna(df[col].mode()[0])

        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            pass

        else:
            df[col] = df[col].fillna("Unknown")

    return df



def clean_data(filepath):
    '''
    main pipeline — loads CSV, runs all cleaning steps 
    returns cleaned dataframe with summary
    '''
    print("Loading data...")

    if not os.path.exists(filepath):
        print(f"Error: File not found at '{filepath}'")
        return None, {}

    # tries UTF-8 first
    try:
        df = pd.read_csv(filepath, encoding="utf-8")
    except UnicodeDecodeError:
        print("UTF-8 decoding failed, trying latin-1...")
        try:
            df = pd.read_csv(filepath, encoding="latin-1")
        except UnicodeDecodeError:
            print("latin-1 failed, trying cp1252...")
            df = pd.read_csv(filepath, encoding="cp1252")

    if df.empty:
        print("Error: File is empty")
        return None, {}

    stats = {
        "original_rows": df.shape[0],
        "original_cols": df.shape[1],
        "original_missing": int(df.isnull().sum().sum()),
        "duplicates_removed": 0,
        "missing_filled": 0,
    }

    print(f"Original shape: {df.shape}")

    print("\nStandardizing column names...")
    df = standardize_columns(df)

    print("\nDetecting column types with AI...")
    column_types, date_formats = detect_column_types(df)

    if not column_types:
        print("Skipping AI type conversion, using pandas defaults...")
    else:
        df = apply_column_types(df, column_types, date_formats)

    print("\nDropping duplicates...")
    before_dedup = len(df)
    df = drop_duplicates(df)
    stats["duplicates_removed"] = before_dedup - len(df)

    print("\nHandling missing values...")
    before_missing = int(df.isnull().sum().sum())
    df = handle_missing_values(df)
    stats["missing_filled"] = before_missing

    stats["cleaned_rows"] = df.shape[0]
    stats["cleaned_cols"] = df.shape[1]
    stats["cleaned_missing"] = int(df.isnull().sum().sum())

    print(f"\nCleaned shape: {df.shape}")
    print("Cleaning complete.")
    return df, stats