import pandas as pd
import numpy as np 
from groq import Groq
import json
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def standardize_columns(df):
    '''
    cleans column names and removes whitespaces and special chars
    returns standardized dataframe
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
    Sends col names to groq AI with API call to classidy columns and date format
    returns column classification dictionary and date format dictionary 
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
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )
        
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "")
        result = json.loads(raw)

       #getting both from  response 
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
    Uses AI column classification to dataframe
    returns updated dataframe 
    '''
    for col, dtype in col_types.items():
        if col not in df.columns:
            continue
        
        try:
            match dtype:
                case "datetime":
                    fmt = date_formats.get(col)  # get AI detected format for this col
                    if fmt:
                        # Use AI detected format first
                        parsed = pd.to_datetime(df[col], format=fmt, errors="coerce")
                        null_count = parsed.isnull().sum()

                        # If too many NaTs fall back to auto detection
                        if null_count > len(df) * 0.1:
                            print(f"  Format {fmt} had {null_count} NaTs, trying auto...")
                            parsed = pd.to_datetime(df[col], errors="coerce")
                        
                        df[col] = parsed
                    else:
                        # No format detected, let pandas figure it out
                        df[col] = pd.to_datetime(df[col], errors="coerce")

                case "year":
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

                case "identifier":
                    if pd.api.types.is_float_dtype(df[col]):
                        #can convert a null val to float so using -1 as place holder for null vals 
                        df[col] = df[col].fillna(-1).astype(int).astype(str)
                        df[col] = df[col].replace("-1", "Unknown")  # restore unknowns
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
    Drops duplicate rows
    returns updated dataframe
    '''
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"Removed {before - after} duplicate rows")
    return df


def handle_missing_values(df): 
    '''
    Fills in missing values based on column types (median for numeric and mode for categorical)
    returns updated df    
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


def remove_outliers(df):
    '''
    removes extreme values from numeric columns using InterQuartile Range (IQR) method 
    values beyond 1.5x the interquartile range above or below is dropped
    returns updated df 
    '''
    numeric_cols = df.select_dtypes(include="number").columns

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        before = len(df)
        df = df[df[col].between(lower_bound, upper_bound)]
        after = len(df)

        if before - after > 0:
            print(f"Outliers removed in '{col}': {before - after} rows")

    return df


def clean_data(filepath):
    '''
    Main pipeline function 
    Loads csv and runs cleaning steps 
    returns cleaned Dataframe  
    '''
    print("Loading data...")
    
    if not os.path.exists(filepath):
        print(f"Error: File not found at '{filepath}'")
        return None
    
    df = pd.read_csv(filepath)
    
    if df.empty:
        print("Error: File is empty")
        return None
    
    print(f"Original shape: {df.shape}")

    print("\nStandardizing column names...")
    df = standardize_columns(df)

    print("\nDetecting column types with AI...")
    column_types, date_formats = detect_column_types(df)

    if not column_types:
        print("Skipping AI type conversion, using pandas defaults...")
    else:
        print(f"AI classified columns as: {column_types}")
        print(f"AI detected date formats: {date_formats}")
        df = apply_column_types(df, column_types, date_formats)

    print("\nDropping duplicates...")
    df = drop_duplicates(df)

    print("\nHandling missing values...")
    df = handle_missing_values(df)

    print("\nRemoving outliers...")
    df = remove_outliers(df)

    print(f"\nCleaned shape: {df.shape}")
    print("Cleaning complete.")
    return df


# if __name__ == "__main__":
#     df = clean_data("./example-datasets/superstore_sales_dataset.csv")
#     print(df.head())