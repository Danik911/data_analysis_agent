import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

async def load_csv(filename: str) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame."""
    try:
        df = pd.read_csv(filename)
        print(f"Successfully loaded {filename} with {len(df)} rows and {len(df.columns)} columns")
        return df
    except Exception as e:
        raise Exception(f"Error loading CSV: {str(e)}")

async def check_missing_values(df: pd.DataFrame) -> Dict:
    """Check for missing values in the dataset and return a detailed report."""
    missing_values = df.isnull().sum().to_dict()
    missing_percentage = (df.isnull().mean() * 100).to_dict()
    
    return {
        "missing_count": {k: int(v) for k, v in missing_values.items() if v > 0},
        "missing_percentage": {k: float(v) for k, v in missing_percentage.items() if v > 0},
        "total_missing_rows": int(df.isnull().any(axis=1).sum()),
        "total_rows": len(df)
    }

async def check_duplicates(df: pd.DataFrame) -> Dict:
    """Check for duplicate rows in the dataset and return a detailed report."""
    duplicate_rows = df.duplicated().sum()
    duplicates_full = df[df.duplicated(keep=False)]
    
    return {
        "duplicate_count": int(duplicate_rows),
        "duplicate_percentage": float(duplicate_rows / len(df) * 100),
        "sample_duplicates": duplicates_full.head(5).to_dict() if duplicate_rows > 0 else {}
    }

async def handle_missing_values(df: pd.DataFrame, strategy: Dict[str, str]) -> pd.DataFrame:
    """
    Handle missing values according to the user-selected strategy.
    
    Args:
        df: DataFrame with missing values
        strategy: Dictionary mapping column names to strategies (mean, median, mode, drop, zero, etc.)
    
    Returns:
        DataFrame with missing values handled
    """
    df_copy = df.copy()
    
    for column, method in strategy.items():
        if column not in df_copy.columns:
            print(f"Warning: Column {column} not found in DataFrame")
            continue
            
        if method == "mean" and pd.api.types.is_numeric_dtype(df_copy[column]):
            df_copy[column] = df_copy[column].fillna(df_copy[column].mean())
            print(f"Filled missing values in {column} with mean: {df_copy[column].mean()}")
            
        elif method == "median" and pd.api.types.is_numeric_dtype(df_copy[column]):
            df_copy[column] = df_copy[column].fillna(df_copy[column].median())
            print(f"Filled missing values in {column} with median: {df_copy[column].median()}")
            
        elif method == "mode":
            mode_value = df_copy[column].mode()[0] if not df_copy[column].mode().empty else None
            df_copy[column] = df_copy[column].fillna(mode_value)
            print(f"Filled missing values in {column} with mode: {mode_value}")
            
        elif method == "zero" and pd.api.types.is_numeric_dtype(df_copy[column]):
            df_copy[column] = df_copy[column].fillna(0)
            print(f"Filled missing values in {column} with zero")
            
        elif method == "drop":
            df_copy = df_copy.dropna(subset=[column])
            print(f"Dropped rows with missing values in {column}. Rows remaining: {len(df_copy)}")
            
        else:
            print(f"Warning: Unsupported method {method} for column {column}")
    
    return df_copy

async def handle_duplicates(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """
    Handle duplicate rows according to the user-selected strategy.
    
    Args:
        df: DataFrame with duplicates
        strategy: One of 'keep_first', 'keep_last', 'drop_all', or 'keep_all'
    
    Returns:
        DataFrame with duplicates handled
    """
    df_copy = df.copy()
    
    if strategy == "keep_first":
        df_copy = df_copy.drop_duplicates(keep='first')
        print(f"Kept first occurrence of duplicates. Rows remaining: {len(df_copy)}")
        
    elif strategy == "keep_last":
        df_copy = df_copy.drop_duplicates(keep='last')
        print(f"Kept last occurrence of duplicates. Rows remaining: {len(df_copy)}")
        
    elif strategy == "drop_all":
        df_copy = df_copy.drop_duplicates(keep=False)
        print(f"Removed all duplicate rows. Rows remaining: {len(df_copy)}")
        
    elif strategy == "keep_all":
        print(f"Kept all duplicates. DataFrame unchanged.")
        
    else:
        print(f"Warning: Unsupported duplicate handling strategy: {strategy}")
    
    return df_copy

async def analyze_data_quality(df: pd.DataFrame) -> Dict:
    """
    Perform comprehensive data quality analysis.
    """
    quality_report = {
        "missing_values": {},
        "data_types": {},
        "unique_values": {},
        "statistics": {}
    }

    # Check missing values
    missing = df.isnull().sum()
    quality_report["missing_values"] = {col: int(count) for col, count in missing.items() if count > 0}

    # Check data types
    quality_report["data_types"] = {col: str(dtype) for col, dtype in df.dtypes.items()}

    # Check unique values for categorical columns
    for col in df.select_dtypes(include=['object', 'category']).columns:
        unique_vals = df[col].nunique()
        quality_report["unique_values"][col] = int(unique_vals)

    # Basic statistics for numeric columns
    for col in df.select_dtypes(include=['number']).columns:
        quality_report["statistics"][col] = {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "mean": float(df[col].mean()),
            "median": float(df[col].median()),
            "std": float(df[col].std())
        }

    return quality_report

async def clean_data(df: pd.DataFrame, cleaning_actions: Dict) -> pd.DataFrame:
    """
    Apply specified cleaning actions to the DataFrame.
    
    Args:
        df: DataFrame to clean
        cleaning_actions: Dictionary of cleaning actions to perform
    
    Returns:
        Cleaned DataFrame
    """
    df_copy = df.copy()
    
    # Handle various cleaning actions as specified in the dictionary
    if "rename_columns" in cleaning_actions:
        df_copy = df_copy.rename(columns=cleaning_actions["rename_columns"])
        
    if "drop_columns" in cleaning_actions:
        df_copy = df_copy.drop(columns=cleaning_actions["drop_columns"])
    
    # Additional cleaning as needed based on the cleaning_actions dictionary
    
    return df_copy