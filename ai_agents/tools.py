import pandas as pd
from typing import Any, Dict, Tuple, Optional
from llama_index.core.workflow import Context

async def load_csv(ctx: Context, filename: str) -> Any:
    """Load a CSV file into a pandas DataFrame."""
    try:
        df = pd.read_csv(filename)
        print(f"Successfully loaded {filename} with {len(df)} rows and {len(df.columns)} columns")
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

async def analyze_data_quality(ctx: Context, dataframe: Any) -> Dict:
    """Analyze the quality of data in a DataFrame."""
    if dataframe is None:
        return {"error": "No dataframe provided"}

    quality_report = {
        "missing_values": {},
        "data_types": {},
        "unique_values": {},
        "statistics": {}
    }

    # Check for missing values
    missing = dataframe.isnull().sum()
    quality_report["missing_values"] = {col: int(count) for col, count in missing.items() if count > 0}

    # Check data types
    quality_report["data_types"] = {col: str(dtype) for col, dtype in dataframe.dtypes.items()}

    # Check unique values for categorical columns
    for col in dataframe.select_dtypes(include=['object', 'category']).columns:
        if len(dataframe[col].unique()) < 10:  # Only for columns with few unique values
            quality_report["unique_values"][col] = dataframe[col].value_counts().to_dict()

    # Basic statistics for numeric columns
    for col in dataframe.select_dtypes(include=['number']).columns:
        quality_report["statistics"][col] = {
            "min": float(dataframe[col].min()),
            "max": float(dataframe[col].max()),
            "mean": float(dataframe[col].mean()),
            "median": float(dataframe[col].median()),
            "std": float(dataframe[col].std())
        }

    return quality_report

async def clean_data(ctx: Context, dataframe: Any, cleaning_actions: Dict) -> Tuple[Optional[Any], Dict]:
    """Clean a dataframe based on specified cleaning actions."""
    if dataframe is None:
        return None, {"error": "No dataframe provided"}

    df = dataframe.copy()
    cleaning_summary = {}

    # Handle missing values
    if "handle_missing" in cleaning_actions:
        for col, action in cleaning_actions["handle_missing"].items():
            if action == "drop":
                before_count = len(df)
                df = df.dropna(subset=[col])
                after_count = len(df)
                cleaning_summary[f"dropped_rows_{col}"] = before_count - after_count
            elif action == "mean":
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].mean())
                    cleaning_summary[f"filled_mean_{col}"] = int(dataframe[col].isnull().sum())
            elif action == "median":
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())
                    cleaning_summary[f"filled_median_{col}"] = int(dataframe[col].isnull().sum())
            elif action == "mode":
                df[col] = df[col].fillna(df[col].mode()[0])
                cleaning_summary[f"filled_mode_{col}"] = int(dataframe[col].isnull().sum())
            elif action == "zero":
                df[col] = df[col].fillna(0)
                cleaning_summary[f"filled_zero_{col}"] = int(dataframe[col].isnull().sum())

    # Convert data types
    if "convert_types" in cleaning_actions:
        for col, new_type in cleaning_actions["convert_types"].items():
            try:
                df[col] = df[col].astype(new_type)
                cleaning_summary[f"converted_{col}"] = new_type
            except:
                cleaning_summary[f"failed_convert_{col}"] = new_type

    # Remove outliers
    if "remove_outliers" in cleaning_actions:
        for col in cleaning_actions["remove_outliers"]:
            if pd.api.types.is_numeric_dtype(df[col]):
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                before_count = len(df)
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                after_count = len(df)
                cleaning_summary[f"removed_outliers_{col}"] = before_count - after_count

    # Rename columns
    if "rename_columns" in cleaning_actions:
        df = df.rename(columns=cleaning_actions["rename_columns"])
        cleaning_summary["renamed_columns"] = cleaning_actions["rename_columns"]

    # Drop columns
    if "drop_columns" in cleaning_actions:
        df = df.drop(columns=cleaning_actions["drop_columns"])
        cleaning_summary["dropped_columns"] = cleaning_actions["drop_columns"]

    return df, cleaning_summary