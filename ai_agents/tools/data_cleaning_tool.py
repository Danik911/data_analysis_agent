import pandas as pd
import numpy as np
import io
from typing import Dict, List, Any, Tuple, Optional
from llama_index.core.tools import FunctionTool

async def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load a dataset from a file path.
    """
    try:
        file_extension = file_path.split('.')[-1].lower()
        if file_extension == 'csv':
            df = pd.read_csv(file_path)
        elif file_extension in ['xls', 'xlsx']:
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")
        return df
    except Exception as e:
        raise Exception(f"Error loading dataset: {str(e)}")

async def get_dataset_overview(df: Optional[pd.DataFrame] = None) -> Dict:
    """
    Get an overview of the dataset including head, info, and shape.
    """
    if df is None:
        return {"error": "No dataframe available"}
    
    # Convert DataFrame.info() to string representation
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_string = buffer.getvalue()
    
    return {
        "head": df.head().to_dict(),
        "shape": df.shape,
        "columns": list(df.columns),
        "info": info_string,
    }

async def generate_descriptive_stats(df: Optional[pd.DataFrame] = None) -> Dict:
    """
    Generate descriptive statistics for the dataset.
    """
    if df is None:
        return {"error": "No dataframe available"}
    
    stats = {}
    
    # Basic statistics for numerical columns
    numeric_stats = df.describe().to_dict()
    stats["numeric"] = numeric_stats
    
    # For categorical columns, get counts and frequencies
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    cat_stats = {}
    for col in categorical_cols:
        value_counts = df[col].value_counts()
        frequency = df[col].value_counts(normalize=True)
        cat_stats[col] = {
            "counts": value_counts.to_dict(),
            "frequency": {k: float(v) for k, v in frequency.to_dict().items()}
        }
    stats["categorical"] = cat_stats
    
    return stats

async def check_missing_values(df: Optional[pd.DataFrame] = None) -> Dict:
    """
    Check for missing values in the dataset.
    """
    if df is None:
        return {"error": "No dataframe available"}
    
    missing_values = df.isnull().sum().to_dict()
    missing_percentage = (df.isnull().mean() * 100).to_dict()
    
    return {
        "missing_count": {k: int(v) for k, v in missing_values.items() if v > 0},
        "missing_percentage": {k: float(v) for k, v in missing_percentage.items() if v > 0},
        "total_missing_rows": int(df.isnull().any(axis=1).sum()),
        "total_rows": len(df)
    }

async def check_duplicates(df: Optional[pd.DataFrame] = None) -> Dict:
    """
    Check for duplicate rows in the dataset.
    """
    if df is None:
        return {"error": "No dataframe available"}
    
    duplicate_rows = df.duplicated().sum()
    duplicates_full = df[df.duplicated(keep=False)]
    
    return {
        "duplicate_count": int(duplicate_rows),
        "duplicate_percentage": float(duplicate_rows / len(df) * 100),
        "sample_duplicates": duplicates_full.head(5).to_dict() if duplicate_rows > 0 else {}
    }

async def identify_outliers(df: Optional[pd.DataFrame] = None, method: str = "iqr", columns: List[str] = None) -> Dict:
    """
    Identify outliers in numerical columns using IQR or Z-score method.
    
    Args:
        method: 'iqr' for Interquartile Range or 'zscore' for Z-score method
        columns: List of columns to check for outliers. If None, checks all numerical columns.
    """
    if df is None:
        return {"error": "No dataframe available"}
    
    numeric_cols = df.select_dtypes(include=['number']).columns if columns is None else [col for col in columns if col in df.select_dtypes(include=['number']).columns]
    
    outliers = {}
    outlier_indices = set()
    
    for col in numeric_cols:
        col_outliers = {}
        if method.lower() == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            col_outliers_indices = df[outlier_mask].index.tolist()
            col_outliers_values = df.loc[outlier_mask, col].tolist()
            
            col_outliers = {
                "method": "IQR",
                "bounds": {"lower": float(lower_bound), "upper": float(upper_bound)},
                "count": int(outlier_mask.sum()),
                "indices": col_outliers_indices[:10],  # Limiting to 10 indices for brevity
                "values": col_outliers_values[:10]  # Limiting to 10 values for brevity
            }
            
            outlier_indices.update(col_outliers_indices)
            
        elif method.lower() == 'zscore':
            from scipy import stats
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            threshold = 3
            outlier_mask = z_scores > threshold
            
            # Get indices of the original dataframe that correspond to outliers
            valid_indices = df[col].dropna().index
            col_outliers_indices = valid_indices[outlier_mask].tolist()
            col_outliers_values = df.loc[col_outliers_indices, col].tolist()
            
            col_outliers = {
                "method": "Z-score",
                "threshold": threshold,
                "count": int(sum(outlier_mask)),
                "indices": col_outliers_indices[:10],  # Limiting to 10 indices for brevity
                "values": col_outliers_values[:10]  # Limiting to 10 values for brevity
            }
            
            outlier_indices.update(col_outliers_indices)
        
        outliers[col] = col_outliers
    
    return {
        "outliers_by_column": outliers,
        "total_outlier_rows": len(outlier_indices)
    }

async def handle_duplicates(action: str, df: Optional[pd.DataFrame] = None) -> Dict:
    """
    Handle duplicates based on user input.
    
    Args:
        action: One of 'keep_first', 'keep_last', 'keep_none', or 'ignore'
    """
    if df is None:
        return {"error": "No dataframe available"}
    
    original_shape = df.shape
    
    if action.lower() == 'keep_first':
        df = df.drop_duplicates(keep='first')
    elif action.lower() == 'keep_last':
        df = df.drop_duplicates(keep='last')
    elif action.lower() == 'keep_none':
        df = df.drop_duplicates(keep=False)
    elif action.lower() == 'ignore':
        return {"message": "Duplicates were left in the dataset", "rows_affected": 0}
    else:
        return {"error": f"Unknown action: {action}. Use 'keep_first', 'keep_last', 'keep_none', or 'ignore'."}
    
    rows_removed = original_shape[0] - df.shape[0]
    
    return {
        "action": action,
        "original_rows": original_shape[0],
        "new_rows": df.shape[0],
        "rows_removed": rows_removed
    }

async def handle_outliers(action: str, method: str = "iqr", columns: List[str] = None, df: Optional[pd.DataFrame] = None) -> Dict:
    """
    Handle outliers based on user input.
    
    Args:
        action: One of 'remove', 'cap', 'replace_mean', 'replace_median', or 'ignore'
        method: 'iqr' or 'zscore' for outlier detection method
        columns: List of columns to process. If None, processes all numerical columns.
    """
    if df is None:
        return {"error": "No dataframe available"}
    
    # First identify outliers
    outlier_results = await identify_outliers(df, method, columns)
    
    if "error" in outlier_results:
        return outlier_results
    
    if action.lower() == 'ignore':
        return {"message": "Outliers were left in the dataset", "rows_affected": 0}
    
    # Get columns with outliers
    cols_with_outliers = [col for col, data in outlier_results["outliers_by_column"].items() if data["count"] > 0]
    
    if not cols_with_outliers:
        return {"message": "No outliers found to handle", "rows_affected": 0}
    
    processed_columns = {}
    df_modified = df.copy()
    
    for col in cols_with_outliers:
        if method.lower() == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            outlier_indices = df[outlier_mask].index
        else:  # zscore
            from scipy import stats
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            threshold = 3
            valid_indices = df[col].dropna().index
            outlier_indices = valid_indices[z_scores > threshold]
        
        if action.lower() == 'remove':
            # We'll collect all indices and remove rows at the end
            processed_columns[col] = {"action": "remove", "indices": outlier_indices.tolist(), "count": len(outlier_indices)}
        
        elif action.lower() == 'cap':
            if method.lower() == 'iqr':
                df_modified.loc[df[col] < lower_bound, col] = lower_bound
                df_modified.loc[df[col] > upper_bound, col] = upper_bound
            else:  # zscore
                mean = df[col].mean()
                std = df[col].std()
                cap_lower = mean - 3 * std
                cap_upper = mean + 3 * std
                df_modified.loc[df[col] < cap_lower, col] = cap_lower
                df_modified.loc[df[col] > cap_upper, col] = cap_upper
            
            processed_columns[col] = {"action": "cap", "indices": outlier_indices.tolist(), "count": len(outlier_indices)}
        
        elif action.lower() == 'replace_mean':
            mean_value = df[col].mean()
            df_modified.loc[outlier_indices, col] = mean_value
            processed_columns[col] = {"action": "replace_mean", "indices": outlier_indices.tolist(), "count": len(outlier_indices), "value": float(mean_value)}
        
        elif action.lower() == 'replace_median':
            median_value = df[col].median()
            df_modified.loc[outlier_indices, col] = median_value
            processed_columns[col] = {"action": "replace_median", "indices": outlier_indices.tolist(), "count": len(outlier_indices), "value": float(median_value)}
        
        else:
            return {"error": f"Unknown action: {action}. Use 'remove', 'cap', 'replace_mean', 'replace_median', or 'ignore'."}
    
    # If action is remove, we need to remove all outlier rows
    if action.lower() == 'remove':
        all_outlier_indices = set()
        for col_data in processed_columns.values():
            all_outlier_indices.update(col_data["indices"])
        
        df_modified = df_modified.drop(list(all_outlier_indices))
    
    return {
        "action": action,
        "original_rows": df.shape[0],
        "new_rows": df_modified.shape[0],
        "processed_columns": processed_columns
    }

async def ask_user_action(question: str, options: List[str]) -> str:
    """
    Ask the user for input with a specific question and options.
    This implements the human-in-the-loop design.
    
    Args:
        question: Question to ask the user
        options: List of possible options user can choose from
    
    Returns:
        User's response
    """
    print(f"\n{question}")
    for i, option in enumerate(options, 1):
        print(f"{i}. {option}")
    
    choice = None
    while choice is None:
        try:
            user_input = input("Enter your choice (number): ")
            choice_num = int(user_input)
            if 1 <= choice_num <= len(options):
                choice = options[choice_num - 1]
            else:
                print(f"Please enter a number between 1 and {len(options)}")
        except ValueError:
            print("Please enter a number")
    
    return choice