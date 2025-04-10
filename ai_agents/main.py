import os
import asyncio
import nest_asyncio
from getpass import getpass
from dotenv import load_dotenv
import pandas as pd

# Load environment variables
load_dotenv()
nest_asyncio.apply()

# Set API keys
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY') or getpass("Enter OPENAI_API_KEY: ")
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

from ai_agents.agents.analysis_agent import create_data_analysis_agent
from ai_agents.tools.data_preparation_tools import handle_missing_values, load_csv, analyze_data_quality
from ai_agents.tools.data_cleaning_tool import check_missing_values, check_duplicates, handle_duplicates

async def run_preparation_workflow(file_path: str):
    """Run the data preparation workflow with user interaction for cleaning decisions."""
    try:
        # Load the dataset
        print(f"Loading dataset from {file_path}...")
        df = await load_csv(file_path)
        
        # Check for missing values
        print("\nChecking for missing values...")
        missing_info = await check_missing_values(df)
        print(missing_info)
        
        # If missing values exist, ask user how to handle them
        if missing_info["total_missing_rows"] > 0:
            print("\nHow would you like to handle missing values?")
            strategies = {}
            
            for column in missing_info["missing_count"].keys():
                print(f"\nMissing values in column '{column}': {missing_info['missing_count'][column]} ({missing_info['missing_percentage'][column]:.2f}%)")
                
                if pd.api.types.is_numeric_dtype(df[column]):
                    options = ["mean", "median", "zero", "drop"]
                    print("Options: 1=Fill with mean, 2=Fill with median, 3=Fill with zero, 4=Drop rows")
                else:
                    options = ["mode", "drop"]
                    print("Options: 1=Fill with mode, 2=Drop rows")
                
                choice = input("Enter your choice (number): ")
                strategy = options[int(choice)-1] if choice.isdigit() and 1 <= int(choice) <= len(options) else "drop"
                strategies[column] = strategy
            
            # Apply missing value handling
            df = await handle_missing_values(df, strategies)
        
        # Check for duplicates
        print("\nChecking for duplicates...")
        duplicate_info = await check_duplicates(df)
        print(duplicate_info)
        
        # If duplicates exist, ask user how to handle them
        if duplicate_info["duplicate_count"] > 0:
            print(f"\nFound {duplicate_info['duplicate_count']} duplicate rows ({duplicate_info['duplicate_percentage']:.2f}%)")
            print("How would you like to handle duplicates?")
            print("Options: 1=Keep first occurrence, 2=Keep last occurrence, 3=Remove all duplicates, 4=Keep all duplicates")
            
            options = ["keep_first", "keep_last", "drop_all", "keep_all"]
            choice = input("Enter your choice (number): ")
            strategy = options[int(choice)-1] if choice.isdigit() and 1 <= int(choice) <= len(options) else "keep_first"
            
            # Apply duplicate handling
            df = await handle_duplicates(df, strategy)
        
        # Final data quality check
        print("\nFinal data quality analysis...")
        quality_report = await analyze_data_quality(df)
        
        print("\nData preparation completed successfully.")
        return df
        
    except Exception as e:
        print(f"Error in data preparation workflow: {str(e)}")
        return None

async def main():
    """Main function to run the data analysis pipeline with separate agents."""
    # Define the path to your CSV file
    csv_path = "data/Commute_Times_V1.csv"
    
    # Step 1: Data preparation with user interaction for cleaning decisions
    print(f"Starting data preparation for {csv_path}...")
    cleaned_df = await run_preparation_workflow(csv_path)
    
    # If preparation is successful, continue with analysis
    if cleaned_df is not None:
        print("\nData preparation completed successfully.")
        
        # Save cleaned data for analysis
        cleaned_path = "data/cleaned_commute_times.csv"
        cleaned_df.to_csv(cleaned_path, index=False)
        
        print("\nCreating data analysis agent...")
        analysis_agent = create_data_analysis_agent(cleaned_path, "commute_analysis_index")
        
        print("\nData summary:")
        print(cleaned_df.describe())
        
        # Run some example queries
        print("\nQuestion: What is the average commute time?")
        response = analysis_agent.query("What is the average commute time? Provide the exact numerical value.")
        print(f"Answer: {response}")
        
        print("\nQuestion: Which transportation mode is most efficient?")
        response = analysis_agent.query("Which transportation mode is most efficient? Rank them by efficiency with numerical speed ratios.")
        print(f"Answer: {response}")
        
        print("\nQuestion: Is there a correlation between distance and commute time?")
        response = analysis_agent.query("Is there a correlation between distance and commute time? Provide the correlation coefficient and interpret it.")
        print(f"Answer: {response}")
    else:
        print("Data preparation failed. Check the logs for details.")

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())