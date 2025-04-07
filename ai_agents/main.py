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

from ai_agents.data_preparation_workflow import run_data_preparation_pipeline
from ai_agents.analysis_engine import create_data_analysis_agent


async def main():
    """Main function to run the data analysis pipeline."""
    # Define the path to your CSV file
    csv_path = "data/Commute_Times_V1.csv"
    
    print(f"Starting data preparation for {csv_path}...")
    
    # Run data preparation workflow
    result = await run_data_preparation_pipeline(csv_path)
    
    if result:
        df = result.dataframe
        print("\nFirst 5 rows of prepared data:")
        print(df.head())
        
        print("\nData summary:")
        print(df.describe())
        
        # Create an analysis agent with the index
        print("\nCreating data analysis agent...")
        agent = create_data_analysis_agent(csv_path, "commute_times_index")
        
        # Run some example queries to demonstrate the agent's capabilities
        print("\nQuestion: What is the average commute time?")
        response = agent.query("What is the average commute time? Provide the exact numerical value.")
        print(f"Answer: {response}")
        
        print("\nQuestion: Which transportation mode is most efficient?")
        response = agent.query("Which transportation mode is most efficient? Rank them by efficiency with numerical speed ratios.")
        print(f"Answer: {response}")
        
        print("\nQuestion: Is there a correlation between distance and commute time?")
        response = agent.query("Is there a correlation between distance and commute time? Provide the correlation coefficient and interpret it.")
        print(f"Answer: {response}")
    else:
        print("Data preparation failed. Check the logs for details.")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())