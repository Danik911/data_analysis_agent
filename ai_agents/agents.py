from llama_index.core import Settings
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from statsmodels.datasets.utils import load_csv
from ai_agents.tools import *

# Create function tools
load_csv_tool = FunctionTool.from_defaults(fn=load_csv)
analyze_data_quality_tool = FunctionTool.from_defaults(fn=analyze_data_quality)
clean_data_tool = FunctionTool.from_defaults(fn=clean_data)

# Initialize the OpenAI LLM
llm = Settings.llm

# Create the data preparation agent
data_prep_agent = FunctionAgent(
    name="DataPrepAgent",
    description="Prepares and cleans data for analysis",
    system_prompt="""You are a data preparation agent that helps clean and prepare CSV data for analysis.
    You can load CSV files, analyze data quality, and perform cleaning operations.
    When you find issues in the data, explain them clearly and recommend appropriate cleaning actions.
    Be thorough in your analysis but prioritize the most important issues.
    """,
    llm=llm,
    tools=[load_csv_tool, analyze_data_quality_tool, clean_data_tool]
)