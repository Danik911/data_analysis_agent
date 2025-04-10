from llama_index.core import Settings
from llama_index.core.agent import ReActAgent  # Add this line
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.llms.openai import OpenAI
import pandas as pd

from ai_agents.data_indexing import create_data_query_engine

# Initialize the OpenAI LLM
llm = OpenAI(model="gpt-3.5-turbo")
Settings.llm = llm

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
    tools=[]
)

def create_data_analysis_agent(csv_file_path: str, persist_dir: str = "commute_index"):
    """Create an agent specialized for data analysis."""
    
    # Create a query engine for the data
    query_engine = create_data_query_engine(csv_file_path, persist_dir)
    
    # Create a tool from query engine
    query_engine_tool = QueryEngineTool.from_defaults(
        query_engine=query_engine,
        name="data_analysis_tool",
        description="""Use this tool to analyze data and answer questions about the dataset.
        The tool provides access to the entire dataset and can perform statistical analysis,
        identify patterns, generate visualizations, and answer specific queries.
        """
    )
    
    # Add a specialized function to compute transportation mode efficiency
    def compute_mode_efficiency(query: str) -> str:  # Changed from "_" to "query"
        """Calculate each mode's average speed from the CSV and rank them."""
        df = pd.read_csv(csv_file_path)
        # Drop any rows with zero or NaN time to avoid division by zero
        df = df[df["Time"] > 0]
        df["Speed"] = df["Distance"] / (df["Time"] / 60.0)  # miles/min * 60 = mph

        # Group by Mode to get average speeds
        avg_speeds = df.groupby("Mode")["Speed"].mean().sort_values(ascending=False)
        
        # Build a neat answer string
        answer_lines = []
        for i, (mode, speed) in enumerate(avg_speeds.items(), start=1):
            answer_lines.append(f"{i}. {mode}: {speed:.2f} mph")
        return "Modes ranked by average speed:\n" + "\n".join(answer_lines)
    
    # Create a specialized tool for transportation efficiency
    compute_mode_efficiency_tool = FunctionTool.from_defaults(
        fn=compute_mode_efficiency,
        name="transportation_efficiency_tool",
        description="Calculates and ranks transportation modes by efficiency (speed in mph)"
    )
    
    # Initialize the LLM
    llm = OpenAI(model="gpt-4o-mini-2024-07-18")
    
    # Create the analysis agent with the transportation efficiency tool
    analysis_agent = ReActAgent.from_tools(
        [query_engine_tool, compute_mode_efficiency_tool],
        llm=llm,
        verbose=True,
        system_prompt="""You are a precise data analysis assistant focused purely on analysis.
        
        Your key responsibilities are:
        1. Analyzing prepared data to extract insights
        2. Answering specific questions with precise numerical values
        3. Identifying trends, patterns, and relationships in the data
        4. Providing statistical measures and confidence levels
        
        When analyzing data:
        - Always provide specific numerical values in your answers
        - Include exact calculations, not approximations
        - Show your reasoning process when conducting complex analysis
        - Present results clearly with appropriate context
        
        IMPORTANT: When asked about transportation mode efficiency or which mode is most efficient,
        ALWAYS use the transportation_efficiency_tool to calculate actual speeds from the dataset.
        
        You work with data that has already been cleaned and prepared by a separate agent,
        so focus entirely on extracting meaningful insights from the data.
        """
    )
    
    return analysis_agent