from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool
from ai_agents.data_indexing import create_data_query_engine
import pandas as pd

def create_data_analysis_agent(csv_file_path: str, persist_dir: str = "commute_analysis_index"):
    query_engine = create_data_query_engine(csv_file_path, persist_dir)
    
    # Create a specialized tool to compute speed ranks
    def compute_mode_efficiency(_: str) -> str:
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

    # Create a tool that the agent can call
    efficiency_tool = QueryEngineTool(
        name="efficiency_tool",
        query_engine=None,  # Not using a standard query engine for this.
        description="Compute transportation mode efficiency from the CSV data.",
        func=compute_mode_efficiency,
    )

    llm = ...  # your existing LLM setup

    # Include the custom efficiency tool inside the ReActAgent
    analysis_agent = ReActAgent.from_tools(
        tools=[query_engine, efficiency_tool],
        llm=llm,
        verbose=True,
        system_prompt="""You are a data analysis assistant with access to a dataset containing columns: [Case, Mode, Distance, Time].
When a user asks "Which transportation mode is most efficient?" or similar, you MUST call 'efficiency_tool' to calculate speeds from the dataset instead of guessing.
"""
    )
    return analysis_agent