from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import QueryEngineTool
from ai_agents.data_indexing import create_data_query_engine


def create_data_analysis_agent(csv_file_path: str, persist_dir: str = "storage"):
    """Create a data analysis agent with a query engine tool."""
    # Create a query engine
    query_engine = create_data_query_engine(csv_file_path, persist_dir)
    
    # Create a tool from query engine
    query_engine_tool = QueryEngineTool.from_defaults(
        query_engine=query_engine,
        name="commute_data_tool",
        description="""Useful for answering questions about commute times and transportation modes from the dataset.
        
        The dataset contains information about:
        - Different transportation modes (Car, Bus, Cycle, Walk)
        - Distance traveled in miles
        - Commute time in minutes
        
        You can ask about:
        - Average commute times (overall or by mode) with exact numerical values
        - Speed and efficiency comparisons between transportation modes with numerical ratios
        - Correlation analysis between distance and time with specific correlation coefficients
        - Statistical summaries with precise numbers (min, max, median values)
        
        Always query for specific numerical data rather than general statements.
        """
    )
    
    # Create the data analysis agent
    llm = OpenAI(model="gpt-4o-mini-2024-07-18")
    data_analysis_agent = ReActAgent.from_tools(
        [query_engine_tool],
        llm=llm,
        verbose=True,
        system_prompt="""You are a precise data analysis assistant that helps analyze commute time data.

        When answering questions:
        1. ALWAYS provide specific numerical values, statistics, and metrics when available
        2. Include exact averages, counts, percentages, and other numerical data in your answers
        3. When comparing or ranking, provide the actual numerical differences
        4. If correlation is mentioned, include the correlation coefficient
        5. Present data in a clear, organized format
        
        Your answers should be factual, precise, and data-driven, based on the actual numbers in the dataset.
        Avoid vague generalizations like "there is a correlation" - instead say "there is a strong positive correlation of 0.72 between X and Y".
        """
    )
    
    return data_analysis_agent