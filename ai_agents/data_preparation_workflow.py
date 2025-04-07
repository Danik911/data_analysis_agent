import asyncio
from typing import Optional, Union
import pandas as pd
from llama_index.core.workflow import Workflow, Context, step, Event, StartEvent, StopEvent

from ai_agents.agents import data_prep_agent
from ai_agents.tools import analyze_data_quality, clean_data, load_csv
from ai_agents.events import (
    InputRequiredEvent, DataPreparedEvent, HumanResponseEvent, 
    DataLoadedEvent
)


class DataPreparationWorkflow(Workflow):
    """Workflow for preparing data for analysis."""

    @step
    async def load_data(self, ctx: Context, ev: StartEvent) -> Union[DataLoadedEvent, InputRequiredEvent]:
        """Load CSV data and validate its structure."""
        # Check if filename is provided - now we explicitly accept StartEvent
        if not hasattr(ev, 'filename') or not ev.filename:
            return InputRequiredEvent(
                question="Please provide the CSV filename to analyze:",
                context={},
                step_name="load_data"
            )

        # Use direct tool call instead of agent chat
        try:
            df = await load_csv(ctx, ev.filename)
            await ctx.set("current_dataframe", df)
            
            # Get basic metadata
            metadata = {
                "columns": list(df.columns),
                "rows": len(df),
                "dtypes": {col: str(df[col].dtype) for col in df.columns}
            }
            
            return DataLoadedEvent(
                filename=ev.filename,
                dataframe=df,
                metadata=metadata
            )
        except Exception as e:
            return InputRequiredEvent(
                question=f"I couldn't load the file {ev.filename}. Please provide a valid CSV filename:",
                context={"error": str(e)},
                step_name="load_data"
            )

    @step
    async def analyze_quality(self, ctx: Context, ev: DataLoadedEvent) -> Union[DataPreparedEvent, StopEvent]:
        """Analyze data quality and suggest cleaning operations."""
        df = ev.dataframe
        await ctx.set("current_dataframe", df)
        
        # Get metadata from the input event
        metadata = ev.metadata
        
        # Analyze data quality
        quality_report = await analyze_data_quality(ctx, df)
        
        # Determine cleaning actions based on quality report
        cleaning_actions = await self._suggest_cleaning_actions(quality_report)
        
        if not cleaning_actions:
            return StopEvent(
                reason="No cleaning actions required",
                result=DataPreparedEvent(
                    dataframe=df,
                    cleaning_summary={},
                    quality_report=quality_report,
                    metadata=metadata  # Add missing metadata field
                )
            )
            
        # Clean the data
        cleaned_df, cleaning_summary = await clean_data(ctx, df, cleaning_actions)
        
        return DataPreparedEvent(
            dataframe=cleaned_df,
            cleaning_summary=cleaning_summary,
            quality_report=quality_report,
            metadata=metadata  # Add missing metadata field
        )
        
    async def _suggest_cleaning_actions(self, quality_report: dict) -> dict:
        """Suggest cleaning actions based on quality report."""
        cleaning_actions = {}
        
        # Handle missing values
        if quality_report.get("missing_values"):
            cleaning_actions["handle_missing"] = {}
            for col, count in quality_report["missing_values"].items():
                # Use appropriate strategy based on column type
                col_type = quality_report["data_types"].get(col)
                if "float" in str(col_type) or "int" in str(col_type):
                    cleaning_actions["handle_missing"][col] = "mean"
                else:
                    cleaning_actions["handle_missing"][col] = "mode"
        
        return cleaning_actions


async def run_data_preparation_pipeline(filename: str) -> Optional[DataPreparedEvent]:
    """Run the full data preparation pipeline."""
    workflow = DataPreparationWorkflow()
    ctx = Context(workflow=workflow)  # Fixed: pass the workflow to Context
    
    # Start with loading the data
    event = StartEvent(filename=filename)
    
    while True:
        if isinstance(event, StartEvent):
            event = await workflow.load_data(ctx, event)
            
        elif isinstance(event, DataLoadedEvent):
            event = await workflow.analyze_quality(ctx, event)
            
        elif isinstance(event, InputRequiredEvent):
            # In a notebook/script environment, we'd prompt the user
            # For simplicity, we'll just return None here
            print(f"Input required: {event.question}")
            return None
            
        elif isinstance(event, DataPreparedEvent):
            return event
            
        elif isinstance(event, StopEvent):
            if hasattr(event, 'result') and event.result:
                return event.result
            return None
            
        else:
            print(f"Unexpected event type: {type(event)}")
            return None