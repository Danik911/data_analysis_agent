from llama_index.core.workflow import Workflow, Context, StartEvent, step, StopEvent, Event

from ai_agents.agents import data_prep_agent
from ai_agents.tools import analyze_data_quality, clean_data
from events import InputRequiredEvent, DataPreparedEvent, HumanResponseEvent, DataLoadedEvent


class DataPreparationWorkflow(Workflow):

    @step
    async def load_data(self, ctx: Context, ev: StartEvent) -> DataLoadedEvent | InputRequiredEvent:
        """Load CSV data and validate its structure."""
        # Check if filename is provided
        if not hasattr(ev, 'filename'):
            return InputRequiredEvent(
                question="Please provide the CSV filename to analyze:",
                context={},
                step_name="load_data"
            )

        # Use the agent to load the file
        # Replace astream_chat with async_chat
        load_response = await data_prep_agent.async_chat(
            f"Please load the CSV file named {ev.filename} and provide a brief summary of its contents."
        )

        # Since async_chat doesn't stream, we can directly get the response
        result = str(load_response)

        # Extract DataFrame from context
        df = await ctx.get("current_dataframe")
        if df is None:
            return InputRequiredEvent(
                question=f"I couldn't load the file {ev.filename}. Please provide a valid CSV filename:",
                context={"error": "File not found or invalid format"},
                step_name="load_data"
            )

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

    @step
    async def handle_input(self, ctx: Context, ev: InputRequiredEvent) -> HumanResponseEvent:
        """Handle user input requests and convert them to events the workflow can use."""
        # Write the question to the stream so the user can see it
        ctx.write_event_to_stream(ev)

        # Wait for the human response
        human_response = await ctx.wait_for_event(
            HumanResponseEvent,
            requirements={"step_name": ev.step_name}
        )

        # Return the human response directly
        return human_response

    @step
    async def prepare_data(self, ctx: Context,
                           ev: DataLoadedEvent | HumanResponseEvent) -> DataPreparedEvent | InputRequiredEvent:
        """Clean and prepare data for analysis."""
        # If this is a human response
        if isinstance(ev, HumanResponseEvent) and ev.step_name == "prepare_data":
            # Process human response
            # Get the current data quality issues
            data_quality = await ctx.get("data_quality")
            cleaning_actions = eval(ev.response)  # Be careful with eval in production

            # Use the agent to clean the data
            df = await ctx.get("current_dataframe")
            cleaned_df, cleaning_summary = await clean_data(ctx, df, cleaning_actions)

            # Save cleaned dataframe to context
            await ctx.set("current_dataframe", cleaned_df)

            return DataPreparedEvent(
                dataframe=cleaned_df,
                metadata={"columns": list(cleaned_df.columns), "rows": len(cleaned_df)},
                cleaning_summary=cleaning_summary
            )

        # If this is from the data loading step
        df = ev.dataframe

        # Use the agent to analyze data quality
        quality_report = await analyze_data_quality(ctx, df)

        # Save data quality and dataframe to context
        await ctx.set("data_quality", quality_report)
        await ctx.set("current_dataframe", df)

        # Create a suggested cleaning plan based on quality issues
        suggested_cleaning = {}

        # Suggest handling missing values
        if quality_report.get("missing_values"):
            suggested_cleaning["handle_missing"] = {}
            for col, count in quality_report["missing_values"].items():
                col_type = quality_report["data_types"].get(col)
                if 'float' in col_type or 'int' in col_type:
                    suggested_cleaning["handle_missing"][col] = "mean"
                else:
                    suggested_cleaning["handle_missing"][col] = "mode"

        # Ask user for confirmation on cleaning actions
        return InputRequiredEvent(
            question="I've analyzed the data and found some issues. Here's a suggested cleaning plan. Please modify as needed:",
            context={
                "data_quality": quality_report,
                "suggested_cleaning": suggested_cleaning
            },
            step_name="prepare_data"
        )

    @step
    async def analyze_data(self, ctx: Context, ev: DataPreparedEvent) -> StopEvent:
        """Analyze the prepared data and finish the workflow."""
        # Store the prepared dataframe for use in subsequent workflows
        await ctx.set("prepared_dataframe", ev.dataframe)

        # Here you would typically do some analysis or call another agent
        # For now we'll just return a success message

        return StopEvent(
            result={
                "status": "success",
                "message": "Data preparation completed successfully",
                "dataframe_shape": f"{ev.metadata['rows']} rows, {len(ev.metadata['columns'])} columns",
                "cleaning_summary": ev.cleaning_summary
            }
        )