from ai_agents.data_preparation_workflow import DataPreparationWorkflow


async def run_data_preparation_pipeline(filename):
    """Run the data preparation pipeline with human-in-the-loop interactions."""
    # Initialize the workflow
    workflow = DataPreparationWorkflow(timeout=300)

    # Start the workflow
    handler = workflow.run(filename=filename)

    # Handle events
    async for event in handler.stream_events():
        if isinstance(event, InputRequiredEvent):
            # Display question to the user
            print(f"\n[QUESTION] {event.question}")

            # Display context if available
            if event.context:
                print("\nContext:")
                for key, value in event.context.items():
                    print(f"- {key}: {value}")

            # Get user response
            response = input("\nYour response: ")

            # Send response back to the workflow
            handler.ctx.send_event(
                HumanResponseEvent(
                    response=response,
                    step_name=event.step_name
                )
            )

    # Get final result
    result = await handler
    print("\nWorkflow completed!")
    print(f"Data preparation summary: {result.get('cleaning_summary', {})}")

    # Check if we have a dataframe to print shape
    prepared_df = await handler.ctx.get("prepared_dataframe")
    if prepared_df is not None:
        print(f"Processed dataframe shape: {prepared_df.shape}")

    return result