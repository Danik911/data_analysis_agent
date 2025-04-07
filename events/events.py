from llama_index.core.workflow import Event, Workflow, step, Context, StartEvent


# Base data event that contains the dataframe
class DataEvent(Event):
    dataframe: object  # Pandas DataFrame
    metadata: dict  # Additional information about the data


# Events for data loading and preparation
class DataLoadedEvent(DataEvent):
    filename: str


class DataPreparedEvent(DataEvent):
    cleaning_summary: dict  # Summary of cleaning operations applied


# Human input events
class InputRequiredEvent(Event):
    question: str  # Question to ask the human
    context: dict  # Additional context
    step_name: str  # Which step is requesting input


class HumanResponseEvent(Event):
    response: str  # Human's response
    step_name: str  # Which step the response is for