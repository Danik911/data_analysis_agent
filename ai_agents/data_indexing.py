import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings,
    load_index_from_storage
)
from llama_index.core.schema import Document
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms.openai import OpenAI


def csv_to_document(file_path: str) -> List[Document]:
    """Convert a CSV file to LlamaIndex Documents with meaningful aggregations."""
    try:
        # Read the CSV
        df = pd.read_csv(file_path)
        
        # Create a list to hold all documents
        documents = []
        
        # 1. Create documents for each row as before
        for idx, row in df.iterrows():
            # Convert row to string format
            content = "\n".join([f"{col}: {val}" for col, val in row.items()])
            
            # Create metadata
            metadata = {
                "row_id": idx,
                "source": file_path,
                "file_type": "csv",
                "doc_type": "row_data"
            }
            
            # Create document
            doc = Document(text=content, metadata=metadata)
            documents.append(doc)
        
        # 2. Create a document with overall statistics
        overall_stats = {
            "total_records": len(df),
            "average_commute_time": df["Time"].mean(),
            "median_commute_time": df["Time"].median(),
            "min_commute_time": df["Time"].min(),
            "max_commute_time": df["Time"].max(),
            "std_commute_time": df["Time"].std(),
            "average_distance": df["Distance"].mean(),
            "median_distance": df["Distance"].median(),
            "correlation_distance_time": df["Distance"].corr(df["Time"])
        }
        
        overall_stats_content = "OVERALL DATASET STATISTICS:\n" + "\n".join([f"{k}: {v}" for k, v in overall_stats.items()])
        overall_stats_doc = Document(
            text=overall_stats_content,
            metadata={
                "source": file_path,
                "file_type": "csv",
                "doc_type": "overall_statistics"
            }
        )
        documents.append(overall_stats_doc)
        
        # 3. Create documents for each transportation mode with their statistics
        for mode in df["Mode"].unique():
            mode_df = df[df["Mode"] == mode]
            mode_stats = {
                "transportation_mode": mode,
                "count": len(mode_df),
                "percentage": len(mode_df) / len(df) * 100,
                "average_commute_time": mode_df["Time"].mean(),
                "median_commute_time": mode_df["Time"].median(),
                "min_commute_time": mode_df["Time"].min(),
                "max_commute_time": mode_df["Time"].max(),
                "average_distance": mode_df["Distance"].mean(),
                "median_distance": mode_df["Distance"].median(),
                "efficiency_ratio": mode_df["Distance"].mean() / mode_df["Time"].mean() if mode_df["Time"].mean() > 0 else 0
            }
            
            mode_content = f"STATISTICS FOR {mode} TRANSPORTATION MODE:\n" + "\n".join([f"{k}: {v}" for k, v in mode_stats.items()])
            mode_doc = Document(
                text=mode_content,
                metadata={
                    "source": file_path,
                    "file_type": "csv",
                    "doc_type": "mode_statistics"
                }
            )
            documents.append(mode_doc)
        
        print(f"Created {len(documents)} documents from {file_path}")
        return documents
    
    except Exception as e:
        print(f"Error converting CSV to documents: {e}")
        return []


def build_index(documents: List[Document], persist_dir: str = "storage") -> VectorStoreIndex:
    """Build a vector index from documents and persist it."""
    # Check if index already exists
    if os.path.exists(persist_dir):
        print(f"Loading existing index from {persist_dir}")
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)
    else:
        print(f"Building new index and saving to {persist_dir}")
        # Configure node parser for how documents are split
        parser = SimpleNodeParser.from_defaults()
        nodes = parser.get_nodes_from_documents(documents)
        
        # Create and save the index
        index = VectorStoreIndex(nodes)
        index.storage_context.persist(persist_dir=persist_dir)
    
    return index


def create_data_query_engine(csv_file_path: str, persist_dir: str = "storage") -> BaseQueryEngine:
    """Create a query engine from a CSV file."""
    # Set the LLM
    Settings.llm = OpenAI(model="gpt-3.5-turbo")
    
    # Convert CSV to documents
    documents = csv_to_document(csv_file_path)
    
    # Build index
    index = build_index(documents, persist_dir)
    
    # Create query engine
    query_engine = index.as_query_engine()
    
    return query_engine