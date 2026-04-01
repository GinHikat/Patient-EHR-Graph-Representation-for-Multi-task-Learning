import pandas as pd
import numpy as np
import io
from neo4j import GraphDatabase
from typing import List, Dict, Any, Optional
import tempfile
import subprocess
import os, sys
import json
import time

from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
load_dotenv(env_path)

#For neo4j
URI = os.getenv('NEO4J_URI')
USERNAME = os.getenv('NEO4J_USERNAME', 'neo4j')
AUTH = (USERNAME, os.getenv('NEO4J_AUTH'))
DATABASE = os.getenv('NEO4J_DATABASE', 'neo4j')

driver = GraphDatabase.driver(URI, auth=AUTH, keep_alive=True)

def query_neo4j(query: str, **params):
    """
    Function to query the Neo4j db, used for Query and retrieving data
    """
    records, summary, keys = driver.execute_query(
        query,
        **params,  #Expect params to be in dictionary format
        database=DATABASE
    ) #type: ignore
    return [record.data() for record in records]

def dml_ddl_neo4j(query: str, progress = True, implicit = False, **params):
    """
    Safe DML/DDL executor for Neo4j Aura.
    Auto-retries on transient connection failures.
    Creates short-lived sessions to avoid Aura's 5-minute idle timeout.
    
    Args:
        query: The Cypher query.
        progress: Whether to print operation statistics.
        implicit: Set to True for batch queries using 'CALL ... IN TRANSACTIONS'.
        **params: Parameters to include in the query.
    """

    MAX_RETRY = 5
    RETRY_DELAY = 1  # seconds

    for attempt in range(1, MAX_RETRY + 1):
        try:
            # Use a fresh session for every write to avoid Aura idle timeout
            with driver.session(database=DATABASE) as session:
                if implicit:
                    # 'CALL ... IN TRANSACTIONS' requires an implicit (auto-commit) transaction.
                    # Bypassing execute_write ensures no explicit transaction is started.
                    result = session.run(query, **params).consume()
                else:
                    result = session.execute_write(
                        lambda tx: tx.run(query, **params).consume()
                    )

            c = result.counters
            if progress == True:
                print(
                    f"Created {c.nodes_created} nodes, "
                    f"{c.relationships_created} rels "
                    f"in {result.result_available_after} ms."
                )
            return result

        except Exception as e:
            err = str(e)

            # All errors Aura produces when the connection is killed
            transient_errors = [
                "WinError 10054",
                "ConnectionResetError",
                "ServiceUnavailable",
                "SessionExpired",
                "TransientError",
                "IncompleteCommit",
                "Read from defunct connection",
                "BoltConnectionFatality",
            ]

            if any(t in err for t in transient_errors):
                print(f"[Retry {attempt}/{MAX_RETRY}] Transient error:", err)
                time.sleep(RETRY_DELAY)
                continue  # retry the loop

            # Non-retryable error → raise immediately
            raise

    raise RuntimeError(
        f"Neo4j query failed after {MAX_RETRY} retries.\nQuery:\n{query}"
    )

#File format changes
def save_to_txt(text: str, file_name: str):
    with open(f"D:/Study/Education/Projects/Group_Project/source/document/text_format/{file_name}.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print("File saved as output.txt")

def print_tree(folder_path, prefix=""):
    '''
    Print the folder files and subfolders in hierarchy tree format

    Input:
        folder_path: path to folder
    
    Output:
        print the tree structure
    '''

    try:
        items = sorted(os.listdir(folder_path))
    except PermissionError:
        print(prefix + "└── [Permission Denied]")
        return

    for i, item in enumerate(items):
        path = os.path.join(folder_path, item)
        connector = "└── " if i == len(items) - 1 else "├── "

        print(prefix + connector + item)

        if os.path.isdir(path):
            extension = "    " if i == len(items) - 1 else "│   "
            print_tree(path, prefix + extension)
