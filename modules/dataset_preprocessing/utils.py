import ast
import pandas as pd
import numpy as np

def to_list(df, col):
    """
    Convert a column of string representations of lists to actual lists.
    
    Args:
        df (pd.DataFrame): The DataFrame to process.
        col (str): The column to convert.
    """
    def safe_parse(x):
        if pd.isna(x) or x == '':
            return []
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return []
    
    df[col] = df[col].apply(safe_parse)

def title(df, col):
    '''
    Convert a column of strings to title case.
    
    Args:
        df (pd.DataFrame): The DataFrame to process.
        col (str): The column to convert.
    '''
    df[col] = df[col].apply(lambda x: x.title())

def safe_parse(x):
    '''
    Safely parse a string to a list.
    
    Args:
        x (str): The string to parse.
    
    Returns:
        list: The parsed list.
    '''
    if not isinstance(x, str) or x.strip() in ('', 'nan', 'None'):
        return []
    try:
        result = ast.literal_eval(x)
        return result if isinstance(result, list) else [result]
    except (ValueError, SyntaxError):
        return [x]  