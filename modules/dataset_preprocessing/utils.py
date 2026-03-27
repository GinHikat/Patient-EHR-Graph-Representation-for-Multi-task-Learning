import ast
import pandas as pd
import numpy as np

def to_list(df, col):
    def safe_parse(x):
        if pd.isna(x) or x == '':
            return []
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return []
    
    df[col] = df[col].apply(safe_parse)

def title(df, col):
    df[col] = df[col].apply(lambda x: x.title())

