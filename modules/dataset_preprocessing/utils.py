import ast
import pandas as pd
import numpy as np
import sys, os
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

data_dir = 'F:/Din/Study/Education/Projects/Thesis/data' # don't mind this hardcoded path lol

mimic_path = os.path.join(data_dir, 'mimic_iv')

hosp = os.path.join(mimic_path, 'hosp')

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

def data_preview(type, name):

    '''
    Preview only a subset of the dataframe for large dataframes, used for stream processing
    '''

    sample_path = os.path.join(mimic_path, type, f'{name}.csv')

    con = duckdb.connect()

    query = f"""
    SELECT * 
    FROM read_csv_auto('{sample_path}',
    ignore_errors = True
    )
    """

    result = con.execute(query)

    chunk = result.fetch_df_chunk(500)

    return chunk

def read_full(type, name):

    sample_path = os.path.join(mimic_path, type, f'{name}.csv')

    df = pd.read_csv(sample_path)
    
    return df

def to_date(df, col):
    '''
    Convert a column of strings to datetime.
    
    Args:
        df (pd.DataFrame): The DataFrame to process.
        col (str): The column to convert.
    '''
    df[col] = pd.to_datetime(df[col])

def shift_year(series, offset_series):
    '''
    Shift the year of a datetime series by an offset series.
    
    Args:
        series (pd.Series): The datetime series to shift.
        offset_series (pd.Series): The offset series.
    
    Returns:
        pd.Series: The shifted datetime series.
    '''
    result = []
    for dt, offset in zip(series, offset_series):
        if pd.notna(dt):
            new_year = dt.year + int(offset)
            try:
                result.append(dt.replace(year=new_year))
            except ValueError:
                # Feb 29 in a non-leap target year → use Feb 28
                result.append(dt.replace(year=new_year, day=28))
        else:
            result.append(pd.NaT)
    return pd.Series(result, index=series.index)

    