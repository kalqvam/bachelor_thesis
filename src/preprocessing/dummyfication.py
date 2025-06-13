import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union

from ..utils import (
    DEFAULT_TICKER_COLUMN, print_subsection_header, 
    format_number, print_processing_stats
)


def add_shock_dummy(df: pd.DataFrame,
                   shock_name: str,
                   year: Optional[Union[int, List[int]]] = None,
                   quarter: Optional[Union[int, List[int]]] = None,
                   date_column: str = 'quarter',
                   verbose: bool = True) -> Tuple[pd.DataFrame, Dict]:
    
    if verbose:
        print_subsection_header(f"Adding Shock Dummy: {shock_name}")
    
    if date_column not in df.columns:
        if verbose:
            print(f"Error: '{date_column}' column not found in the DataFrame")
        return df.copy(), {'status': 'date_column_not_found'}
    
    original_shape = df.shape
    df_result = df.copy()
    
    df_result[shock_name] = 0
    
    if pd.api.types.is_string_dtype(df_result[date_column]):
        if df_result[date_column].iloc[0].find('Q') > 0:
            df_result['temp_year'] = df_result[date_column].str.split('-').str[0].astype(int)
            df_result['temp_quarter'] = df_result[date_column].str.split('-Q').str[1].astype(int)
        else:
            df_result['temp_date'] = pd.to_datetime(df_result[date_column])
            df_result['temp_year'] = df_result['temp_date'].dt.year
            df_result['temp_quarter'] = df_result['temp_date'].dt.quarter
            df_result.drop('temp_date', axis=1, inplace=True)
    elif pd.api.types.is_datetime64_any_dtype(df_result[date_column]):
        df_result['temp_year'] = df_result[date_column].dt.year
        df_result['temp_quarter'] = df_result[date_column].dt.quarter
    else:
        if verbose:
            print(f"Error: Cannot parse date format in column '{date_column}'")
        return df.copy(), {'status': 'date_format_error'}
    
    if year is not None and not isinstance(year, list):
        year = [year]
    if quarter is not None and not isinstance(quarter, list):
        quarter = [quarter]
    
    rows_affected = 0
    
    if year is not None and quarter is not None:
        for y in year:
            for q in quarter:
                mask = (df_result['temp_year'] == y) & (df_result['temp_quarter'] == q)
                df_result.loc[mask, shock_name] = 1
                rows_affected += mask.sum()
        
        if verbose:
            period_desc = f"year(s) {year}, quarter(s) {quarter}"
    
    elif year is not None:
        for y in year:
            mask = df_result['temp_year'] == y
            df_result.loc[mask, shock_name] = 1
            rows_affected += mask.sum()
        
        if verbose:
            period_desc = f"year(s) {year}"
    
    else:
        if verbose:
            print("Error: At least 'year' must be specified")
        return df.copy(), {'status': 'insufficient_parameters'}
    
    df_result.drop(['temp_year', 'temp_quarter'], axis=1, errors='ignore', inplace=True)
    
    shock_coverage = (rows_affected / len(df_result)) * 100 if len(df_result) > 0 else 0
    
    stats = {
        'original_shape': original_shape,
        'final_shape': df_result.shape,
        'shock_name': shock_name,
        'rows_affected': rows_affected,
        'shock_coverage_percent': shock_coverage,
        'date_column_used': date_column,
        'period_description': period_desc if 'period_desc' in locals() else 'unknown'
    }
    
    if verbose:
        print(f"Shock dummy '{shock_name}' results:")
        print(f"  Period: {stats['period_description']}")
        print(f"  Rows affected: {format_number(rows_affected)} ({shock_coverage:.1f}%)")
        print(f"  Column '{shock_name}' added successfully")
    
    return df_result, stats


def add_time_dummy(df: pd.DataFrame,
                  ticker_column: str = DEFAULT_TICKER_COLUMN,
                  date_column: str = 'quarter',
                  dummy_name: str = 'time_dummy',
                  verbose: bool = True) -> Tuple[pd.DataFrame, Dict]:
    
    if verbose:
        print_subsection_header(f"Adding Time Dummy: {dummy_name}")
    
    if ticker_column not in df.columns:
        if verbose:
            print(f"Error: '{ticker_column}' column not found in the DataFrame")
        return df.copy(), {'status': 'ticker_column_not_found'}
    
    if date_column not in df.columns:
        if verbose:
            print(f"Error: '{date_column}' column not found in the DataFrame")
        return df.copy(), {'status': 'date_column_not_found'}
    
    original_shape = df.shape
    df_result = df.copy()
    
    if not pd.api.types.is_datetime64_any_dtype(df_result[date_column]):
        if pd.api.types.is_string_dtype(df_result[date_column]):
            if df_result[date_column].iloc[0].find('Q') > 0:
                df_result['temp_year'] = df_result[date_column].str.split('-').str[0].astype(int)
                df_result['temp_quarter'] = df_result[date_column].str.split('-Q').str[1].astype(int)
                df_result['temp_date'] = df_result['temp_year'] * 10 + df_result['temp_quarter']
                sort_column = 'temp_date'
            else:
                try:
                    df_result['temp_date'] = pd.to_datetime(df_result[date_column])
                    sort_column = 'temp_date'
                except:
                    sort_column = date_column
        else:
            sort_column = date_column
    else:
        sort_column = date_column
    
    df_result = df_result.sort_values([ticker_column, sort_column])
    
    df_result[dummy_name] = 0
    
    company_stats = {}
    
    for company in df_result[ticker_column].unique():
        company_mask = df_result[ticker_column] == company
        company_indices = df_result[company_mask].index
        
        for i, idx in enumerate(company_indices, 1):
            df_result.loc[idx, dummy_name] = i
        
        company_stats[company] = len(company_indices)
    
    if 'temp_date' in df_result.columns:
        df_result.drop('temp_date', axis=1, inplace=True)
    if 'temp_year' in df_result.columns:
        df_result.drop(['temp_year', 'temp_quarter'], axis=1, inplace=True)
    
    stats = {
        'original_shape': original_shape,
        'final_shape': df_result.shape,
        'dummy_name': dummy_name,
        'unique_companies': len(company_stats),
        'min_time_periods': min(company_stats.values()) if company_stats else 0,
        'max_time_periods': max(company_stats.values()) if company_stats else 0,
        'avg_time_periods': sum(company_stats.values()) / len(company_stats) if company_stats else 0,
        'ticker_column_used': ticker_column,
        'date_column_used': date_column
    }
    
    if verbose:
        print(f"Time dummy '{dummy_name}' results:")
        print(f"  Companies processed: {format_number(stats['unique_companies'])}")
        print(f"  Time periods per company: {stats['min_time_periods']}-{stats['max_time_periods']} (avg: {stats['avg_time_periods']:.1f})")
        print(f"  Column '{dummy_name}' added successfully")
    
    return df_result, stats
