import pandas as pd
from typing import Dict, Set, Tuple, Any

from ...utils import (
    DEFAULT_TICKER_COLUMN, DEFAULT_SAMPLE_DISPLAY_SIZE,
    print_section_header, format_number
)
from .utils import validate_required_columns, remove_companies_and_report


def filter_companies_by_multiple_columns(df: pd.DataFrame,
                                       column_filters: Dict[str, Dict[str, float]],
                                       ticker_column: str = DEFAULT_TICKER_COLUMN,
                                       verbose: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    
    if verbose:
        print_section_header("Multi-Column Value Range Filtering")
    
    original_shape = df.shape
    original_companies = df[ticker_column].nunique()
    
    if verbose:
        print(f"Original dataset shape: {original_shape}")
        print(f"Number of unique companies: {format_number(original_companies)}")
    
    for column_name in column_filters.keys():
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found in the dataset.")
    
    all_companies_to_remove = set()
    column_removal_stats = {}
    
    for column_name, filter_values in column_filters.items():
        min_value = filter_values.get('min_value')
        max_value = filter_values.get('max_value')
        
        companies_to_remove_for_column = _identify_companies_outside_range(
            df, column_name, min_value, max_value, ticker_column
        )
        
        all_companies_to_remove.update(companies_to_remove_for_column)
        
        column_removal_stats[column_name] = {
            'min_value': min_value,
            'max_value': max_value,
            'companies_removed': len(companies_to_remove_for_column),
            'companies_removed_list': list(companies_to_remove_for_column)
        }
        
        if verbose:
            _print_column_filter_results(column_name, min_value, max_value, 
                                       companies_to_remove_for_column)
    
    df_filtered, removal_stats = remove_companies_and_report(
        df, all_companies_to_remove, "Multi-Column Value Range Filtering", 
        ticker_column, verbose=False
    )
    
    stats = {
        'original_shape': original_shape,
        'final_shape': df_filtered.shape,
        'original_companies': original_companies,
        'final_companies': df_filtered[ticker_column].nunique(),
        'total_removed_companies': len(all_companies_to_remove),
        'column_stats': column_removal_stats,
        'removal_stats': removal_stats,
        'retention_rate_companies': (df_filtered[ticker_column].nunique() / original_companies) * 100 if original_companies > 0 else 0,
        'retention_rate_rows': (df_filtered.shape[0] / original_shape[0]) * 100 if original_shape[0] > 0 else 0
    }
    
    if verbose:
        _print_overall_filter_summary(stats)
    
    return df_filtered, stats


def _identify_companies_outside_range(df: pd.DataFrame,
                                    column_name: str,
                                    min_value: float,
                                    max_value: float,
                                    ticker_column: str) -> Set[str]:
    
    companies_to_remove = set()
    
    for ticker in df[ticker_column].unique():
        ticker_data = df[df[ticker_column] == ticker]
        
        should_remove = False
        if min_value is not None and (ticker_data[column_name] < min_value).any():
            should_remove = True
        
        if not should_remove and max_value is not None and (ticker_data[column_name] > max_value).any():
            should_remove = True
        
        if should_remove:
            companies_to_remove.add(ticker)
    
    return companies_to_remove


def _print_column_filter_results(column_name: str,
                                min_value: float,
                                max_value: float,
                                companies_removed: Set[str]) -> None:
    
    print(f"\nFiltering companies based on {column_name} values...")
    if min_value is not None:
        print(f"Minimum acceptable value: {min_value}")
    if max_value is not None:
        print(f"Maximum acceptable value: {max_value}")
    
    print(f"Found {format_number(len(companies_removed))} companies with {column_name} values outside the specified range.")
    
    if companies_removed:
        companies_list = list(companies_removed)
        if len(companies_list) <= DEFAULT_SAMPLE_DISPLAY_SIZE:
            print(f"Removed companies: {', '.join(companies_list)}")
        else:
            sample = companies_list[:DEFAULT_SAMPLE_DISPLAY_SIZE]
            print(f"Sample of removed companies: {', '.join(sample)}...")


def _print_overall_filter_summary(stats: Dict[str, Any]) -> None:
    print(f"\nTotal companies removed across all filters: {format_number(stats['total_removed_companies'])}")
    
    print(f"\nFiltering results:")
    print(f"Rows removed: {format_number(stats['removal_stats']['rows_removed'])}")
    print(f"Final dataset shape: {stats['final_shape'][0]} rows, {stats['final_shape'][1]} columns")
    print(f"Final number of companies: {format_number(stats['final_companies'])}")
    
    print(f"Percentage of rows retained: {stats['retention_rate_rows']:.2f}%")
    print(f"Percentage of companies retained: {stats['retention_rate_companies']:.2f}%")
    
    print("\nColumn-specific filtering results:")
    for col, col_stats in stats['column_stats'].items():
        print(f"  {col}:")
        if col_stats['min_value'] is not None:
            print(f"    Min value: {col_stats['min_value']}")
        if col_stats['max_value'] is not None:
            print(f"    Max value: {col_stats['max_value']}")
        print(f"    Companies removed: {format_number(col_stats['companies_removed'])}")
