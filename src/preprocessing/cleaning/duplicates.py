import pandas as pd
from typing import Dict, List, Optional, Tuple

from ...utils import (
    DEFAULT_TICKER_COLUMN, print_subsection_header, 
    format_number, print_processing_stats
)


def identify_duplicates(df: pd.DataFrame, 
                       ticker_column: str = DEFAULT_TICKER_COLUMN,
                       date_column: str = 'quarter_year') -> pd.DataFrame:
    duplicates_mask = df.duplicated(subset=[ticker_column, date_column], keep=False)
    duplicate_rows = df[duplicates_mask].copy()
    
    return duplicate_rows


def analyze_duplicate_patterns(df: pd.DataFrame,
                             ticker_column: str = DEFAULT_TICKER_COLUMN,
                             date_column: str = 'quarter_year') -> Dict[str, any]:
    duplicate_rows = identify_duplicates(df, ticker_column, date_column)
    
    if duplicate_rows.empty:
        return {
            'has_duplicates': False,
            'total_duplicate_rows': 0,
            'affected_tickers': 0,
            'duplicate_periods': {}
        }
    
    duplicate_groups = duplicate_rows.groupby([ticker_column, date_column]).size()
    
    analysis = {
        'has_duplicates': True,
        'total_duplicate_rows': len(duplicate_rows),
        'affected_tickers': duplicate_rows[ticker_column].nunique(),
        'duplicate_periods': duplicate_groups.to_dict(),
        'max_duplicates_per_period': duplicate_groups.max(),
        'avg_duplicates_per_period': duplicate_groups.mean()
    }
    
    return analysis


def remove_duplicates_keep_latest(df: pd.DataFrame,
                                ticker_column: str = DEFAULT_TICKER_COLUMN,
                                date_column: str = 'quarter_year',
                                sort_column: str = 'date',
                                verbose: bool = True) -> Tuple[pd.DataFrame, Dict]:
    
    if verbose:
        print_subsection_header("Handling Duplicate Records")
    
    original_shape = df.shape
    
    duplicate_analysis = analyze_duplicate_patterns(df, ticker_column, date_column)
    
    if not duplicate_analysis['has_duplicates']:
        if verbose:
            print("No duplicate ticker-period combinations found.")
        
        stats = {
            'original_shape': original_shape,
            'final_shape': df.shape,
            'duplicates_found': 0,
            'duplicates_removed': 0,
            'affected_tickers': 0
        }
        return df.copy(), stats
    
    if verbose:
        print(f"Found {duplicate_analysis['affected_tickers']} tickers with duplicate {date_column} entries")
        print(f"Total duplicate rows: {duplicate_analysis['total_duplicate_rows']}")
    
    if sort_column in df.columns:
        df_sorted = df.sort_values([ticker_column, date_column, sort_column])
        df_cleaned = df_sorted.drop_duplicates(subset=[ticker_column, date_column], keep='last')
        if verbose:
            print(f"Kept the latest observation for each duplicate period based on '{sort_column}' column")
    else:
        df_cleaned = df.drop_duplicates(subset=[ticker_column, date_column], keep='last')
        if verbose:
            print(f"Kept the last occurrence for each duplicate period (no {sort_column} column found)")
    
    removed_count = original_shape[0] - df_cleaned.shape[0]
    
    if verbose:
        print(f"Removed {format_number(removed_count)} duplicate rows")
        print(f"Dataset shape: {original_shape} → {df_cleaned.shape}")
    
    stats = {
        'original_shape': original_shape,
        'final_shape': df_cleaned.shape,
        'duplicates_found': duplicate_analysis['total_duplicate_rows'],
        'duplicates_removed': removed_count,
        'affected_tickers': duplicate_analysis['affected_tickers'],
        'duplicate_analysis': duplicate_analysis
    }
    
    return df_cleaned, stats


def remove_duplicates_custom_logic(df: pd.DataFrame,
                                  ticker_column: str = DEFAULT_TICKER_COLUMN,
                                  date_column: str = 'quarter_year',
                                  priority_columns: Optional[List[str]] = None,
                                  aggregation_rules: Optional[Dict[str, str]] = None,
                                  verbose: bool = True) -> Tuple[pd.DataFrame, Dict]:
    
    if verbose:
        print_subsection_header("Handling Duplicates with Custom Logic")
    
    original_shape = df.shape
    duplicate_analysis = analyze_duplicate_patterns(df, ticker_column, date_column)
    
    if not duplicate_analysis['has_duplicates']:
        if verbose:
            print("No duplicates found.")
        return df.copy(), {'original_shape': original_shape, 'final_shape': df.shape}
    
    if priority_columns:
        sort_cols = [ticker_column, date_column] + priority_columns
        df_sorted = df.sort_values(sort_cols, ascending=[True, True] + [False] * len(priority_columns))
        df_cleaned = df_sorted.drop_duplicates(subset=[ticker_column, date_column], keep='first')
        
        if verbose:
            print(f"Resolved duplicates using priority columns: {priority_columns}")
    
    elif aggregation_rules:
        df_cleaned = df.groupby([ticker_column, date_column]).agg(aggregation_rules).reset_index()
        
        if verbose:
            print(f"Resolved duplicates using aggregation rules: {aggregation_rules}")
    
    else:
        df_cleaned = df.drop_duplicates(subset=[ticker_column, date_column], keep='last')
        
        if verbose:
            print("Resolved duplicates by keeping last occurrence")
    
    removed_count = original_shape[0] - df_cleaned.shape[0]
    
    if verbose:
        print(f"Removed {format_number(removed_count)} duplicate rows")
        print(f"Dataset shape: {original_shape} → {df_cleaned.shape}")
    
    stats = {
        'original_shape': original_shape,
        'final_shape': df_cleaned.shape,
        'duplicates_removed': removed_count,
        'method_used': 'priority_columns' if priority_columns else 'aggregation' if aggregation_rules else 'keep_last'
    }
    
    return df_cleaned, stats


def validate_no_duplicates(df: pd.DataFrame,
                          ticker_column: str = DEFAULT_TICKER_COLUMN,
                          date_column: str = 'quarter_year') -> bool:
    duplicate_analysis = analyze_duplicate_patterns(df, ticker_column, date_column)
    return not duplicate_analysis['has_duplicates']


def print_duplicate_summary(stats: Dict, title: str = "Duplicate Handling Summary"):
    print_processing_stats(stats, title)
    
    if 'duplicate_analysis' in stats and stats['duplicate_analysis']['has_duplicates']:
        duplicate_info = stats['duplicate_analysis']
        print(f"Duplicate pattern details:")
        print(f"  Max duplicates per period: {duplicate_info['max_duplicates_per_period']}")
        print(f"  Avg duplicates per period: {duplicate_info['avg_duplicates_per_period']:.1f}")
        
        if len(duplicate_info['duplicate_periods']) <= 10:
            print(f"  Affected periods: {list(duplicate_info['duplicate_periods'].keys())}")
        else:
            sample_periods = list(duplicate_info['duplicate_periods'].keys())[:5]
            print(f"  Sample affected periods: {sample_periods}...")


def get_duplicate_examples(df: pd.DataFrame,
                          ticker_column: str = DEFAULT_TICKER_COLUMN,
                          date_column: str = 'quarter_year',
                          max_examples: int = 5) -> pd.DataFrame:
    duplicates = identify_duplicates(df, ticker_column, date_column)
    
    if duplicates.empty:
        return pd.DataFrame()
    
    unique_periods = duplicates[[ticker_column, date_column]].drop_duplicates().head(max_examples)
    
    examples = []
    for _, period in unique_periods.iterrows():
        ticker = period[ticker_column]
        date = period[date_column]
        
        period_duplicates = df[
            (df[ticker_column] == ticker) & 
            (df[date_column] == date)
        ].copy()
        
        examples.append(period_duplicates)
    
    if examples:
        return pd.concat(examples, ignore_index=True)
    
    return pd.DataFrame()
