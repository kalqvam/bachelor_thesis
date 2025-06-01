import pandas as pd
import numpy as np
from typing import Set, List, Tuple, Dict, Any

from ...utils import (
    DEFAULT_TICKER_COLUMN, DEFAULT_OUTLIER_THRESHOLD, EXCLUDED_PERIODS,
    print_section_header, format_number
)
from .utils import validate_required_columns, remove_companies_and_report


def process_outliers_in_dataset(df: pd.DataFrame,
                               threshold: float = DEFAULT_OUTLIER_THRESHOLD,
                               ticker_column: str = DEFAULT_TICKER_COLUMN,
                               date_column: str = 'quarter',
                               verbose: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    
    if verbose:
        print_section_header("Statistical Outlier Detection and Correction")
    
    original_shape = df.shape
    
    required_columns = ['ebitda', 'revenue', 'cashAndCashEquivalents', 'totalDebt', 'totalAssets']
    columns_to_analyze = validate_required_columns(df, required_columns, "outlier detection")
    
    if not columns_to_analyze:
        return df.copy(), {'status': 'no_required_columns'}
    
    df_result = df.copy()
    df_result[date_column] = pd.to_datetime(df_result[date_column])
    df_result['year'] = df_result[date_column].dt.year
    df_result['quarter_num'] = df_result[date_column].dt.quarter
    
    if verbose:
        print(f"Original dataset shape: {original_shape}")
        print(f"Number of unique tickers: {format_number(df_result[ticker_column].nunique())}")
        print(f"Analyzing columns: {columns_to_analyze}")
    
    excluded_mask = df_result.apply(_is_excluded_period, axis=1)
    excluded_count = excluded_mask.sum()
    
    if verbose:
        print(f"Number of rows in excluded periods: {format_number(excluded_count)}")
    
    tickers_to_remove = set()
    modification_stats = {col: {'total_outliers': 0, 'single_spikes': 0, 'fixed_outliers': 0} 
                         for col in columns_to_analyze}
    
    for ticker in df_result[ticker_column].unique():
        ticker_data = df_result[df_result[ticker_column] == ticker].copy()
        
        non_excluded_data = ticker_data[~ticker_data.apply(_is_excluded_period, axis=1)]
        if len(non_excluded_data) < 2:
            continue
        
        has_non_single_spike_outlier = False
        
        for column in columns_to_analyze:
            column_mean = non_excluded_data[column].mean()
            column_std = non_excluded_data[column].std()
            
            if column_std == 0 or np.isnan(column_std):
                continue
            
            outlier_mask = abs(ticker_data[column] - column_mean) > threshold * column_std
            outlier_indices = ticker_data[outlier_mask].index.tolist()
            
            modification_stats[column]['total_outliers'] += len(outlier_indices)
            
            for idx in outlier_indices:
                if _is_excluded_period(df_result.loc[idx]):
                    continue
                
                sorted_ticker_data = ticker_data.sort_values(date_column)
                sorted_indices = sorted_ticker_data.index.tolist()
                current_pos = sorted_indices.index(idx)
                
                has_enough_neighbors = (current_pos >= 2) and (current_pos < len(sorted_indices) - 2)
                
                if not has_enough_neighbors:
                    is_single = False
                else:
                    neighbor_indices = [
                        sorted_indices[current_pos - 2],
                        sorted_indices[current_pos - 1],
                        sorted_indices[current_pos + 1],
                        sorted_indices[current_pos + 2]
                    ]
                    
                    is_single = True
                    for n_idx in neighbor_indices:
                        if abs(df_result.loc[n_idx, column] - column_mean) > 1 * column_std:
                            is_single = False
                            break
                
                if is_single:
                    modification_stats[column]['single_spikes'] += 1
                    
                    prev_idx = sorted_indices[current_pos - 1]
                    next_idx = sorted_indices[current_pos + 1]
                    
                    prev_val = df_result.loc[prev_idx, column]
                    next_val = df_result.loc[next_idx, column]
                    
                    df_result.loc[idx, column] = (prev_val + next_val) / 2
                    modification_stats[column]['fixed_outliers'] += 1
                else:
                    has_non_single_spike_outlier = True
        
        if has_non_single_spike_outlier:
            tickers_to_remove.add(ticker)
    
    df_filtered, removal_stats = remove_companies_and_report(
        df_result, tickers_to_remove, "Persistent Outlier Removal", ticker_column, verbose=False
    )
    
    df_filtered = df_filtered.drop(['year', 'quarter_num'], axis=1, errors='ignore')
    
    stats = {
        'original_shape': original_shape,
        'final_shape': df_filtered.shape,
        'excluded_periods_count': excluded_count,
        'threshold_used': threshold,
        'modification_stats': modification_stats,
        'removal_stats': removal_stats
    }
    
    if verbose:
        _print_outlier_summary(stats, tickers_to_remove)
    
    return df_filtered, stats


def _is_excluded_period(row: pd.Series) -> bool:
    return (row['year'], row['quarter_num']) in EXCLUDED_PERIODS


def _print_outlier_summary(stats: Dict[str, Any], tickers_removed: Set[str]) -> None:
    print("\nOutlier Processing Summary:")
    print(f"Total tickers processed: {stats['original_shape'][0] // 4 if stats['original_shape'][0] > 0 else 0}")  # Rough estimate
    print(f"Tickers removed: {len(tickers_removed)} ({len(tickers_removed) / (stats['original_shape'][0] // 4) * 100:.2f}%)" if stats['original_shape'][0] > 0 else "Tickers removed: 0")
    
    if tickers_removed:
        tickers_list = sorted(list(tickers_removed))
        if len(tickers_list) <= 10:
            print(f"Tickers removed: {', '.join(tickers_list)}")
        else:
            print(f"Sample tickers removed: {', '.join(tickers_list[:10])}...")
    
    for column, col_stats in stats['modification_stats'].items():
        if col_stats['total_outliers'] > 0:
            print(f"\n{column.upper()} Statistics:")
            print(f"  Total outliers detected: {format_number(col_stats['total_outliers'])}")
            print(f"  Single spike outliers: {format_number(col_stats['single_spikes'])}")
            print(f"  Fixed by neighbor averaging: {format_number(col_stats['fixed_outliers'])}")
    
    print(f"\nFinal dataset shape: {stats['final_shape']}")
    if stats['final_shape'][0] > 0:
        final_tickers = stats['final_shape'][0] // 4
        print(f"Final number of unique tickers: ~{format_number(final_tickers)}")
