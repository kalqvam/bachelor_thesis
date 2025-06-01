import pandas as pd
import numpy as np
from typing import List, Set, Tuple, Dict, Any

from ...utils import (
    DEFAULT_TICKER_COLUMN, print_subsection_header, 
    format_number, print_processing_stats
)


def validate_required_columns(df: pd.DataFrame, 
                            required_columns: List[str], 
                            operation_name: str) -> List[str]:
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Warning: Missing columns for {operation_name}: {missing_columns}")
    
    return [col for col in required_columns if col in df.columns]


def remove_companies_and_report(df: pd.DataFrame,
                              companies_to_remove: Set[str],
                              operation_name: str,
                              ticker_column: str = DEFAULT_TICKER_COLUMN,
                              verbose: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    
    original_shape = df.shape
    original_companies = df[ticker_column].nunique()
    
    if not companies_to_remove:
        stats = {
            'original_shape': original_shape,
            'final_shape': df.shape,
            'companies_removed': 0,
            'rows_removed': 0,
            'retention_rate': 100.0,
            'operation': operation_name
        }
        return df.copy(), stats
    
    df_filtered = df[~df[ticker_column].isin(companies_to_remove)].copy()
    
    rows_removed = original_shape[0] - df_filtered.shape[0]
    companies_removed = len(companies_to_remove)
    final_companies = df_filtered[ticker_column].nunique()
    retention_rate = (final_companies / original_companies) * 100 if original_companies > 0 else 0
    
    stats = {
        'original_shape': original_shape,
        'final_shape': df_filtered.shape,
        'companies_removed': companies_removed,
        'companies_removed_list': list(companies_to_remove),
        'rows_removed': rows_removed,
        'retention_rate': retention_rate,
        'operation': operation_name
    }
    
    if verbose:
        print(f"\n{operation_name} Results:")
        print(f"Companies removed: {format_number(companies_removed)}")
        print(f"Rows removed: {format_number(rows_removed)}")
        print(f"Retention rate: {retention_rate:.1f}%")
        print(f"Final shape: {df_filtered.shape}")
    
    return df_filtered, stats


def create_declowning_flags_column(df: pd.DataFrame) -> pd.DataFrame:
    df_result = df.copy()
    if 'clown_flags' not in df_result.columns:
        df_result['clown_flags'] = ""
    return df_result


def finalize_clown_flags(df: pd.DataFrame) -> pd.DataFrame:
    df_result = df.copy()
    df_result['clown_flags'] = df_result['clown_flags'].replace("", np.nan)
    return df_result


def print_declowning_summary(modifications: Dict[str, int], 
                           companies_removed: int,
                           operation_name: str) -> None:
    print(f"\n===== {operation_name} Results =====")
    
    for modification_type, count in modifications.items():
        print(f"{modification_type}: {format_number(count)}")
    
    if companies_removed > 0:
        print(f"Companies removed: {format_number(companies_removed)}")
    
    total_modifications = sum(modifications.values())
    print(f"Total modifications: {format_number(total_modifications)}")
