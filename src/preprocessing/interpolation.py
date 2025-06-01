import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional

from ..utils import (
    DEFAULT_TICKER_COLUMN, print_subsection_header, 
    format_number
)


def find_consecutive_zeros(series: pd.Series) -> List[int]:
    is_zero = (series == 0)

    if not any(is_zero):
        return []

    runs = []
    run_length = 0

    for i, zero in enumerate(is_zero):
        if zero:
            run_length += 1
        elif run_length > 0:
            runs.append(run_length)
            run_length = 0

    if run_length > 0:
        runs.append(run_length)

    return runs


def process_financial_data_interpolation(df: pd.DataFrame,
                                       consecutive_threshold: int = 3,
                                       columns_to_analyze: Optional[List[str]] = None,
                                       ticker_column: str = DEFAULT_TICKER_COLUMN,
                                       date_column: str = 'quarter',
                                       verbose: bool = True) -> Tuple[pd.DataFrame, Dict]:
    
    if verbose:
        print_subsection_header("Processing Financial Data with Zero Interpolation")
    
    if columns_to_analyze is None:
        columns_to_analyze = ['ebitda', 'revenue', 'cashAndCashEquivalents', 'totalDebt', 'totalAssets']
    
    df_copy = df.copy()
    df_copy[date_column] = pd.to_datetime(df_copy[date_column])
    df_copy['year'] = df_copy[date_column].dt.year
    df_copy['quarter_num'] = df_copy[date_column].dt.quarter

    companies_to_remove = set()

    for ticker in df_copy[ticker_column].unique():
        ticker_data = df_copy[df_copy[ticker_column] == ticker].sort_values(date_column)

        if len(ticker_data) < 2:
            continue

        for col in columns_to_analyze:
            if col not in ticker_data.columns:
                continue

            consecutive_runs = find_consecutive_zeros(ticker_data[col])

            if any(run_length > consecutive_threshold for run_length in consecutive_runs):
                companies_to_remove.add(ticker)
                break

    original_company_count = df_copy[ticker_column].nunique()
    
    if companies_to_remove:
        df_filtered = df_copy[~df_copy[ticker_column].isin(companies_to_remove)].copy()
    else:
        df_filtered = df_copy.copy()

    for ticker in df_filtered[ticker_column].unique():
        ticker_mask = df_filtered[ticker_column] == ticker
        ticker_indices = df_filtered[ticker_mask].index

        for col in columns_to_analyze:
            if col in df_filtered.columns:
                if (df_filtered.loc[ticker_indices, col] == 0).any():
                    series = df_filtered.loc[ticker_indices, col].replace(0, np.nan)
                    interpolated = series.interpolate(method='linear')
                    df_filtered.loc[ticker_indices, col] = interpolated

    df_filtered = df_filtered.drop(['year', 'quarter_num'], axis=1, errors='ignore')

    stats = {
        'original_company_count': original_company_count,
        'removed_company_count': len(companies_to_remove),
        'remaining_company_count': df_filtered[ticker_column].nunique(),
        'companies_removed': list(companies_to_remove),
        'consecutive_threshold': consecutive_threshold
    }

    if verbose:
        print(f"Original number of companies: {stats['original_company_count']}")
        print(f"Companies removed due to exceeding {consecutive_threshold} consecutive zeros: {stats['removed_company_count']}")
        print(f"Remaining companies after filtering: {stats['remaining_company_count']}")

    return df_filtered, stats
