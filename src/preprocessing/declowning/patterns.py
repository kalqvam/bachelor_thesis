import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Dict, Any

from ...utils import (
    DEFAULT_TICKER_COLUMN, print_section_header, format_number
)
from .utils import (
    validate_required_columns, remove_companies_and_report,
    create_declowning_flags_column, finalize_clown_flags
)


def clean_financial_data(df: pd.DataFrame,
                        ticker_column: str = DEFAULT_TICKER_COLUMN,
                        date_column: str = 'quarter',
                        verbose: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    
    if verbose:
        print_section_header("🤡 CLOWN ELIMINATION - Pattern Detection")
    
    df_original = df.copy()
    df_result = create_declowning_flags_column(df)
    
    required_columns = ['ebitda', 'revenue']
    available_columns = validate_required_columns(df_result, required_columns, "pattern detection")
    
    if not available_columns:
        return df_result, {'status': 'no_required_columns'}
    
    df_result[date_column] = pd.to_datetime(df_result[date_column])
    
    companies_to_remove = set()
    modification_counts = {
        'Reversed EBITDA signs fixed': 0,
        '10x too large EBITDA values fixed': 0,
        '10x too small EBITDA values fixed': 0
    }
    
    if verbose:
        print(f"Processing {format_number(df_result[ticker_column].nunique())} unique tickers...")
    
    for ticker in df_result[ticker_column].unique():
        company_data = df_result[df_result[ticker_column] == ticker].copy()
        company_data = company_data.sort_values(date_column)
        
        if len(company_data) <= 2:
            continue
        
        if 'ebitda' in company_data.columns and len(company_data) >= 3:
            _process_ebitda_patterns(df_result, company_data, ticker, modification_counts, verbose)
        
        if len(company_data) >= 4 and 'revenue' in company_data.columns and 'ebitda' in company_data.columns:
            if _detect_revenue_gross_profit_confusion(company_data, ticker, verbose):
                companies_to_remove.add(ticker)
    
    df_result = finalize_clown_flags(df_result)
    
    df_filtered, removal_stats = remove_companies_and_report(
        df_result, companies_to_remove, "Revenue/Gross Profit Confusion Detection", 
        ticker_column, verbose=False
    )
    
    modification_counts['Companies removed due to revenue/gross profit confusion'] = len(companies_to_remove)
    
    if verbose:
        _print_clown_results(modification_counts, df_filtered)
    
    stats = {
        'original_shape': df_original.shape,
        'final_shape': df_filtered.shape,
        'modifications': modification_counts,
        'removal_stats': removal_stats,
        'clown_flags_added': df_filtered['clown_flags'].notna().sum(),
        'percentage_modified': (df_filtered['clown_flags'].notna().sum() / len(df_filtered) * 100) if len(df_filtered) > 0 else 0
    }
    
    return df_filtered, stats


def _process_ebitda_patterns(df_result: pd.DataFrame, 
                           company_data: pd.DataFrame, 
                           ticker: str, 
                           modification_counts: Dict[str, int],
                           verbose: bool) -> None:
    
    ebitda_series = company_data['ebitda'].values
    date_series = company_data['quarter'].values
    original_indices = company_data.index.values
    
    if len(ebitda_series) >= 5:
        for i in range(2, len(ebitda_series) - 2):
            if any(pd.isna(val) for val in ebitda_series[i-2:i+3]):
                continue
            
            prev_prev, prev, current, next_val, next_next = ebitda_series[i-2:i+3]
            
            current_sign_positive = current > 0
            neighbors_sign_positive = [val > 0 for val in [prev_prev, prev, next_val, next_next]]
            
            consistent_opposite_signs = all(sign != current_sign_positive for sign in neighbors_sign_positive)
            
            if consistent_opposite_signs:
                abs_current = abs(current)
                
                magnitudes_similar = (
                    abs_current <= 1.75 * abs(prev) and abs_current >= abs(prev) / 1.75 and
                    abs_current <= 1.75 * abs(next_val) and abs_current >= abs(next_val) / 1.75
                )
                
                if magnitudes_similar:
                    logical_progression = False
                    
                    if current_sign_positive:
                        logical_progression = abs(prev) >= abs(next_val)
                    else:
                        logical_progression = abs(prev) <= abs(next_val)
                    
                    if logical_progression:
                        idx = original_indices[i]
                        df_result.at[idx, 'ebitda'] = -current
                        df_result.at[idx, 'clown_flags'] += "reversed_ebitda;"
                        modification_counts['Reversed EBITDA signs fixed'] += 1
                        if verbose:
                            print(f"🤡 Fixed reversed EBITDA sign for {ticker} at {date_series[i]}: {current} -> {-current} (5-quarter check)")
    
    for i in range(1, len(ebitda_series) - 1):
        if any(pd.isna(val) for val in [ebitda_series[i-1], ebitda_series[i], ebitda_series[i+1]]):
            continue
        
        current = df_result.loc[original_indices[i], 'ebitda']
        prev = df_result.loc[original_indices[i-1], 'ebitda']
        next_val = df_result.loc[original_indices[i+1], 'ebitda']
        
        if current == 0 or prev == 0 or next_val == 0:
            continue
        
        if (current > 0 and prev > 0 and next_val > 0) or (current < 0 and prev < 0 and next_val < 0):
            ratio_prev = abs(current / prev)
            ratio_next = abs(current / next_val)
            
            if (9 <= ratio_prev <= 11) and (9 <= ratio_next <= 11):
                idx = original_indices[i]
                corrected_value = current / 10
                df_result.at[idx, 'ebitda'] = corrected_value
                df_result.at[idx, 'clown_flags'] += "10x_ebitda_large;"
                modification_counts['10x too large EBITDA values fixed'] += 1
                if verbose:
                    print(f"🤡 Fixed 10x too large EBITDA for {ticker} at {date_series[i]}: {current} -> {corrected_value}")
            
            if (0.09 <= ratio_prev <= 0.11) and (0.09 <= ratio_next <= 0.11):
                idx = original_indices[i]
                corrected_value = current * 10
                df_result.at[idx, 'ebitda'] = corrected_value
                df_result.at[idx, 'clown_flags'] += "10x_ebitda_small;"
                modification_counts['10x too small EBITDA values fixed'] += 1
                if verbose:
                    print(f"🤡 Fixed 10x too small EBITDA for {ticker} at {date_series[i]}: {current} -> {corrected_value}")


def _detect_revenue_gross_profit_confusion(company_data: pd.DataFrame, 
                                         ticker: str, 
                                         verbose: bool) -> bool:
    
    company_data_copy = company_data.copy()
    company_data_copy['revenue_pct_change'] = company_data_copy['revenue'].pct_change().abs()
    high_volatility_periods = company_data_copy['revenue_pct_change'] > 1.0
    
    if high_volatility_periods.sum() >= 2:
        high_vol_idxs = company_data_copy[high_volatility_periods].index
        
        ebitda_stable_during_rev_jumps = True
        for idx in high_vol_idxs:
            pos = company_data_copy.index.get_loc(idx)
            if pos > 0:
                rev_change = company_data_copy.loc[idx, 'revenue_pct_change']
                ebitda_pct_change = abs((company_data_copy['ebitda'].iloc[pos] - company_data_copy['ebitda'].iloc[pos-1]) /
                                      company_data_copy['ebitda'].iloc[pos-1] if company_data_copy['ebitda'].iloc[pos-1] != 0 else 0)
                
                if pd.notna(ebitda_pct_change) and ebitda_pct_change >= 0.5 * rev_change:
                    ebitda_stable_during_rev_jumps = False
                    break
        
        if ebitda_stable_during_rev_jumps and high_volatility_periods.sum() >= 2:
            if verbose:
                print(f"🤡 Detected gross profit instead of revenue pattern for {ticker} - marked for removal")
            return True
    
    return False


def _print_clown_results(modification_counts: Dict[str, int], df_result: pd.DataFrame) -> None:
    print("\n===== 🤡 CLOWN ELIMINATION RESULTS 🤡 =====")
    
    for modification_type, count in modification_counts.items():
        print(f"{modification_type}: {format_number(count)}")
    
    total_clown_flags = df_result['clown_flags'].notna().sum()
    percentage_modified = (total_clown_flags / len(df_result) * 100) if len(df_result) > 0 else 0
    
    print(f"Total clown flags: {format_number(total_clown_flags)}")
    print(f"Percentage of rows modified: {percentage_modified:.2f}%")
    print("🤡" * 20)
