import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional

from ...utils import (
    DEFAULT_TICKER_COLUMN, print_subsection_header, 
    format_number, print_processing_stats
)


def calculate_expected_observations(lower_year: int, upper_year: int, 
                                  frequency: str = 'quarterly') -> int:
    years_span = upper_year - lower_year + 1
    
    if frequency == 'quarterly':
        return years_span * 4
    elif frequency == 'annual':
        return years_span
    elif frequency == 'monthly':
        return years_span * 12
    else:
        raise ValueError(f"Unsupported frequency: {frequency}")


def get_ticker_observation_counts(df: pd.DataFrame,
                                ticker_column: str = DEFAULT_TICKER_COLUMN) -> pd.Series:
    return df[ticker_column].value_counts()


def identify_incomplete_tickers(df: pd.DataFrame,
                              expected_observations: int,
                              ticker_column: str = DEFAULT_TICKER_COLUMN) -> Tuple[List[str], List[str]]:
    
    ticker_counts = get_ticker_observation_counts(df, ticker_column)
    
    complete_tickers = ticker_counts[ticker_counts == expected_observations].index.tolist()
    incomplete_tickers = ticker_counts[ticker_counts != expected_observations].index.tolist()
    
    return complete_tickers, incomplete_tickers


def validate_complete_observations(df: pd.DataFrame,
                                 lower_year: int,
                                 upper_year: int,
                                 ticker_column: str = DEFAULT_TICKER_COLUMN,
                                 frequency: str = 'quarterly',
                                 verbose: bool = True) -> Dict:
    
    if verbose:
        print_subsection_header(f"Validating Complete Time Series ({lower_year}-{upper_year})")
    
    expected_observations = calculate_expected_observations(lower_year, upper_year, frequency)
    ticker_counts = get_ticker_observation_counts(df, ticker_column)
    
    complete_tickers, incomplete_tickers = identify_incomplete_tickers(
        df, expected_observations, ticker_column
    )
    
    validation_results = {
        'expected_observations': expected_observations,
        'total_tickers': len(ticker_counts),
        'complete_tickers': len(complete_tickers),
        'incomplete_tickers': len(incomplete_tickers),
        'complete_ticker_list': complete_tickers,
        'incomplete_ticker_list': incomplete_tickers,
        'completion_rate': (len(complete_tickers) / len(ticker_counts)) * 100 if len(ticker_counts) > 0 else 0,
        'frequency': frequency
    }
    
    if incomplete_tickers:
        incomplete_stats = ticker_counts[ticker_counts != expected_observations]
        validation_results.update({
            'min_observations': incomplete_stats.min(),
            'max_observations': incomplete_stats.max(),
            'avg_observations': incomplete_stats.mean()
        })
    
    if verbose:
        print(f"Expected observations per ticker: {expected_observations}")
        print(f"Total tickers: {format_number(validation_results['total_tickers'])}")
        print(f"Complete tickers: {format_number(validation_results['complete_tickers'])} ({validation_results['completion_rate']:.1f}%)")
        print(f"Incomplete tickers: {format_number(validation_results['incomplete_tickers'])}")
        
        if incomplete_tickers:
            print(f"Incomplete ticker observation range: {validation_results['min_observations']}-{validation_results['max_observations']}")
    
    return validation_results


def filter_complete_tickers(df: pd.DataFrame,
                          lower_year: int,
                          upper_year: int,
                          ticker_column: str = DEFAULT_TICKER_COLUMN,
                          frequency: str = 'quarterly',
                          verbose: bool = True) -> Tuple[pd.DataFrame, Dict]:
    
    if verbose:
        print_subsection_header("Filtering to Complete Time Series Only")
    
    original_shape = df.shape
    
    validation_results = validate_complete_observations(
        df, lower_year, upper_year, ticker_column, frequency, verbose=False
    )
    
    complete_tickers = validation_results['complete_ticker_list']
    
    if not complete_tickers:
        if verbose:
            print("⚠️  No tickers with complete time series found!")
        return pd.DataFrame(), validation_results
    
    df_filtered = df[df[ticker_column].isin(complete_tickers)].copy()
    
    rows_removed = original_shape[0] - df_filtered.shape[0]
    tickers_removed = validation_results['incomplete_tickers']
    
    stats = {
        'original_shape': original_shape,
        'final_shape': df_filtered.shape,
        'rows_removed': rows_removed,
        'tickers_removed': tickers_removed,
        'tickers_kept': len(complete_tickers),
        'retention_rate': (df_filtered.shape[0] / original_shape[0]) * 100 if original_shape[0] > 0 else 0,
        'validation_details': validation_results
    }
    
    if verbose:
        print(f"Filtering results:")
        print(f"  Original: {format_number(original_shape[0])} rows, {validation_results['total_tickers']} tickers")
        print(f"  Filtered: {format_number(df_filtered.shape[0])} rows, {len(complete_tickers)} tickers")
        print(f"  Removed: {format_number(rows_removed)} rows, {tickers_removed} tickers")
        print(f"  Retention: {stats['retention_rate']:.1f}%")
    
    return df_filtered, stats


def validate_panel_balance(df: pd.DataFrame,
                         ticker_column: str = DEFAULT_TICKER_COLUMN,
                         date_column: str = 'quarter',
                         verbose: bool = True) -> Dict:
    
    if verbose:
        print_subsection_header("Validating Panel Balance")
    
    if date_column not in df.columns:
        return {'status': 'date_column_not_found', 'balanced': False}
    
    ticker_counts = get_ticker_observation_counts(df, ticker_column)
    unique_counts = ticker_counts.unique()
    
    is_balanced = len(unique_counts) == 1
    
    balance_results = {
        'balanced': is_balanced,
        'total_tickers': len(ticker_counts),
        'unique_observation_counts': unique_counts.tolist(),
        'min_observations': ticker_counts.min(),
        'max_observations': ticker_counts.max(),
        'most_common_count': ticker_counts.mode().iloc[0] if len(ticker_counts) > 0 else 0
    }
    
    if not is_balanced:
        count_distribution = ticker_counts.value_counts().sort_index()
        balance_results['count_distribution'] = count_distribution.to_dict()
        
        underrepresented = ticker_counts[ticker_counts < balance_results['most_common_count']].index.tolist()
        overrepresented = ticker_counts[ticker_counts > balance_results['most_common_count']].index.tolist()
        
        balance_results['underrepresented_tickers'] = underrepresented
        balance_results['overrepresented_tickers'] = overrepresented
    
    if verbose:
        if is_balanced:
            print(f"✓ Panel is balanced: all {balance_results['total_tickers']} tickers have {balance_results['min_observations']} observations")
        else:
            print(f"✗ Panel is unbalanced:")
            print(f"  Observation counts range: {balance_results['min_observations']}-{balance_results['max_observations']}")
            print(f"  Most common count: {balance_results['most_common_count']}")
            
            if 'count_distribution' in balance_results:
                print(f"  Distribution: {dict(list(balance_results['count_distribution'].items())[:5])}")
    
    return balance_results


def rebalance_panel_to_common_period(df: pd.DataFrame,
                                   ticker_column: str = DEFAULT_TICKER_COLUMN,
                                   date_column: str = 'quarter',
                                   method: str = 'most_common',
                                   verbose: bool = True) -> Tuple[pd.DataFrame, Dict]:
    
    if verbose:
        print_subsection_header("Rebalancing Panel to Common Period")
    
    original_shape = df.shape
    balance_check = validate_panel_balance(df, ticker_column, date_column, verbose=False)
    
    if balance_check['balanced']:
        if verbose:
            print("Panel is already balanced, no changes needed")
        return df.copy(), {'status': 'already_balanced', 'original_shape': original_shape}
    
    if method == 'most_common':
        target_count = balance_check['most_common_count']
    elif method == 'minimum':
        target_count = balance_check['min_observations']
    elif method == 'maximum':
        target_count = balance_check['max_observations']
    else:
        raise ValueError(f"Unknown rebalancing method: {method}")
    
    if verbose:
        print(f"Rebalancing to {target_count} observations per ticker using '{method}' method")
    
    rebalanced_data = []
    tickers_modified = []
    
    for ticker in df[ticker_column].unique():
        ticker_data = df[df[ticker_column] == ticker].sort_values(date_column)
        
        if len(ticker_data) == target_count:
            rebalanced_data.append(ticker_data)
        elif len(ticker_data) > target_count:
            if method == 'most_common' or method == 'minimum':
                rebalanced_data.append(ticker_data.tail(target_count))
            else:
                rebalanced_data.append(ticker_data.head(target_count))
            tickers_modified.append(ticker)
        else:
            if verbose:
                print(f"⚠️  Ticker {ticker} has only {len(ticker_data)} observations, less than target {target_count}")
            continue
    
    if rebalanced_data:
        df_rebalanced = pd.concat(rebalanced_data, ignore_index=True)
    else:
        df_rebalanced = pd.DataFrame()
    
    stats = {
        'original_shape': original_shape,
        'final_shape': df_rebalanced.shape,
        'target_observations': target_count,
        'tickers_modified': len(tickers_modified),
        'tickers_removed': balance_check['total_tickers'] - len(rebalanced_data),
        'method_used': method,
        'rebalancing_successful': len(rebalanced_data) > 0
    }
    
    if verbose:
        print(f"Rebalancing results:")
        print(f"  Target observations per ticker: {target_count}")
        print(f"  Tickers modified: {len(tickers_modified)}")
        print(f"  Final shape: {df_rebalanced.shape}")
        print(f"  Success: {stats['rebalancing_successful']}")
    
    return df_rebalanced, stats


def get_panel_time_coverage(df: pd.DataFrame,
                          ticker_column: str = DEFAULT_TICKER_COLUMN,
                          date_column: str = 'quarter',
                          verbose: bool = True) -> Dict:
    
    if date_column not in df.columns:
        return {'status': 'date_column_not_found'}
    
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        try:
            date_series = pd.to_datetime(df[date_column])
        except:
            return {'status': 'date_conversion_failed'}
    else:
        date_series = df[date_column]
    
    coverage_stats = {
        'total_tickers': df[ticker_column].nunique(),
        'total_periods': len(date_series.unique()),
        'date_range': {
            'start': date_series.min(),
            'end': date_series.max(),
            'span_days': (date_series.max() - date_series.min()).days
        },
        'coverage_by_ticker': {},
        'gaps_detected': []
    }
    
    all_periods = sorted(date_series.unique())
    
    for ticker in df[ticker_column].unique():
        ticker_dates = df[df[ticker_column] == ticker][date_column]
        ticker_periods = sorted(ticker_dates.unique())
        
        coverage_stats['coverage_by_ticker'][ticker] = {
            'periods_covered': len(ticker_periods),
            'coverage_rate': len(ticker_periods) / len(all_periods),
            'first_period': ticker_periods[0] if ticker_periods else None,
            'last_period': ticker_periods[-1] if ticker_periods else None
        }
        
        missing_periods = set(all_periods) - set(ticker_periods)
        if missing_periods:
            coverage_stats['gaps_detected'].append({
                'ticker': ticker,
                'missing_periods': len(missing_periods),
                'sample_missing': sorted(list(missing_periods))[:3]
            })
    
    if verbose:
        print_subsection_header("Panel Time Coverage Analysis")
        print(f"Total tickers: {coverage_stats['total_tickers']}")
        print(f"Total time periods: {coverage_stats['total_periods']}")
        print(f"Date range: {coverage_stats['date_range']['start']} to {coverage_stats['date_range']['end']}")
        print(f"Tickers with gaps: {len(coverage_stats['gaps_detected'])}")
    
    return coverage_stats


def validate_panel_structure_integrity(df: pd.DataFrame,
                                     lower_year: int,
                                     upper_year: int,
                                     ticker_column: str = DEFAULT_TICKER_COLUMN,
                                     date_column: str = 'quarter',
                                     frequency: str = 'quarterly',
                                     verbose: bool = True) -> Dict:
    
    if verbose:
        print_subsection_header("Comprehensive Panel Structure Validation")
    
    validation_results = {
        'timestamp': pd.Timestamp.now(),
        'parameters': {
            'year_range': f"{lower_year}-{upper_year}",
            'frequency': frequency,
            'ticker_column': ticker_column,
            'date_column': date_column
        }
    }
    
    completeness_check = validate_complete_observations(
        df, lower_year, upper_year, ticker_column, frequency, verbose=False
    )
    validation_results['completeness'] = completeness_check
    
    balance_check = validate_panel_balance(df, ticker_column, date_column, verbose=False)
    validation_results['balance'] = balance_check
    
    coverage_check = get_panel_time_coverage(df, ticker_column, date_column, verbose=False)
    validation_results['coverage'] = coverage_check
    
    overall_health = {
        'is_complete': completeness_check['completion_rate'] == 100,
        'is_balanced': balance_check['balanced'],
        'has_gaps': len(coverage_check.get('gaps_detected', [])) > 0,
        'health_score': 0
    }
    
    health_score = 0
    if overall_health['is_complete']:
        health_score += 40
    else:
        health_score += (completeness_check['completion_rate'] / 100) * 40
    
    if overall_health['is_balanced']:
        health_score += 30
    
    if not overall_health['has_gaps']:
        health_score += 30
    
    overall_health['health_score'] = health_score
    validation_results['overall_health'] = overall_health
    
    if verbose:
        print(f"Panel Structure Health Report:")
        print(f"  Completeness: {completeness_check['completion_rate']:.1f}%")
        print(f"  Balanced: {'Yes' if balance_check['balanced'] else 'No'}")
        print(f"  Time gaps: {'Yes' if overall_health['has_gaps'] else 'No'}")
        print(f"  Overall health score: {health_score:.1f}/100")
        
        if health_score >= 90:
            print("  Status: ✓ Excellent panel structure")
        elif health_score >= 70:
            print("  Status: ⚠️  Good panel structure with minor issues")
        else:
            print("  Status: ❌ Panel structure needs attention")
    
    return validation_results
