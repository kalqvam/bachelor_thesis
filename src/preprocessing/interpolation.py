import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Set

from ..utils import (
    DEFAULT_TICKER_COLUMN, print_subsection_header, 
    format_number, print_processing_stats
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


def identify_companies_with_excessive_zeros(df: pd.DataFrame,
                                          columns_to_analyze: List[str],
                                          consecutive_threshold: int = 3,
                                          ticker_column: str = DEFAULT_TICKER_COLUMN,
                                          date_column: str = 'quarter',
                                          verbose: bool = True) -> Set[str]:
    
    if verbose:
        print_subsection_header(f"Identifying Companies with {consecutive_threshold}+ Consecutive Zeros")
    
    companies_to_remove = set()
    
    for ticker in df[ticker_column].unique():
        ticker_data = df[df[ticker_column] == ticker].sort_values(date_column)
        
        if len(ticker_data) < 2:
            continue
        
        for col in columns_to_analyze:
            if col not in ticker_data.columns:
                continue
            
            consecutive_runs = find_consecutive_zeros(ticker_data[col])
            
            if any(run_length > consecutive_threshold for run_length in consecutive_runs):
                companies_to_remove.add(ticker)
                if verbose and len(companies_to_remove) <= 10:
                    print(f"  {ticker}: {max(consecutive_runs)} consecutive zeros in {col}")
                break
    
    if verbose:
        print(f"Found {len(companies_to_remove)} companies with excessive consecutive zeros")
    
    return companies_to_remove


def interpolate_zeros_for_ticker(ticker_data: pd.DataFrame, 
                                columns_to_interpolate: List[str],
                                date_column: str = 'quarter') -> pd.DataFrame:
    
    ticker_data = ticker_data.sort_values(date_column)
    processed_data = ticker_data.copy()
    
    for col in columns_to_interpolate:
        if col in processed_data.columns:
            if (processed_data[col] == 0).any():
                series = processed_data[col].replace(0, np.nan)
                interpolated = series.interpolate(method='linear')
                processed_data[col] = interpolated
    
    return processed_data


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
    
    original_shape = df.shape
    df_copy = df.copy()
    
    if date_column == 'quarter' and date_column in df_copy.columns:
        df_copy[date_column] = pd.to_datetime(df_copy[date_column])
    
    df_copy['year'] = df_copy[date_column].dt.year if date_column in df_copy.columns else None
    df_copy['quarter_num'] = df_copy[date_column].dt.quarter if date_column in df_copy.columns else None
    
    companies_to_remove = identify_companies_with_excessive_zeros(
        df_copy, columns_to_analyze, consecutive_threshold, ticker_column, date_column, verbose
    )
    
    if companies_to_remove:
        df_filtered = df_copy[~df_copy[ticker_column].isin(companies_to_remove)].copy()
        if verbose:
            removed_count = len(companies_to_remove)
            print(f"Removed {removed_count} companies due to excessive consecutive zeros")
    else:
        df_filtered = df_copy.copy()
    
    interpolation_count = 0
    processed_companies = []
    
    for ticker in df_filtered[ticker_column].unique():
        ticker_mask = df_filtered[ticker_column] == ticker
        ticker_data = df_filtered[ticker_mask].copy()
        
        original_ticker_data = ticker_data.copy()
        processed_ticker_data = interpolate_zeros_for_ticker(
            ticker_data, columns_to_analyze, date_column
        )
        
        for col in columns_to_analyze:
            if col in ticker_data.columns:
                zeros_interpolated = (original_ticker_data[col] == 0).sum()
                if zeros_interpolated > 0:
                    interpolation_count += zeros_interpolated
        
        df_filtered.loc[ticker_mask] = processed_ticker_data
        processed_companies.append(ticker)
    
    df_filtered = df_filtered.drop(['year', 'quarter_num'], axis=1, errors='ignore')
    
    final_shape = df_filtered.shape
    
    stats = {
        'original_shape': original_shape,
        'final_shape': final_shape,
        'companies_removed': len(companies_to_remove),
        'companies_processed': len(processed_companies),
        'removed_companies_list': list(companies_to_remove),
        'zeros_interpolated': interpolation_count,
        'consecutive_threshold': consecutive_threshold,
        'columns_analyzed': columns_to_analyze,
        'retention_rate': (final_shape[0] / original_shape[0]) * 100 if original_shape[0] > 0 else 0
    }
    
    if verbose:
        print(f"\nInterpolation processing results:")
        print(f"  Original companies: {df[ticker_column].nunique()}")
        print(f"  Companies removed: {len(companies_to_remove)}")
        print(f"  Companies processed: {len(processed_companies)}")
        print(f"  Zeros interpolated: {format_number(interpolation_count)}")
        print(f"  Retention rate: {stats['retention_rate']:.1f}%")
    
    return df_filtered, stats


def validate_interpolation_results(df_original: pd.DataFrame,
                                 df_processed: pd.DataFrame,
                                 columns_to_check: List[str],
                                 ticker_column: str = DEFAULT_TICKER_COLUMN,
                                 verbose: bool = True) -> Dict:
    
    validation_results = {
        'zeros_before': {},
        'zeros_after': {},
        'reduction_summary': {},
        'data_integrity_check': True,
        'warnings': []
    }
    
    for col in columns_to_check:
        if col in df_original.columns and col in df_processed.columns:
            zeros_before = (df_original[col] == 0).sum()
            zeros_after = (df_processed[col] == 0).sum()
            
            validation_results['zeros_before'][col] = zeros_before
            validation_results['zeros_after'][col] = zeros_after
            validation_results['reduction_summary'][col] = {
                'zeros_removed': zeros_before - zeros_after,
                'reduction_rate': ((zeros_before - zeros_after) / zeros_before * 100) if zeros_before > 0 else 0
            }
    
    if df_original[ticker_column].nunique() != df_processed[ticker_column].nunique():
        companies_removed = df_original[ticker_column].nunique() - df_processed[ticker_column].nunique()
        validation_results['warnings'].append(f"{companies_removed} companies were removed during processing")
    
    for col in columns_to_check:
        if col in df_processed.columns:
            if df_processed[col].isna().sum() > df_original[col].isna().sum():
                validation_results['warnings'].append(f"Column {col} has more NaN values after processing")
                validation_results['data_integrity_check'] = False
    
    if verbose:
        print_subsection_header("Interpolation Validation Results")
        for col, summary in validation_results['reduction_summary'].items():
            print(f"{col}:")
            print(f"  Zeros removed: {summary['zeros_removed']}")
            print(f"  Reduction rate: {summary['reduction_rate']:.1f}%")
        
        if validation_results['warnings']:
            print("Warnings:")
            for warning in validation_results['warnings']:
                print(f"  - {warning}")
        else:
            print("✓ No validation warnings")
    
    return validation_results


def get_zero_analysis_summary(df: pd.DataFrame,
                            columns_to_analyze: List[str],
                            ticker_column: str = DEFAULT_TICKER_COLUMN,
                            verbose: bool = True) -> Dict:
    
    summary = {
        'total_observations': len(df),
        'total_companies': df[ticker_column].nunique(),
        'zero_analysis_by_column': {},
        'companies_with_zeros': {}
    }
    
    for col in columns_to_analyze:
        if col in df.columns:
            zero_count = (df[col] == 0).sum()
            zero_rate = (zero_count / len(df)) * 100
            
            companies_with_zeros = df[df[col] == 0][ticker_column].nunique()
            
            summary['zero_analysis_by_column'][col] = {
                'zero_count': zero_count,
                'zero_rate': zero_rate,
                'companies_affected': companies_with_zeros
            }
            
            summary['companies_with_zeros'][col] = companies_with_zeros
    
    if verbose:
        print_subsection_header("Zero Values Analysis Summary")
        print(f"Dataset: {format_number(summary['total_observations'])} observations, {summary['total_companies']} companies")
        
        for col, analysis in summary['zero_analysis_by_column'].items():
            print(f"{col}:")
            print(f"  Zero values: {format_number(analysis['zero_count'])} ({analysis['zero_rate']:.1f}%)")
            print(f"  Companies affected: {analysis['companies_affected']}")
    
    return summary


def create_interpolation_report(original_df: pd.DataFrame,
                              processed_df: pd.DataFrame,
                              processing_stats: Dict,
                              columns_analyzed: List[str],
                              ticker_column: str = DEFAULT_TICKER_COLUMN) -> Dict:
    
    report = {
        'processing_summary': processing_stats,
        'data_quality_before': get_zero_analysis_summary(
            original_df, columns_analyzed, ticker_column, verbose=False
        ),
        'data_quality_after': get_zero_analysis_summary(
            processed_df, columns_analyzed, ticker_column, verbose=False
        ),
        'validation_results': validate_interpolation_results(
            original_df, processed_df, columns_analyzed, ticker_column, verbose=False
        ),
        'recommendations': []
    }
    
    for col in columns_analyzed:
        if col in report['validation_results']['reduction_summary']:
            reduction = report['validation_results']['reduction_summary'][col]
            if reduction['reduction_rate'] > 80:
                report['recommendations'].append(
                    f"Excellent zero reduction in {col} ({reduction['reduction_rate']:.1f}%)"
                )
            elif reduction['reduction_rate'] < 20:
                report['recommendations'].append(
                    f"Consider alternative approaches for {col} (only {reduction['reduction_rate']:.1f}% reduction)"
                )
    
    if processing_stats['companies_removed'] > processing_stats['companies_processed'] * 0.1:
        report['recommendations'].append(
            "High company removal rate - consider adjusting consecutive threshold"
        )
    
    return report
