import pandas as pd
import numpy as np
import re
from typing import List, Dict, Optional, Union, Tuple
from datetime import datetime

from .constants import (
    DEFAULT_TICKER_COLUMN, DEFAULT_DATE_COLUMNS, FINANCIAL_COLUMNS,
    ESG_COLUMNS, QUARTER_YEAR_PATTERN
)


class DataValidationError(Exception):
    pass


def validate_required_columns(df: pd.DataFrame, required_columns: List[str]) -> None:
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise DataValidationError(f"Missing required columns: {missing_columns}")


def validate_ticker_column(df: pd.DataFrame, ticker_column: str = DEFAULT_TICKER_COLUMN) -> None:
    if ticker_column not in df.columns:
        raise DataValidationError(f"Ticker column '{ticker_column}' not found")
    
    if df[ticker_column].isna().any():
        na_count = df[ticker_column].isna().sum()
        raise DataValidationError(f"Ticker column contains {na_count} missing values")
    
    if not df[ticker_column].dtype == 'object':
        raise DataValidationError(f"Ticker column should be string type, found {df[ticker_column].dtype}")


def validate_date_column(df: pd.DataFrame, date_columns: Optional[List[str]] = None) -> str:
    if date_columns is None:
        date_columns = DEFAULT_DATE_COLUMNS
    
    available_date_cols = [col for col in date_columns if col in df.columns]
    
    if not available_date_cols:
        raise DataValidationError(f"No date columns found. Expected one of: {date_columns}")
    
    return available_date_cols[0]


def validate_numeric_columns(df: pd.DataFrame, columns: List[str]) -> Dict[str, List[str]]:
    results = {'valid': [], 'invalid': [], 'missing': []}
    
    for col in columns:
        if col not in df.columns:
            results['missing'].append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            results['valid'].append(col)
        else:
            results['invalid'].append(col)
    
    return results


def validate_financial_columns(df: pd.DataFrame, 
                             required_columns: Optional[List[str]] = None) -> Dict[str, List[str]]:
    if required_columns is None:
        required_columns = FINANCIAL_COLUMNS
    
    return validate_numeric_columns(df, required_columns)


def validate_esg_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    return validate_numeric_columns(df, ESG_COLUMNS)


def validate_quarter_year_format(series: pd.Series) -> Tuple[bool, List[str]]:
    invalid_formats = []
    
    for value in series.dropna().unique():
        if not isinstance(value, str):
            invalid_formats.append(str(value))
            continue
        
        if not re.match(QUARTER_YEAR_PATTERN, value):
            invalid_formats.append(value)
    
    is_valid = len(invalid_formats) == 0
    return is_valid, invalid_formats


def validate_date_format(series: pd.Series, 
                        expected_format: Optional[str] = None) -> Tuple[bool, List[str]]:
    errors = []
    
    if pd.api.types.is_datetime64_any_dtype(series):
        return True, []
    
    sample_values = series.dropna().head(10).tolist()
    
    for value in sample_values:
        try:
            if expected_format:
                datetime.strptime(str(value), expected_format)
            else:
                pd.to_datetime(value)
        except (ValueError, TypeError) as e:
            errors.append(f"{value}: {str(e)}")
    
    return len(errors) == 0, errors


def validate_data_ranges(df: pd.DataFrame, 
                        range_checks: Dict[str, Tuple[float, float]]) -> Dict[str, List]:
    violations = {}
    
    for column, (min_val, max_val) in range_checks.items():
        if column not in df.columns:
            continue
        
        if not pd.api.types.is_numeric_dtype(df[column]):
            continue
        
        column_violations = []
        
        if min_val is not None:
            below_min = df[df[column] < min_val]
            if not below_min.empty:
                column_violations.extend(
                    [f"Row {idx}: {val} < {min_val}" for idx, val in 
                     below_min[column].items()]
                )
        
        if max_val is not None:
            above_max = df[df[column] > max_val]
            if not above_max.empty:
                column_violations.extend(
                    [f"Row {idx}: {val} > {max_val}" for idx, val in 
                     above_max[column].items()]
                )
        
        if column_violations:
            violations[column] = column_violations
    
    return violations


def validate_panel_structure(df: pd.DataFrame, 
                           ticker_column: str = DEFAULT_TICKER_COLUMN,
                           date_column: Optional[str] = None) -> Dict[str, any]:
    if date_column is None:
        date_column = validate_date_column(df)
    
    validation_results = {
        'is_valid_panel': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    try:
        validate_ticker_column(df, ticker_column)
    except DataValidationError as e:
        validation_results['is_valid_panel'] = False
        validation_results['errors'].append(str(e))
        return validation_results
    
    duplicates = df.duplicated(subset=[ticker_column, date_column])
    duplicate_count = duplicates.sum()
    
    if duplicate_count > 0:
        validation_results['warnings'].append(
            f"Found {duplicate_count} duplicate ticker-date combinations"
        )
    
    tickers = df[ticker_column].unique()
    observations_per_ticker = df.groupby(ticker_column).size()
    
    validation_results['stats'] = {
        'total_observations': len(df),
        'unique_tickers': len(tickers),
        'avg_observations_per_ticker': observations_per_ticker.mean(),
        'min_observations_per_ticker': observations_per_ticker.min(),
        'max_observations_per_ticker': observations_per_ticker.max(),
        'duplicate_combinations': duplicate_count
    }
    
    if observations_per_ticker.min() < 2:
        single_obs_tickers = observations_per_ticker[observations_per_ticker < 2].index.tolist()
        validation_results['warnings'].append(
            f"Tickers with single observation: {len(single_obs_tickers)}"
        )
    
    return validation_results


def validate_completeness(df: pd.DataFrame, 
                         columns: List[str],
                         max_missing_rate: float = 0.5) -> Dict[str, Dict]:
    completeness_report = {}
    
    for col in columns:
        if col not in df.columns:
            completeness_report[col] = {
                'status': 'missing_column',
                'missing_count': None,
                'missing_rate': None,
                'passes_threshold': False
            }
            continue
        
        missing_count = df[col].isna().sum()
        missing_rate = missing_count / len(df)
        passes_threshold = missing_rate <= max_missing_rate
        
        completeness_report[col] = {
            'status': 'analyzed',
            'missing_count': missing_count,
            'missing_rate': missing_rate,
            'passes_threshold': passes_threshold
        }
    
    return completeness_report


def validate_consistency(df: pd.DataFrame,
                        consistency_checks: Dict[str, callable]) -> Dict[str, List]:
    violations = {}
    
    for check_name, check_function in consistency_checks.items():
        try:
            check_result = check_function(df)
            if check_result:
                violations[check_name] = check_result
        except Exception as e:
            violations[check_name] = [f"Check failed: {str(e)}"]
    
    return violations


def run_full_validation(df: pd.DataFrame,
                       required_columns: Optional[List[str]] = None,
                       numeric_columns: Optional[List[str]] = None,
                       date_column: Optional[str] = None,
                       ticker_column: str = DEFAULT_TICKER_COLUMN) -> Dict[str, any]:
    
    validation_report = {
        'timestamp': datetime.now(),
        'dataset_shape': df.shape,
        'validation_passed': True,
        'errors': [],
        'warnings': [],
        'details': {}
    }
    
    try:
        if required_columns:
            validate_required_columns(df, required_columns)
            validation_report['details']['required_columns'] = 'passed'
    except DataValidationError as e:
        validation_report['validation_passed'] = False
        validation_report['errors'].append(str(e))
        validation_report['details']['required_columns'] = 'failed'
    
    try:
        validate_ticker_column(df, ticker_column)
        validation_report['details']['ticker_column'] = 'passed'
    except DataValidationError as e:
        validation_report['validation_passed'] = False
        validation_report['errors'].append(str(e))
        validation_report['details']['ticker_column'] = 'failed'
    
    try:
        actual_date_column = validate_date_column(df, [date_column] if date_column else None)
        validation_report['details']['date_column'] = actual_date_column
    except DataValidationError as e:
        validation_report['validation_passed'] =
