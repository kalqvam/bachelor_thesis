import pandas as pd
import numpy as np
import re
from datetime import datetime
from typing import Optional, Tuple, Dict

from ...utils import (
    QUARTER_YEAR_PATTERN, DEFAULT_TICKER_COLUMN, 
    print_subsection_header, format_number, print_processing_stats
)


def convert_quarter_year_to_datetime(quarter_year: str) -> Optional[datetime]:
    if pd.isna(quarter_year) or not isinstance(quarter_year, str):
        return None

    match = re.search(QUARTER_YEAR_PATTERN, quarter_year)
    if match:
        quarter, year = match.groups()
        month = int(quarter) * 3
        return datetime(int(year), month, 1)
    return None


def standardize_quarter_year_format(df: pd.DataFrame,
                                  quarter_year_column: str = 'quarter_year',
                                  verbose: bool = True) -> Tuple[pd.DataFrame, Dict]:
    
    if verbose:
        print_subsection_header("Standardizing Quarter-Year Format")
    
    if quarter_year_column not in df.columns:
        if verbose:
            print(f"Column '{quarter_year_column}' not found, skipping standardization")
        return df.copy(), {'status': 'column_not_found'}
    
    df_result = df.copy()
    original_count = len(df_result)
    conversion_errors = 0
    
    valid_format_mask = df_result[quarter_year_column].apply(
        lambda x: bool(re.match(QUARTER_YEAR_PATTERN, str(x))) if pd.notna(x) else False
    )
    
    invalid_formats = df_result[~valid_format_mask][quarter_year_column].unique()
    
    if len(invalid_formats) > 0 and not all(pd.isna(invalid_formats)):
        if verbose:
            print(f"Found {len(invalid_formats)} invalid quarter-year formats:")
            for fmt in invalid_formats[:5]:
                if pd.notna(fmt):
                    print(f"  {fmt}")
            
        conversion_errors = len(df_result[~valid_format_mask])
    
    stats = {
        'original_count': original_count,
        'valid_formats': valid_format_mask.sum(),
        'invalid_formats': conversion_errors,
        'conversion_success_rate': (valid_format_mask.sum() / original_count) * 100 if original_count > 0 else 0
    }
    
    if verbose:
        print(f"Quarter-year format validation:")
        print(f"  Valid formats: {format_number(stats['valid_formats'])}")
        print(f"  Invalid formats: {format_number(stats['invalid_formats'])}")
        print(f"  Success rate: {stats['conversion_success_rate']:.1f}%")
    
    return df_result, stats


def add_datetime_from_quarter_year(df: pd.DataFrame,
                                 quarter_year_column: str = 'quarter_year',
                                 new_column_name: str = 'quarter',
                                 verbose: bool = True) -> Tuple[pd.DataFrame, Dict]:
    
    if verbose:
        print_subsection_header("Converting Quarter-Year to Datetime")
    
    if quarter_year_column not in df.columns:
        if verbose:
            print(f"Column '{quarter_year_column}' not found")
        return df.copy(), {'status': 'column_not_found'}
    
    df_result = df.copy()
    
    df_result[new_column_name] = df_result[quarter_year_column].apply(convert_quarter_year_to_datetime)
    
    successful_conversions = df_result[new_column_name].notna().sum()
    total_rows = len(df_result)
    failed_conversions = total_rows - successful_conversions
    
    stats = {
        'total_rows': total_rows,
        'successful_conversions': successful_conversions,
        'failed_conversions': failed_conversions,
        'conversion_rate': (successful_conversions / total_rows) * 100 if total_rows > 0 else 0,
        'new_column': new_column_name
    }
    
    if verbose:
        print(f"Datetime conversion results:")
        print(f"  Successful: {format_number(successful_conversions)} ({stats['conversion_rate']:.1f}%)")
        print(f"  Failed: {format_number(failed_conversions)}")
        print(f"  New column created: '{new_column_name}'")
    
    return df_result, stats


def filter_by_year_range(df: pd.DataFrame,
                        lower_year: int,
                        upper_year: int,
                        date_column: str = 'quarter_year',
                        verbose: bool = True) -> Tuple[pd.DataFrame, Dict]:
    
    if verbose:
        print_subsection_header(f"Filtering by Year Range ({lower_year}-{upper_year})")
    
    if date_column not in df.columns:
        if verbose:
            print(f"Column '{date_column}' not found")
        return df.copy(), {'status': 'column_not_found'}
    
    original_shape = df.shape
    
    if date_column == 'quarter_year':
        df['_temp_year'] = df[date_column].apply(
            lambda x: int(re.search(r'Q\d-(\d{4})', str(x)).group(1)) if isinstance(x, str) and re.search(r'Q\d-(\d{4})', str(x)) else np.nan
        )
    elif pd.api.types.is_datetime64_any_dtype(df[date_column]):
        df['_temp_year'] = df[date_column].dt.year
    else:
        try:
            df['_temp_year'] = pd.to_datetime(df[date_column]).dt.year
        except:
            if verbose:
                print(f"Could not extract year from column '{date_column}'")
            return df.copy(), {'status': 'year_extraction_failed'}
    
    year_mask = (df['_temp_year'] >= lower_year) & (df['_temp_year'] <= upper_year)
    df_filtered = df[year_mask].copy()
    df_filtered = df_filtered.drop('_temp_year', axis=1)
    
    removed_rows = original_shape[0] - df_filtered.shape[0]
    
    stats = {
        'original_shape': original_shape,
        'final_shape': df_filtered.shape,
        'rows_removed': removed_rows,
        'retention_rate': (df_filtered.shape[0] / original_shape[0]) * 100 if original_shape[0] > 0 else 0,
        'year_range': f"{lower_year}-{upper_year}"
    }
    
    if verbose:
        print(f"Year filtering results:")
        print(f"  Original: {format_number(original_shape[0])} rows")
        print(f"  Filtered: {format_number(df_filtered.shape[0])} rows")
        print(f"  Removed: {format_number(removed_rows)} rows")
        print(f"  Retention: {stats['retention_rate']:.1f}%")
    
    return df_filtered, stats


def extract_year_quarter_components(df: pd.DataFrame,
                                   date_column: str = 'quarter_year',
                                   add_year_column: bool = True,
                                   add_quarter_column: bool = True,
                                   verbose: bool = True) -> Tuple[pd.DataFrame, Dict]:
    
    if verbose:
        print_subsection_header("Extracting Year and Quarter Components")
    
    if date_column not in df.columns:
        if verbose:
            print(f"Column '{date_column}' not found")
        return df.copy(), {'status': 'column_not_found'}
    
    df_result = df.copy()
    columns_added = []
    
    if date_column == 'quarter_year':
        if add_year_column:
            df_result['year'] = df_result[date_column].apply(
                lambda x: int(re.search(r'Q\d-(\d{4})', str(x)).group(1)) if isinstance(x, str) and re.search(r'Q\d-(\d{4})', str(x)) else np.nan
            )
            columns_added.append('year')
        
        if add_quarter_column:
            df_result['quarter_num'] = df_result[date_column].apply(
                lambda x: int(re.search(r'Q(\d)-\d{4}', str(x)).group(1)) if isinstance(x, str) and re.search(r'Q(\d)-\d{4}', str(x)) else np.nan
            )
            columns_added.append('quarter_num')
    
    elif pd.api.types.is_datetime64_any_dtype(df_result[date_column]):
        if add_year_column:
            df_result['year'] = df_result[date_column].dt.year
            columns_added.append('year')
        
        if add_quarter_column:
            df_result['quarter_num'] = df_result[date_column].dt.quarter
            columns_added.append('quarter_num')
    
    stats = {
        'columns_added': columns_added,
        'total_rows': len(df_result),
        'extraction_method': 'regex' if date_column == 'quarter_year' else 'datetime'
    }
    
    if verbose:
        print(f"Component extraction results:")
        print(f"  Columns added: {columns_added}")
        print(f"  Method used: {stats['extraction_method']}")
    
    return df_result, stats


def validate_date_consistency(df: pd.DataFrame,
                            date_columns: list,
                            ticker_column: str = DEFAULT_TICKER_COLUMN,
                            verbose: bool = True) -> Dict:
    
    if verbose:
        print_subsection_header("Validating Date Consistency")
    
    validation_results = {
        'consistent': True,
        'issues': [],
        'columns_checked': date_columns
    }
    
    existing_columns = [col for col in date_columns if col in df.columns]
    
    if len(existing_columns) < 2:
        validation_results['issues'].append("Need at least 2 date columns for consistency check")
        validation_results['consistent'] = False
        return validation_results
    
    for ticker in df[ticker_column].unique():
        ticker_data = df[df[ticker_column] == ticker]
        
        for i in range(len(existing_columns) - 1):
            col1, col2 = existing_columns[i], existing_columns[i + 1]
            
            if col1 in ticker_data.columns and col2 in ticker_data.columns:
                inconsistent_rows = 0
                
                for _, row in ticker_data.iterrows():
                    val1, val2 = row[col1], row[col2]
                    
                    if pd.notna(val1) and pd.notna(val2):
                        if col1 == 'quarter_year' and pd.api.types.is_datetime64_any_dtype(ticker_data[col2]):
                            expected_date = convert_quarter_year_to_datetime(val1)
                            if expected_date and abs((expected_date - val2).days) > 31:
                                inconsistent_rows += 1
                
                if inconsistent_rows > 0:
                    validation_results['issues'].append(
                        f"Ticker {ticker}: {inconsistent_rows} inconsistent dates between {col1} and {col2}"
                    )
                    validation_results['consistent'] = False
    
    if verbose:
        if validation_results['consistent']:
            print("✓ All date columns are consistent")
        else:
            print(f"✗ Found {len(validation_results['issues'])} consistency issues")
            for issue in validation_results['issues'][:5]:
                print(f"  {issue}")
    
    return validation_results


def standardize_date_columns(df: pd.DataFrame,
                           quarter_year_column: str = 'quarter_year',
                           target_position: int = 1,
                           verbose: bool = True) -> Tuple[pd.DataFrame, Dict]:
    
    if verbose:
        print_subsection_header("Standardizing Date Column Structure")
    
    df_result = df.copy()
    operations = []
    
    if quarter_year_column in df_result.columns and 'quarter' not in df_result.columns:
        df_result, conversion_stats = add_datetime_from_quarter_year(
            df_result, quarter_year_column, 'quarter', verbose=False
        )
        operations.append('added_quarter_datetime')
    
    if 'quarter' in df_result.columns and target_position is not None:
        cols = df_result.columns.tolist()
        if 'quarter' in cols:
            cols.remove('quarter')
            cols.insert(target_position, 'quarter')
            df_result = df_result[cols]
            operations.append(f'moved_quarter_to_position_{target_position}')
    
    stats = {
        'operations_performed': operations,
        'final_shape': df_result.shape,
        'date_columns_present': [col for col in ['quarter', 'quarter_year', 'date'] if col in df_result.columns]
    }
    
    if verbose:
        print(f"Date standardization results:")
        print(f"  Operations: {operations}")
        print(f"  Date columns present: {stats['date_columns_present']}")
    
    return df_result, stats
