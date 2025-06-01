import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, Union
from pathlib import Path

from .constants import (
    CONSOLE_WIDTH, CONSOLE_SEPARATORS, NUMERIC_PRECISION, 
    PERCENTAGE_PRECISION, DISPLAY_TIMESTAMP_FORMAT
)


def print_separator(sep_type: str = 'major'):
    print(CONSOLE_SEPARATORS.get(sep_type, CONSOLE_SEPARATORS['major']))


def print_section_header(title: str, sep_type: str = 'major'):
    print_separator(sep_type)
    print(f"{title.upper()}")
    print_separator(sep_type)


def print_subsection_header(title: str):
    print(f"\n{title}:")
    print(CONSOLE_SEPARATORS['minor'])


def format_number(value: Union[int, float], precision: int = NUMERIC_PRECISION) -> str:
    if pd.isna(value):
        return 'N/A'
    
    if isinstance(value, int) or (isinstance(value, float) and value.is_integer()):
        return f"{int(value):,}"
    
    return f"{value:,.{precision}f}"


def format_percentage(value: Union[int, float], precision: int = PERCENTAGE_PRECISION) -> str:
    if pd.isna(value):
        return 'N/A'
    
    return f"{value:.{precision}f}%"


def format_timestamp(timestamp: Optional[datetime] = None) -> str:
    if timestamp is None:
        timestamp = datetime.now()
    return timestamp.strftime(DISPLAY_TIMESTAMP_FORMAT)


def print_dataset_info(df: pd.DataFrame, dataset_name: str = "Dataset"):
    print(f"\n{dataset_name} Information:")
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    if 'ticker' in df.columns:
        print(f"Unique companies: {df['ticker'].nunique():,}")
    
    if any(col in df.columns for col in ['date', 'quarter', 'quarter_year']):
        date_col = next((col for col in ['date', 'quarter', 'quarter_year'] if col in df.columns), None)
        if date_col:
            try:
                date_series = pd.to_datetime(df[date_col])
                print(f"Date range: {date_series.min()} to {date_series.max()}")
            except:
                print(f"Date column: {date_col} (format not parsed)")


def print_missing_data_summary(df: pd.DataFrame, columns: Optional[list] = None):
    if columns is None:
        columns = df.columns.tolist()
    
    missing_info = []
    total_rows = len(df)
    
    for col in columns:
        if col in df.columns:
            missing_count = df[col].isna().sum()
            missing_pct = (missing_count / total_rows) * 100
            missing_info.append({
                'Column': col,
                'Missing': missing_count,
                'Percentage': missing_pct
            })
    
    if missing_info:
        missing_df = pd.DataFrame(missing_info)
        missing_df = missing_df[missing_df['Missing'] > 0].sort_values('Percentage', ascending=False)
        
        if not missing_df.empty:
            print("\nMissing Data Summary:")
            for _, row in missing_df.iterrows():
                print(f"  {row['Column']}: {row['Missing']:,} ({row['Percentage']:.1f}%)")
        else:
            print("\nNo missing data found in analyzed columns.")


def print_transformation_summary(stats: Dict[str, Any]):
    if 'original_shape' in stats and 'final_shape' in stats:
        orig_rows, orig_cols = stats['original_shape']
        final_rows, final_cols = stats['final_shape']
        
        rows_removed = orig_rows - final_rows
        retention_pct = (final_rows / orig_rows) * 100 if orig_rows > 0 else 0
        
        print(f"\nTransformation Summary:")
        print(f"Original: {orig_rows:,} rows × {orig_cols} columns")
        print(f"Final: {final_rows:,} rows × {final_cols} columns")
        print(f"Removed: {rows_removed:,} rows ({100 - retention_pct:.1f}%)")
        print(f"Retained: {retention_pct:.1f}% of original data")


def print_processing_stats(stats: Dict[str, Any], title: str = "Processing Statistics"):
    print_subsection_header(title)
    
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"{key.replace('_', ' ').title()}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key.replace('_', ' ').title()}: {format_number(sub_value)}")
        elif isinstance(value, (int, float)):
            print(f"{key.replace('_', ' ').title()}: {format_number(value)}")
        elif isinstance(value, list):
            if len(value) <= 10:
                print(f"{key.replace('_', ' ').title()}: {', '.join(map(str, value))}")
            else:
                print(f"{key.replace('_', ' ').title()}: {len(value)} items")
        else:
            print(f"{key.replace('_', ' ').title()}: {value}")


def print_correlation_matrix(corr_matrix: pd.DataFrame, title: str = "Correlation Matrix"):
    print_subsection_header(title)
    print(corr_matrix.round(3))


def print_file_operation(operation: str, file_path: Union[str, Path], success: bool = True):
    path_str = str(file_path)
    status = "✓" if success else "✗"
    print(f"{status} {operation}: {path_str}")


def print_progress_update(current: int, total: int, item_name: str = "items"):
    percentage = (current / total) * 100 if total > 0 else 0
    print(f"Progress: {current:,}/{total:,} {item_name} ({percentage:.1f}%)")


def print_error_summary(errors: list, max_display: int = 10):
    if not errors:
        return
    
    print_subsection_header("Errors Encountered")
    
    displayed = min(len(errors), max_display)
    for i, error in enumerate(errors[:displayed]):
        print(f"  {i+1}. {error}")
    
    if len(errors) > max_display:
        print(f"  ... and {len(errors) - max_display} more errors")


def print_test_results(results: Dict[str, Dict], test_name: str = "Statistical Tests"):
    print_section_header(test_name)
    
    for variable, result in results.items():
        print(f"\nVariable: {variable}")
        
        if 'error' in result:
            print(f"  Error: {result['error']}")
            continue
        
        for key, value in result.items():
            if key == 'error':
                continue
            
            if isinstance(value, float):
                if 'p_value' in key or 'pvalue' in key:
                    significance = "***" if value <= 0.01 else "**" if value <= 0.05 else "*" if value <= 0.1 else ""
                    print(f"  {key.replace('_', ' ').title()}: {value:.4f} {significance}")
                else:
                    print(f"  {key.replace('_', ' ').title()}: {format_number(value)}")
            elif isinstance(value, bool):
                print(f"  {key.replace('_', ' ').title()}: {'Yes' if value else 'No'}")
            else:
                print(f"  {key.replace('_', ' ').title()}: {value}")


def log_execution_time(start_time: datetime, end_time: Optional[datetime] = None, 
                      operation: str = "Operation"):
    if end_time is None:
        end_time = datetime.now()
    
    duration = end_time - start_time
    total_seconds = duration.total_seconds()
    
    if total_seconds < 60:
        time_str = f"{total_seconds:.1f} seconds"
    elif total_seconds < 3600:
        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)
        time_str = f"{minutes}m {seconds}s"
    else:
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        time_str = f"{hours}h {minutes}m"
    
    print(f"{operation} completed in {time_str}")


def create_summary_report(stats: Dict[str, Any], title: str = "Analysis Summary") -> str:
    report_lines = [
        f"{'=' * 50}",
        f"{title.upper()}",
        f"{'=' * 50}",
        f"Generated: {format_timestamp()}",
        ""
    ]
    
    for section, data in stats.items():
        report_lines.append(f"{section.replace('_', ' ').title()}:")
        
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (int, float)):
                    report_lines.append(f"  {key}: {format_number(value)}")
                else:
                    report_lines.append(f"  {key}: {value}")
        else:
            report_lines.append(f"  {data}")
        
        report_lines.append("")
    
    return "\n".join(report_lines)
