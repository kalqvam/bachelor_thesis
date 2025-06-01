import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional

from ...utils import (
    DEFAULT_TICKER_COLUMN, DEFAULT_CONSECUTIVE_MISSING_THRESHOLD,
    print_subsection_header, format_number, print_processing_stats
)


def check_consecutive_missing(series: pd.Series, threshold: int = DEFAULT_CONSECUTIVE_MISSING_THRESHOLD) -> bool:
    max_consecutive = 0
    current_streak = 0

    for value in series:
        if pd.isna(value):
            current_streak += 1
            max_consecutive = max(max_consecutive, current_streak)
            if max_consecutive >= threshold:
                return True
        else:
            current_streak = 0

    return False


def analyze_missing_patterns(df: pd.DataFrame,
                           columns: Optional[List[str]] = None,
                           ticker_column: str = DEFAULT_TICKER_COLUMN,
                           verbose: bool = True) -> Dict:
    
    if verbose:
        print_subsection_header("Analyzing Missing Data Patterns")
    
    if columns is None:
        columns = [col for col in df.columns if col != ticker_column]
    
    missing_analysis = {
        'overall_stats': {},
        'by_ticker': {},
        'by_column': {},
        'patterns': {}
    }
    
    total_cells = len(df) * len(columns)
    total_missing = df[columns].isna().sum().sum()
    
    missing_analysis['overall_stats'] = {
        'total_cells': total_cells,
        'total_missing': total_missing,
        'missing_percentage': (total_missing / total_cells) * 100 if total_cells > 0 else 0
    }
    
    for column in columns:
        if column in df.columns:
            col_missing = df[column].isna().sum()
            col_total = len(df)
            
            missing_analysis['by_column'][column] = {
                'missing_count': col_missing,
                'missing_percentage': (col_missing / col_total) * 100 if col_total > 0 else 0,
                'tickers_affected': df[df[column].isna()][ticker_column].nunique()
            }
    
    for ticker in df[ticker_column].unique():
        ticker_data = df[df[ticker_column] == ticker]
        ticker_missing = ticker_data[columns].isna().sum().sum()
        ticker_total = len(ticker_data) * len(columns)
        
        missing_analysis['by_ticker'][ticker] = {
            'missing_count': ticker_missing,
            'missing_percentage': (ticker_missing / ticker_total) * 100 if ticker_total > 0 else 0
        }
    
    completely_missing_cols = [col for col, stats in missing_analysis['by_column'].items() 
                              if stats['missing_percentage'] == 100]
    
    mostly_missing_cols = [col for col, stats in missing_analysis['by_column'].items() 
                          if 50 <= stats['missing_percentage'] < 100]
    
    missing_analysis['patterns'] = {
        'completely_missing_columns': completely_missing_cols,
        'mostly_missing_columns': mostly_missing_cols,
        'columns_with_no_missing': [col for col, stats in missing_analysis['by_column'].items() 
                                   if stats['missing_percentage'] == 0]
    }
    
    if verbose:
        print(f"Overall missing data: {format_number(total_missing)} / {format_number(total_cells)} cells ({missing_analysis['overall_stats']['missing_percentage']:.1f}%)")
        print(f"Completely missing columns: {len(completely_missing_cols)}")
        print(f"Mostly missing columns (50%+): {len(mostly_missing_cols)}")
        
        if completely_missing_cols:
            print(f"  Completely missing: {completely_missing_cols}")
        if mostly_missing_cols:
            print(f"  Mostly missing: {mostly_missing_cols}")
    
    return missing_analysis


def identify_consecutive_missing_tickers(df: pd.DataFrame,
                                       target_column: str,
                                       threshold: int = DEFAULT_CONSECUTIVE_MISSING_THRESHOLD,
                                       ticker_column: str = DEFAULT_TICKER_COLUMN,
                                       date_column: str = 'quarter',
                                       verbose: bool = True) -> List[str]:
    
    if verbose:
        print_subsection_header(f"Identifying Tickers with {threshold}+ Consecutive Missing {target_column}")
    
    if target_column not in df.columns:
        if verbose:
            print(f"Column '{target_column}' not found")
        return []
    
    consecutive_missing_tickers = []
    
    for ticker in df[ticker_column].unique():
        ticker_data = df[df[ticker_column] == ticker].sort_values(date_column)
        
        if check_consecutive_missing(ticker_data[target_column], threshold):
            consecutive_missing_tickers.append(ticker)
    
    if verbose:
        print(f"Found {len(consecutive_missing_tickers)} tickers with {threshold}+ consecutive missing {target_column}")
        if consecutive_missing_tickers and len(consecutive_missing_tickers) <= 10:
            print(f"Tickers: {consecutive_missing_tickers}")
        elif len(consecutive_missing_tickers) > 10:
            print(f"Sample tickers: {consecutive_missing_tickers[:10]}...")
    
    return consecutive_missing_tickers


def remove_consecutive_missing_tickers(df: pd.DataFrame,
                                     target_column: str,
                                     threshold: int = DEFAULT_CONSECUTIVE_MISSING_THRESHOLD,
                                     ticker_column: str = DEFAULT_TICKER_COLUMN,
                                     date_column: str = 'quarter',
                                     verbose: bool = True) -> Tuple[pd.DataFrame, Dict]:
    
    if verbose:
        print_subsection_header(f"Removing Tickers with Consecutive Missing {target_column}")
    
    original_shape = df.shape
    
    consecutive_missing_tickers = identify_consecutive_missing_tickers(
        df, target_column, threshold, ticker_column, date_column, verbose=False
    )
    
    if not consecutive_missing_tickers:
        if verbose:
            print(f"No tickers found with {threshold}+ consecutive missing {target_column}")
        
        stats = {
            'original_shape': original_shape,
            'final_shape': df.shape,
            'tickers_removed': 0,
            'rows_removed': 0,
            'threshold_used': threshold,
            'target_column': target_column
        }
        return df.copy(), stats
    
    df_filtered = df[~df[ticker_column].isin(consecutive_missing_tickers)].copy()
    
    rows_removed = original_shape[0] - df_filtered.shape[0]
    
    stats = {
        'original_shape': original_shape,
        'final_shape': df_filtered.shape,
        'tickers_removed': len(consecutive_missing_tickers),
        'tickers_removed_list': consecutive_missing_tickers,
        'rows_removed': rows_removed,
        'threshold_used': threshold,
        'target_column': target_column,
        'retention_rate': (df_filtered.shape[0] / original_shape[0]) * 100 if original_shape[0] > 0 else 0
    }
    
    if verbose:
        print(f"Consecutive missing removal results:")
        print(f"  Threshold: {threshold} consecutive periods")
        print(f"  Tickers removed: {len(consecutive_missing_tickers)}")
        print(f"  Rows removed: {format_number(rows_removed)}")
        print(f"  Retention rate: {stats['retention_rate']:.1f}%")
        
        if len(consecutive_missing_tickers) <= 5:
            print(f"  Removed tickers: {consecutive_missing_tickers}")
        elif len(consecutive_missing_tickers) > 5:
            print(f"  Sample removed tickers: {consecutive_missing_tickers[:5]}...")
    
    return df_filtered, stats


def get_missing_data_summary(df: pd.DataFrame,
                           columns: Optional[List[str]] = None,
                           ticker_column: str = DEFAULT_TICKER_COLUMN,
                           verbose: bool = True) -> Dict:
    
    if columns is None:
        columns = [col for col in df.columns if col != ticker_column]
    
    summary = {
        'total_observations': len(df),
        'total_tickers': df[ticker_column].nunique(),
        'columns_analyzed': len(columns),
        'missing_by_column': {},
        'tickers_with_missing': {},
        'recommendations': []
    }
    
    for column in columns:
        if column in df.columns:
            missing_count = df[column].isna().sum()
            missing_pct = (missing_count / len(df)) * 100
            
            summary['missing_by_column'][column] = {
                'count': missing_count,
                'percentage': missing_pct
            }
            
            tickers_with_missing = df[df[column].isna()][ticker_column].nunique()
            summary['tickers_with_missing'][column] = tickers_with_missing
            
            if missing_pct > 70:
                summary['recommendations'].append(f"Consider removing column '{column}' (>{missing_pct:.1f}% missing)")
            elif missing_pct > 30:
                summary['recommendations'].append(f"Consider imputation for column '{column}' ({missing_pct:.1f}% missing)")
    
    if verbose:
        print_subsection_header("Missing Data Summary Report")
        print(f"Dataset: {format_number(summary['total_observations'])} observations, {summary['total_tickers']} tickers")
        print(f"Columns analyzed: {summary['columns_analyzed']}")
        
        print("\nMissing data by column:")
        for column, stats in summary['missing_by_column'].items():
            print(f"  {column}: {format_number(stats['count'])} ({stats['percentage']:.1f}%)")
        
        if summary['recommendations']:
            print("\nRecommendations:")
            for rec in summary['recommendations']:
                print(f"  - {rec}")
    
    return summary


def identify_problematic_tickers(df: pd.DataFrame,
                               columns: Optional[List[str]] = None,
                               missing_threshold: float = 0.5,
                               ticker_column: str = DEFAULT_TICKER_COLUMN,
                               verbose: bool = True) -> List[str]:
    
    if verbose:
        print_subsection_header(f"Identifying Problematic Tickers (>{missing_threshold:.0%} missing)")
    
    if columns is None:
        columns = [col for col in df.columns if col != ticker_column]
    
    problematic_tickers = []
    
    for ticker in df[ticker_column].unique():
        ticker_data = df[df[ticker_column] == ticker]
        
        total_cells = len(ticker_data) * len(columns)
        missing_cells = ticker_data[columns].isna().sum().sum()
        missing_rate = missing_cells / total_cells if total_cells > 0 else 0
        
        if missing_rate > missing_threshold:
            problematic_tickers.append(ticker)
    
    if verbose:
        print(f"Found {len(problematic_tickers)} tickers with >{missing_threshold:.0%} missing data")
        if problematic_tickers and len(problematic_tickers) <= 10:
            print(f"Problematic tickers: {problematic_tickers}")
        elif len(problematic_tickers) > 10:
            print(f"Sample problematic tickers: {problematic_tickers[:10]}...")
    
    return problematic_tickers


def remove_high_missing_tickers(df: pd.DataFrame,
                              columns: Optional[List[str]] = None,
                              missing_threshold: float = 0.5,
                              ticker_column: str = DEFAULT_TICKER_COLUMN,
                              verbose: bool = True) -> Tuple[pd.DataFrame, Dict]:
    
    if verbose:
        print_subsection_header(f"Removing High Missing Data Tickers (>{missing_threshold:.0%})")
    
    original_shape = df.shape
    
    problematic_tickers = identify_problematic_tickers(
        df, columns, missing_threshold, ticker_column, verbose=False
    )
    
    if not problematic_tickers:
        if verbose:
            print(f"No tickers found with >{missing_threshold:.0%} missing data")
        
        stats = {
            'original_shape': original_shape,
            'final_shape': df.shape,
            'tickers_removed': 0,
            'rows_removed': 0,
            'threshold_used': missing_threshold
        }
        return df.copy(), stats
    
    df_filtered = df[~df[ticker_column].isin(problematic_tickers)].copy()
    
    rows_removed = original_shape[0] - df_filtered.shape[0]
    
    stats = {
        'original_shape': original_shape,
        'final_shape': df_filtered.shape,
        'tickers_removed': len(problematic_tickers),
        'tickers_removed_list': problematic_tickers,
        'rows_removed': rows_removed,
        'threshold_used': missing_threshold,
        'retention_rate': (df_filtered.shape[0] / original_shape[0]) * 100 if original_shape[0] > 0 else 0
    }
    
    if verbose:
        print(f"High missing data removal results:")
        print(f"  Threshold: >{missing_threshold:.0%} missing")
        print(f"  Tickers removed: {len(problematic_tickers)}")
        print(f"  Rows removed: {format_number(rows_removed)}")
        print(f"  Retention rate: {stats['retention_rate']:.1f}%")
    
    return df_filtered, stats


def create_missing_data_report(df: pd.DataFrame,
                             ticker_column: str = DEFAULT_TICKER_COLUMN,
                             save_path: Optional[str] = None,
                             verbose: bool = True) -> pd.DataFrame:
    
    if verbose:
        print_subsection_header("Creating Detailed Missing Data Report")
    
    columns_to_analyze = [col for col in df.columns if col != ticker_column]
    
    report_data = []
    
    for ticker in df[ticker_column].unique():
        ticker_data = df[df[ticker_column] == ticker]
        
        ticker_report = {
            'ticker': ticker,
            'total_observations': len(ticker_data),
            'total_possible_cells': len(ticker_data) * len(columns_to_analyze),
            'total_missing_cells': ticker_data[columns_to_analyze].isna().sum().sum(),
        }
        
        ticker_report['overall_missing_rate'] = (
            ticker_report['total_missing_cells'] / ticker_report['total_possible_cells'] 
            if ticker_report['total_possible_cells'] > 0 else 0
        )
        
        for column in columns_to_analyze:
            if column in ticker_data.columns:
                missing_count = ticker_data[column].isna().sum()
                ticker_report[f'{column}_missing_count'] = missing_count
                ticker_report[f'{column}_missing_rate'] = missing_count / len(ticker_data) if len(ticker_data) > 0 else 0
                
                ticker_report[f'{column}_has_consecutive'] = check_consecutive_missing(
                    ticker_data[column], DEFAULT_CONSECUTIVE_MISSING_THRESHOLD
                )
        
        report_data.append(ticker_report)
    
    report_df = pd.DataFrame(report_data)
    
    if save_path:
        report_df.to_csv(save_path, index=False)
        if verbose:
            print(f"Missing data report saved to: {save_path}")
    
    if verbose:
        print(f"Report created for {len(report_df)} tickers")
        print(f"Columns in report: {len(report_df.columns)}")
        
        high_missing_tickers = len(report_df[report_df['overall_missing_rate'] > 0.5])
        print(f"Tickers with >50% missing: {high_missing_tickers}")
    
    return report_df


def validate_missing_data_acceptable(df: pd.DataFrame,
                                   max_missing_rate: float = 0.2,
                                   columns: Optional[List[str]] = None,
                                   ticker_column: str = DEFAULT_TICKER_COLUMN,
                                   verbose: bool = True) -> Dict:
    
    if columns is None:
        columns = [col for col in df.columns if col != ticker_column]
    
    validation_result = {
        'acceptable': True,
        'max_allowed_rate': max_missing_rate,
        'overall_missing_rate': 0,
        'problematic_columns': [],
        'problematic_tickers': [],
        'recommendations': []
    }
    
    total_cells = len(df) * len(columns)
    total_missing = df[columns].isna().sum().sum()
    overall_rate = total_missing / total_cells if total_cells > 0 else 0
    
    validation_result['overall_missing_rate'] = overall_rate
    
    if overall_rate > max_missing_rate:
        validation_result['acceptable'] = False
        validation_result['recommendations'].append(
            f"Overall missing rate ({overall_rate:.1%}) exceeds threshold ({max_missing_rate:.1%})"
        )
    
    for column in columns:
        if column in df.columns:
            col_missing_rate = df[column].isna().sum() / len(df)
            if col_missing_rate > max_missing_rate:
                validation_result['problematic_columns'].append({
                    'column': column,
                    'missing_rate': col_missing_rate
                })
                validation_result['acceptable'] = False
    
    problematic_tickers = identify_problematic_tickers(
        df, columns, max_missing_rate, ticker_column, verbose=False
    )
    
    if problematic_tickers:
        validation_result['problematic_tickers'] = problematic_tickers
        validation_result['acceptable'] = False
        validation_result['recommendations'].append(
            f"Remove {len(problematic_tickers)} tickers with >{max_missing_rate:.0%} missing data"
        )
    
    if validation_result['problematic_columns']:
        validation_result['recommendations'].append(
            f"Address missing data in {len(validation_result['problematic_columns'])} columns"
        )
    
    if verbose:
        print_subsection_header("Missing Data Validation")
        if validation_result['acceptable']:
            print(f"✓ Missing data is acceptable (overall rate: {overall_rate:.1%})")
        else:
            print(f"✗ Missing data exceeds acceptable levels")
            print(f"Overall missing rate: {overall_rate:.1%} (max allowed: {max_missing_rate:.1%})")
            
            if validation_result['problematic_columns']:
                print(f"Problematic columns: {len(validation_result['problematic_columns'])}")
            
            if validation_result['problematic_tickers']:
                print(f"Problematic tickers: {len(validation_result['problematic_tickers'])}")
            
            print("Recommendations:")
            for rec in validation_result['recommendations']:
                print(f"  - {rec}")
    
    return validation_result
