import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Callable

from ...utils import (
    DEFAULT_TICKER_COLUMN, print_subsection_header, 
    format_number, print_processing_stats
)


def filter_by_ticker_list(df: pd.DataFrame,
                         ticker_list: List[str],
                         include: bool = True,
                         ticker_column: str = DEFAULT_TICKER_COLUMN,
                         verbose: bool = True) -> Tuple[pd.DataFrame, Dict]:
    
    if verbose:
        action = "Including" if include else "Excluding"
        print_subsection_header(f"{action} Specific Tickers")
    
    original_shape = df.shape
    original_tickers = df[ticker_column].nunique()
    
    if include:
        df_filtered = df[df[ticker_column].isin(ticker_list)].copy()
        action_performed = "included"
    else:
        df_filtered = df[~df[ticker_column].isin(ticker_list)].copy()
        action_performed = "excluded"
    
    final_tickers = df_filtered[ticker_column].nunique()
    rows_removed = original_shape[0] - df_filtered.shape[0]
    tickers_affected = abs(original_tickers - final_tickers)
    
    stats = {
        'original_shape': original_shape,
        'final_shape': df_filtered.shape,
        'rows_removed': rows_removed,
        'original_tickers': original_tickers,
        'final_tickers': final_tickers,
        'tickers_affected': tickers_affected,
        'ticker_list_size': len(ticker_list),
        'action_performed': action_performed,
        'retention_rate': (df_filtered.shape[0] / original_shape[0]) * 100 if original_shape[0] > 0 else 0
    }
    
    if verbose:
        print(f"Ticker filtering results:")
        print(f"  Action: {action_performed} {len(ticker_list)} tickers")
        print(f"  Original tickers: {original_tickers}")
        print(f"  Final tickers: {final_tickers}")
        print(f"  Rows removed: {format_number(rows_removed)}")
        print(f"  Retention rate: {stats['retention_rate']:.1f}%")
    
    return df_filtered, stats


def filter_by_value_range(df: pd.DataFrame,
                         column: str,
                         min_value: Optional[float] = None,
                         max_value: Optional[float] = None,
                         inclusive: bool = True,
                         verbose: bool = True) -> Tuple[pd.DataFrame, Dict]:
    
    if verbose:
        print_subsection_header(f"Filtering by Value Range: {column}")
    
    if column not in df.columns:
        if verbose:
            print(f"Column '{column}' not found")
        return df.copy(), {'status': 'column_not_found'}
    
    original_shape = df.shape
    
    if min_value is None and max_value is None:
        if verbose:
            print("No range specified, returning original data")
        return df.copy(), {'status': 'no_range_specified'}
    
    mask = pd.Series([True] * len(df), index=df.index)
    
    if min_value is not None:
        if inclusive:
            mask = mask & (df[column] >= min_value)
        else:
            mask = mask & (df[column] > min_value)
    
    if max_value is not None:
        if inclusive:
            mask = mask & (df[column] <= max_value)
        else:
            mask = mask & (df[column] < max_value)
    
    df_filtered = df[mask].copy()
    
    rows_removed = original_shape[0] - df_filtered.shape[0]
    
    stats = {
        'original_shape': original_shape,
        'final_shape': df_filtered.shape,
        'rows_removed': rows_removed,
        'column': column,
        'min_value': min_value,
        'max_value': max_value,
        'inclusive': inclusive,
        'retention_rate': (df_filtered.shape[0] / original_shape[0]) * 100 if original_shape[0] > 0 else 0
    }
    
    if verbose:
        range_str = f"{min_value if min_value is not None else '-∞'} to {max_value if max_value is not None else '+∞'}"
        print(f"Value range filtering results:")
        print(f"  Column: {column}")
        print(f"  Range: {range_str} ({'inclusive' if inclusive else 'exclusive'})")
        print(f"  Rows removed: {format_number(rows_removed)}")
        print(f"  Retention rate: {stats['retention_rate']:.1f}%")
    
    return df_filtered, stats


def filter_by_custom_condition(df: pd.DataFrame,
                              condition: Union[str, Callable],
                              condition_name: str = "Custom condition",
                              verbose: bool = True) -> Tuple[pd.DataFrame, Dict]:
    
    if verbose:
        print_subsection_header(f"Applying Custom Filter: {condition_name}")
    
    original_shape = df.shape
    
    try:
        if isinstance(condition, str):
            mask = df.eval(condition)
        elif callable(condition):
            mask = condition(df)
        else:
            raise ValueError("Condition must be a string expression or callable function")
        
        df_filtered = df[mask].copy()
        
        rows_removed = original_shape[0] - df_filtered.shape[0]
        
        stats = {
            'original_shape': original_shape,
            'final_shape': df_filtered.shape,
            'rows_removed': rows_removed,
            'condition_name': condition_name,
            'condition_type': 'string' if isinstance(condition, str) else 'function',
            'retention_rate': (df_filtered.shape[0] / original_shape[0]) * 100 if original_shape[0] > 0 else 0,
            'success': True
        }
        
        if verbose:
            print(f"Custom filtering results:")
            print(f"  Condition: {condition_name}")
            print(f"  Type: {stats['condition_type']}")
            print(f"  Rows removed: {format_number(rows_removed)}")
            print(f"  Retention rate: {stats['retention_rate']:.1f}%")
    
    except Exception as e:
        if verbose:
            print(f"Error applying custom condition: {str(e)}")
        
        stats = {
            'original_shape': original_shape,
            'final_shape': original_shape,
            'rows_removed': 0,
            'condition_name': condition_name,
            'error': str(e),
            'success': False
        }
        df_filtered = df.copy()
    
    return df_filtered, stats


def remove_outliers_by_iqr(df: pd.DataFrame,
                          columns: Union[str, List[str]],
                          iqr_multiplier: float = 1.5,
                          ticker_column: str = DEFAULT_TICKER_COLUMN,
                          per_ticker: bool = True,
                          verbose: bool = True) -> Tuple[pd.DataFrame, Dict]:
    
    if verbose:
        scope = "per ticker" if per_ticker else "overall"
        print_subsection_header(f"Removing IQR Outliers ({scope})")
    
    if isinstance(columns, str):
        columns = [columns]
    
    original_shape = df.shape
    outlier_rows = set()
    column_stats = {}
    
    for column in columns:
        if column not in df.columns:
            if verbose:
                print(f"Column '{column}' not found, skipping")
            continue
        
        if per_ticker:
            for ticker in df[ticker_column].unique():
                ticker_data = df[df[ticker_column] == ticker]
                ticker_outliers = _identify_iqr_outliers(
                    ticker_data, column, iqr_multiplier
                )
                outlier_rows.update(ticker_outliers)
        else:
            global_outliers = _identify_iqr_outliers(df, column, iqr_multiplier)
            outlier_rows.update(global_outliers)
        
        total_outliers = len([idx for idx in outlier_rows if df.loc[idx, column] == df.loc[idx, column]])
        column_stats[column] = {
            'outliers_identified': total_outliers,
            'outlier_rate': (total_outliers / len(df)) * 100 if len(df) > 0 else 0
        }
    
    df_filtered = df.drop(index=outlier_rows).copy()
    
    rows_removed = len(outlier_rows)
    
    stats = {
        'original_shape': original_shape,
        'final_shape': df_filtered.shape,
        'rows_removed': rows_removed,
        'columns_processed': columns,
        'iqr_multiplier': iqr_multiplier,
        'per_ticker': per_ticker,
        'column_stats': column_stats,
        'retention_rate': (df_filtered.shape[0] / original_shape[0]) * 100 if original_shape[0] > 0 else 0
    }
    
    if verbose:
        print(f"IQR outlier removal results:")
        print(f"  Columns processed: {len(columns)}")
        print(f"  IQR multiplier: {iqr_multiplier}")
        print(f"  Scope: {'per ticker' if per_ticker else 'overall'}")
        print(f"  Rows removed: {format_number(rows_removed)}")
        print(f"  Retention rate: {stats['retention_rate']:.1f}%")
        
        for col, col_stats in column_stats.items():
            print(f"    {col}: {col_stats['outliers_identified']} outliers ({col_stats['outlier_rate']:.1f}%)")
    
    return df_filtered, stats


def _identify_iqr_outliers(df: pd.DataFrame, column: str, iqr_multiplier: float) -> List[int]:
    data = df[column].dropna()
    
    if len(data) < 4:
        return []
    
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - iqr_multiplier * IQR
    upper_bound = Q3 + iqr_multiplier * IQR
    
    outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
    outlier_indices = df[outlier_mask].index.tolist()
    
    return outlier_indices


def filter_by_percentile_range(df: pd.DataFrame,
                              column: str,
                              lower_percentile: float = 5,
                              upper_percentile: float = 95,
                              ticker_column: str = DEFAULT_TICKER_COLUMN,
                              per_ticker: bool = True,
                              verbose: bool = True) -> Tuple[pd.DataFrame, Dict]:
    
    if verbose:
        scope = "per ticker" if per_ticker else "overall"
        print_subsection_header(f"Filtering by Percentile Range ({scope})")
    
    if column not in df.columns:
        if verbose:
            print(f"Column '{column}' not found")
        return df.copy(), {'status': 'column_not_found'}
    
    original_shape = df.shape
    rows_to_keep = set(df.index)
    
    if per_ticker:
        for ticker in df[ticker_column].unique():
            ticker_data = df[df[ticker_column] == ticker]
            
            if len(ticker_data) < 10:
                continue
            
            lower_bound = ticker_data[column].quantile(lower_percentile / 100)
            upper_bound = ticker_data[column].quantile(upper_percentile / 100)
            
            ticker_outliers = ticker_data[
                (ticker_data[column] < lower_bound) | 
                (ticker_data[column] > upper_bound)
            ].index
            
            rows_to_keep = rows_to_keep - set(ticker_outliers)
    else:
        lower_bound = df[column].quantile(lower_percentile / 100)
        upper_bound = df[column].quantile(upper_percentile / 100)
        
        outliers = df[
            (df[column] < lower_bound) | 
            (df[column] > upper_bound)
        ].index
        
        rows_to_keep = rows_to_keep - set(outliers)
    
    df_filtered = df.loc[list(rows_to_keep)].copy()
    
    rows_removed = original_shape[0] - df_filtered.shape[0]
    
    stats = {
        'original_shape': original_shape,
        'final_shape': df_filtered.shape,
        'rows_removed': rows_removed,
        'column': column,
        'lower_percentile': lower_percentile,
        'upper_percentile': upper_percentile,
        'per_ticker': per_ticker,
        'retention_rate': (df_filtered.shape[0] / original_shape[0]) * 100 if original_shape[0] > 0 else 0
    }
    
    if verbose:
        print(f"Percentile filtering results:")
        print(f"  Column: {column}")
        print(f"  Range: {lower_percentile}th - {upper_percentile}th percentile")
        print(f"  Scope: {'per ticker' if per_ticker else 'overall'}")
        print(f"  Rows removed: {format_number(rows_removed)}")
        print(f"  Retention rate: {stats['retention_rate']:.1f}%")
    
    return df_filtered, stats


def apply_multiple_filters(df: pd.DataFrame,
                          filters: List[Dict],
                          verbose: bool = True) -> Tuple[pd.DataFrame, Dict]:
    
    if verbose:
        print_subsection_header(f"Applying {len(filters)} Sequential Filters")
    
    original_shape = df.shape
    df_current = df.copy()
    filter_results = []
    
    for i, filter_config in enumerate(filters):
        filter_type = filter_config.get('type')
        filter_name = filter_config.get('name', f"Filter {i+1}")
        
        if verbose:
            print(f"\nApplying {filter_name} ({filter_type})...")
        
        if filter_type == 'ticker_list':
            df_current, filter_stats = filter_by_ticker_list(
                df_current, 
                filter_config['ticker_list'],
                filter_config.get('include', True),
                filter_config.get('ticker_column', DEFAULT_TICKER_COLUMN),
                verbose=False
            )
        
        elif filter_type == 'value_range':
            df_current, filter_stats = filter_by_value_range(
                df_current,
                filter_config['column'],
                filter_config.get('min_value'),
                filter_config.get('max_value'),
                filter_config.get('inclusive', True),
                verbose=False
            )
        
        elif filter_type == 'custom_condition':
            df_current, filter_stats = filter_by_custom_condition(
                df_current,
                filter_config['condition'],
                filter_config.get('condition_name', filter_name),
                verbose=False
            )
        
        elif filter_type == 'iqr_outliers':
            df_current, filter_stats = remove_outliers_by_iqr(
                df_current,
                filter_config['columns'],
                filter_config.get('iqr_multiplier', 1.5),
                filter_config.get('ticker_column', DEFAULT_TICKER_COLUMN),
                filter_config.get('per_ticker', True),
                verbose=False
            )
        
        elif filter_type == 'percentile_range':
            df_current, filter_stats = filter_by_percentile_range(
                df_current,
                filter_config['column'],
                filter_config.get('lower_percentile', 5),
                filter_config.get('upper_percentile', 95),
                filter_config.get('ticker_column', DEFAULT_TICKER_COLUMN),
                filter_config.get('per_ticker', True),
                verbose=False
            )
        
        else:
            if verbose:
                print(f"Unknown filter type: {filter_type}")
            continue
        
        filter_stats['filter_name'] = filter_name
        filter_stats['filter_type'] = filter_type
        filter_results.append(filter_stats)
        
        if verbose:
            retention = filter_stats.get('retention_rate', 0)
            rows_removed = filter_stats.get('rows_removed', 0)
            print(f"  {filter_name}: removed {format_number(rows_removed)} rows (retention: {retention:.1f}%)")
    
    total_rows_removed = original_shape[0] - df_current.shape[0]
    
    overall_stats = {
        'original_shape': original_shape,
        'final_shape': df_current.shape,
        'total_rows_removed': total_rows_removed,
        'filters_applied': len(filters),
        'filter_results': filter_results,
        'overall_retention_rate': (df_current.shape[0] / original_shape[0]) * 100 if original_shape[0] > 0 else 0
    }
    
    if verbose:
        print(f"\nOverall filtering results:")
        print(f"  Filters applied: {len(filters)}")
        print(f"  Total rows removed: {format_number(total_rows_removed)}")
        print(f"  Overall retention: {overall_stats['overall_retention_rate']:.1f}%")
        print(f"  Final shape: {df_current.shape}")
    
    return df_current, overall_stats


def get_filtering_recommendations(df: pd.DataFrame,
                                ticker_column: str = DEFAULT_TICKER_COLUMN,
                                numeric_columns: Optional[List[str]] = None,
                                verbose: bool = True) -> Dict:
    
    if verbose:
        print_subsection_header("Generating Filtering Recommendations")
    
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if ticker_column in numeric_columns:
            numeric_columns.remove(ticker_column)
    
    recommendations = {
        'outlier_recommendations': [],
        'missing_data_recommendations': [],
        'value_range_recommendations': [],
        'ticker_recommendations': []
    }
    
    for column in numeric_columns:
        if column in df.columns:
            data = df[column].dropna()
            if len(data) > 10:
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                
                outliers = data[(data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR)]
                outlier_rate = len(outliers) / len(data) * 100
                
                if outlier_rate > 5:
                    recommendations['outlier_recommendations'].append({
                        'column': column,
                        'outlier_rate': outlier_rate,
                        'recommendation': f"Consider removing IQR outliers in {column} ({outlier_rate:.1f}% of data)"
                    })
    
    for column in df.columns:
        if column != ticker_column:
            missing_rate = df[column].isna().sum() / len(df) * 100
            if missing_rate > 20:
                recommendations['missing_data_recommendations'].append({
                    'column': column,
                    'missing_rate': missing_rate,
                    'recommendation': f"High missing data in {column} ({missing_rate:.1f}%) - consider removal or imputation"
                })
    
    for column in numeric_columns:
        if column in df.columns:
            data = df[column].dropna()
            if len(data) > 0:
                min_val, max_val = data.min(), data.max()
                
                if min_val < 0 and column in ['revenue', 'assets', 'market_cap']:
                    recommendations['value_range_recommendations'].append({
                        'column': column,
                        'issue': 'negative_values',
                        'recommendation': f"Remove negative values in {column} (financial metrics should be positive)"
                    })
                
                if 'ratio' in column.lower() and (data == 0).sum() > len(data) * 0.1:
                    zero_rate = (data == 0).sum() / len(data) * 100
                    recommendations['value_range_recommendations'].append({
                        'column': column,
                        'issue': 'excessive_zeros',
                        'recommendation': f"High proportion of zeros in {column} ({zero_rate:.1f}%) - consider filtering"
                    })
    
    ticker_obs = df[ticker_column].value_counts()
    min_obs = ticker_obs.min()
    max_obs = ticker_obs.max()
    
    if max_obs > min_obs * 2:
        unbalanced_tickers = ticker_obs[ticker_obs < ticker_obs.median()].index.tolist()
        recommendations['ticker_recommendations'].append({
            'issue': 'unbalanced_panel',
            'affected_tickers': len(unbalanced_tickers),
            'recommendation': f"Panel is unbalanced - consider filtering to tickers with consistent observations"
        })
    
    if len(ticker_obs[ticker_obs < 8]) > 0:
        short_series_tickers = ticker_obs[ticker_obs < 8].index.tolist()
        recommendations['ticker_recommendations'].append({
            'issue': 'short_time_series',
            'affected_tickers': len(short_series_tickers),
            'recommendation': f"Remove {len(short_series_tickers)} tickers with <8 observations for robust analysis"
        })
    
    if verbose:
        total_recommendations = sum(len(recs) for recs in recommendations.values())
        print(f"Generated {total_recommendations} filtering recommendations:")
        
        for category, recs in recommendations.items():
            if recs:
                print(f"\n{category.replace('_', ' ').title()}:")
                for rec in recs:
                    print(f"  - {rec['recommendation']}")
    
    return recommendations


def create_filter_pipeline_config(recommendations: Dict) -> List[Dict]:
    pipeline = []
    
    for rec in recommendations.get('missing_data_recommendations', []):
        if rec['missing_rate'] > 50:
            pipeline.append({
                'type': 'custom_condition',
                'name': f"Remove high missing {rec['column']}",
                'condition': f"{rec['column']}.notna()"
            })
    
    for rec in recommendations.get('outlier_recommendations', []):
        if rec['outlier_rate'] > 10:
            pipeline.append({
                'type': 'iqr_outliers',
                'name': f"Remove {rec['column']} outliers",
                'columns': [rec['column']],
                'iqr_multiplier': 1.5,
                'per_ticker': True
            })
    
    for rec in recommendations.get('value_range_recommendations', []):
        if rec['issue'] == 'negative_values':
            pipeline.append({
                'type': 'value_range',
                'name': f"Remove negative {rec['column']}",
                'column': rec['column'],
                'min_value': 0
            })
    
    return pipeline
