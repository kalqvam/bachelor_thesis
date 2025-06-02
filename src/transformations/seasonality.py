import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from typing import Union, List, Dict, Tuple, Optional
import warnings

from ..utils import (
    DEFAULT_TICKER_COLUMN, SEASONALITY_MAX_LAGS, SEASONALITY_ALPHA, 
    SEASONALITY_MIN_DATA_POINTS, print_section_header, print_subsection_header,
    format_number, print_processing_stats
)

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100


def prepare_seasonality_data(df: pd.DataFrame, 
                            date_column: str = 'quarter',
                            verbose: bool = True) -> Tuple[pd.DataFrame, Dict]:
    if verbose:
        print_subsection_header("Preparing Data for Seasonality Analysis")
    
    df_result = df.copy()
    
    if 'quarter' in df_result.columns and 'date_quarter' not in df.columns:
        df_result['date_quarter'] = pd.to_datetime(df_result['quarter'])
        temp_column_created = True
        if verbose:
            print("Created temporary 'date_quarter' column from 'quarter'")
    else:
        df_result['date_quarter'] = pd.to_datetime(df_result[date_column])
        temp_column_created = False
        if verbose:
            print(f"Using existing '{date_column}' column")
    
    df_result = df_result.sort_values([DEFAULT_TICKER_COLUMN, 'date_quarter'])
    
    stats = {
        'original_shape': df.shape,
        'temp_column_created': temp_column_created,
        'date_column_used': date_column,
        'total_tickers': df_result[DEFAULT_TICKER_COLUMN].nunique()
    }
    
    return df_result, stats


def detect_seasonality(series: pd.Series, 
                      ticker: str, 
                      column: str,
                      max_lags: int = SEASONALITY_MAX_LAGS,
                      alpha: float = SEASONALITY_ALPHA,
                      verbose: bool = False) -> Tuple[bool, Optional[int], Dict]:
    
    series_length = len(series)
    series_variance = np.var(series.dropna())
    variance_threshold = 1e-7
    
    result_stats = {
        'series_length': series_length,
        'series_variance': series_variance,
        'method_used': None,
        'acf_spikes': [],
        'detection_details': {}
    }
    
    if series_length < max_lags + 5:
        if verbose:
            print(f"[{ticker}] {column}: Insufficient data points ({series_length}), skipping seasonality detection")
        result_stats['method_used'] = 'insufficient_data'
        return False, None, result_stats
    
    if series_variance < variance_threshold:
        if verbose:
            print(f"[{ticker}] {column}: Series is effectively constant (variance={series_variance:.10f}), skipping seasonality detection")
        result_stats['method_used'] = 'constant_series'
        return False, None, result_stats
    
    if verbose:
        print(f"\n=== Seasonality Detection for {ticker} ({column}) ===")
        print(f"Time series length: {series_length}")
        print(f"Series variance: {series_variance:.10f}")
    
    try:
        adf_result = adfuller(series.dropna())
        is_stationary = adf_result[1] < alpha
        
        result_stats['adf_statistic'] = adf_result[0]
        result_stats['adf_pvalue'] = adf_result[1]
        result_stats['is_stationary'] = is_stationary
        
        if verbose:
            print(f"ADF Test Results:")
            print(f"  ADF Statistic: {adf_result[0]:.4f}")
            print(f"  p-value: {adf_result[1]:.4f}")
            print(f"  Is stationary: {is_stationary}")
    except Exception as e:
        if verbose:
            print(f"ADF test failed: {str(e)}")
        result_stats['adf_error'] = str(e)
    
    from statsmodels.tsa.stattools import acf
    acf_values = acf(series.dropna(), nlags=max_lags, fft=True)
    
    if verbose:
        print(f"ACF Values:")
        for i, val in enumerate(acf_values):
            if i == 0:
                continue
            print(f"  Lag {i}: ACF = {val:.4f}")
    
    acf_values_for_spikes = acf_values[1:]
    
    acf_spikes = []
    for i in range(1, len(acf_values_for_spikes)-1):
        lag = i + 1
        if acf_values_for_spikes[i] > acf_values_for_spikes[i-1] and acf_values_for_spikes[i] > acf_values_for_spikes[i+1]:
            acf_spikes.append(lag)
    
    result_stats['acf_spikes'] = acf_spikes
    
    if verbose:
        print(f"ACF spikes detected at lags: {acf_spikes}")
    
    if len(acf_spikes) >= 2:
        potential_period = acf_spikes[0]
        if verbose:
            print(f"First ACF spike at lag {potential_period}, considering as potential period")
        
        multiple_spikes = [lag for lag in acf_spikes if lag != potential_period and lag % potential_period == 0]
        
        if multiple_spikes:
            if verbose:
                print(f"Confirmed seasonality: Found spikes at multiples of {potential_period}: {multiple_spikes}")
            
            significant_lags = []
            
            for lag in [potential_period] + multiple_spikes:
                try:
                    lb_result = acorr_ljungbox(series.dropna(), lags=[lag])
                    
                    if isinstance(lb_result, tuple):
                        p_value = lb_result[1][0]
                    else:
                        p_value = lb_result['lb_pvalue'].values[0]
                    
                    if verbose:
                        print(f"  Testing period {potential_period} at lag {lag}: Ljung-Box p-value = {p_value:.4f}")
                    
                    if p_value < alpha:
                        significant_lags.append(lag)
                except Exception as e:
                    if verbose:
                        print(f"  Ljung-Box test failed for lag {lag}: {str(e)}")
            
            if not significant_lags:
                if verbose:
                    print(f"  WARNING: Period {potential_period} has ACF spikes but no statistically significant lags. Will still apply STL decomposition.")
            
            final_period = potential_period
            if potential_period > 4:
                final_period = potential_period // 2
                if verbose:
                    print(f"First spike is at lag > 4, adjusting period from {potential_period} to {final_period}")
            
            result_stats['method_used'] = 'acf_spikes'
            result_stats['detection_details'] = {
                'potential_period': potential_period,
                'final_period': final_period,
                'multiple_spikes': multiple_spikes,
                'significant_lags': significant_lags
            }
            
            if verbose:
                print(f"Detected seasonality with period {final_period} for {ticker}")
            return True, final_period, result_stats
    
    if len(acf_values) >= 7:
        if verbose:
            print("\nPrimary spike detection method didn't find seasonality.")
            print("Checking for alternating pattern (period = 2) as fallback:")
        
        alternating_pattern_detected = False
        alternating_count = 0
        min_pairs_to_check = min(6, (len(acf_values) - 1) // 2)
        
        pct_changes = []
        for i in range(1, len(acf_values)-1):
            change = abs((acf_values[i+1] - acf_values[i]) / acf_values[i]) * 100
            pct_changes.append(change)
            if verbose:
                print(f"  % change from lag {i} to {i+1}: {change:.2f}%")
        
        alternating_ratios = []
        for i in range(0, len(pct_changes)-1, 2):
            if pct_changes[i+1] > 0:
                ratio = pct_changes[i] / pct_changes[i+1]
                alternating_ratios.append(ratio)
                if verbose:
                    print(f"  Ratio of changes: ({i+1}->{i+2})/({i+2}->{i+3}) = {ratio:.2f}")
                
                if ratio <= 0.51:
                    alternating_count += 1
        
        if alternating_count >= min_pairs_to_check // 2:
            if verbose:
                print(f"  Alternating pattern detected in {alternating_count}/{len(alternating_ratios)} pairs")
                print(f"  This suggests a period = 2 seasonality")
                print(f"Detected seasonality with period 2 for {ticker} (through alternating pattern)")
            
            result_stats['method_used'] = 'alternating_pattern'
            result_stats['detection_details'] = {
                'alternating_count': alternating_count,
                'total_pairs': len(alternating_ratios),
                'alternating_ratios': alternating_ratios
            }
            
            return True, 2, result_stats
        else:
            if verbose:
                print(f"  No strong alternating pattern detected ({alternating_count}/{len(alternating_ratios)} pairs)")
    
    if verbose:
        print(f"No seasonality detected for {ticker} in {column}")
    
    result_stats['method_used'] = 'no_seasonality'
    return False, None, result_stats


def remove_seasonality(series: pd.Series, 
                      period: int, 
                      ticker: str = "",
                      verbose: bool = False) -> Tuple[pd.Series, Dict]:
    
    stats = {
        'period_used': period,
        'original_length': len(series),
        'success': False,
        'seasonal_strength': np.nan,
        'method_used': None
    }
    
    if len(series) < period * 2:
        if verbose:
            print(f"WARNING: Time series too short ({len(series)}) for STL with period {period}, skipping")
        stats['method_used'] = 'too_short'
        return series, stats
    
    seasonal_smoothing = period if period % 2 == 1 else period + 1
    
    if verbose:
        print(f"Applying STL decomposition with period={period}, seasonal_smoothing={seasonal_smoothing}")
    
    try:
        stl = STL(series,
                period=period,
                seasonal=seasonal_smoothing,
                trend=None,
                robust=True)
        result = stl.fit()
        
        trend = result.trend
        seasonal = result.seasonal
        residual = result.resid
        
        seasonal_strength = (np.var(seasonal) / (np.var(series - trend))) * 100
        
        if verbose:
            print(f"STL decomposition components:")
            print(f"  Trend range: {trend.min():.2f} to {trend.max():.2f}")
            print(f"  Seasonal range: {seasonal.min():.2f} to {seasonal.max():.2f}")
            print(f"  Residual range: {residual.min():.2f} to {residual.max():.2f}")
            print(f"  Seasonal strength: {seasonal_strength:.2f}%")
        
        deseasonalized = trend + residual
        
        stats.update({
            'success': True,
            'seasonal_strength': seasonal_strength,
            'trend_range': (trend.min(), trend.max()),
            'seasonal_range': (seasonal.min(), seasonal.max()),
            'residual_range': (residual.min(), residual.max()),
            'method_used': 'stl_decomposition'
        })
        
        return deseasonalized, stats
    
    except Exception as e:
        if verbose:
            print(f"ERROR in STL decomposition: {str(e)}")
            print("Falling back to original series")
        
        stats.update({
            'method_used': 'error_fallback',
            'error': str(e)
        })
        
        return series, stats


def plot_stl_decomposition(series: pd.Series, 
                          period: int, 
                          ticker: str, 
                          column: str) -> bool:
    if len(series) < period * 2:
        print(f"Time series too short for STL decomposition visualization")
        return False
    
    seasonal_smoothing = period if period % 2 == 1 else period + 1
    
    try:
        stl = STL(series,
                period=period,
                seasonal=seasonal_smoothing,
                trend=None,
                robust=True)
        result = stl.fit()
        
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        
        axes[0].plot(series.index, series.values)
        axes[0].set_title(f'Original {column} for {ticker}')
        
        axes[1].plot(series.index, result.trend)
        axes[1].set_title('Trend Component')
        
        axes[2].plot(series.index, result.seasonal)
        axes[2].set_title(f'Seasonal Component (period={period}, smoothing={seasonal_smoothing})')
        
        axes[3].plot(series.index, result.resid)
        axes[3].set_title('Residual Component')
        
        plt.tight_layout()
        plt.show()
        
        return True
    
    except Exception as e:
        print(f"Error in STL decomposition visualization: {str(e)}")
        return False


def process_company_seasonality(company_df: pd.DataFrame, 
                               column: str, 
                               ticker: str,
                               max_lags: int = SEASONALITY_MAX_LAGS,
                               alpha: float = SEASONALITY_ALPHA,
                               min_data_points: int = SEASONALITY_MIN_DATA_POINTS,
                               plot: bool = False,
                               verbose: bool = False) -> Tuple[pd.DataFrame, Dict]:
    
    company_df = company_df.sort_values('date_quarter')
    
    stats = {
        'ticker': ticker,
        'column': column,
        'data_points': len(company_df),
        'seasonality_detected': False,
        'period': None,
        'processing_status': 'success'
    }
    
    if verbose:
        print(f"\nProcessing {ticker} for {column}:")
        print(f"Number of data points: {len(company_df)}")
    
    series = company_df.set_index('date_quarter')[column].astype(float)
    
    missing_count = series.isna().sum()
    missing_pct = missing_count / len(series) * 100
    stats['missing_count'] = missing_count
    stats['missing_percentage'] = missing_pct
    
    if verbose:
        print(f"Missing values: {missing_count} ({missing_pct:.1f}%)")
    
    if missing_pct > 20:
        if verbose:
            print(f"Too many missing values for {ticker} {column}, skipping seasonality detection")
        stats['processing_status'] = 'too_many_missing'
        company_df[f'{column}_deseasonalized'] = company_df[column]
        return company_df, stats
    
    if missing_count > 0:
        series = series.interpolate(method='linear')
        if verbose:
            print(f"Filled {missing_count} missing values with linear interpolation")
    
    min_val = series.min()
    max_val = series.max()
    
    if (abs(min_val - 0) < 1e-6 and abs(max_val - 0) < 1e-6) or (abs(min_val - 1) < 1e-6 and abs(max_val - 1) < 1e-6):
        if verbose:
            print(f"RESULT: Series is constant at {min_val} (likely min-max normalized extreme), skipping seasonality detection")
        stats['processing_status'] = 'constant_series'
        company_df[f'{column}_deseasonalized'] = company_df[column]
        return company_df, stats
    
    is_seasonal, period, detection_stats = detect_seasonality(
        series, ticker, column, max_lags, alpha, verbose
    )
    
    stats.update(detection_stats)
    stats['seasonality_detected'] = is_seasonal
    stats['period'] = period
    
    if is_seasonal and period is not None:
        if verbose:
            print(f"RESULT: Detected seasonality with period {period} for {ticker} in {column}")
        
        if plot:
            plt.figure(figsize=(15, 10))
            
            plt.subplot(3, 1, 1)
            plt.plot(series.index, series.values)
            plt.title(f"Original {column} for {ticker}")
            
            plt.subplot(3, 1, 2)
            plot_acf(series.dropna(), lags=min(20, len(series)//2), ax=plt.gca())
            plt.title(f"ACF for {column}")
            
            plt.subplot(3, 1, 3)
            plot_pacf(series.dropna(), lags=min(20, len(series)//2), ax=plt.gca())
            plt.title(f"PACF for {column}")
            
            plt.tight_layout()
            plt.show()
            
            plot_stl_decomposition(series, period, ticker, column)
        
        deseasonalized, deseason_stats = remove_seasonality(series, period, ticker, verbose)
        stats['deseasonalization'] = deseason_stats
        
        company_df[f'{column}_deseasonalized'] = deseasonalized.values
        
        if deseason_stats['success']:
            seasonal_component = series.values - deseasonalized.values
            seasonal_magnitude = np.abs(seasonal_component).mean()
            seasonal_pct = (seasonal_magnitude / np.abs(series.values).mean()) * 100
            
            stats['seasonal_magnitude'] = seasonal_magnitude
            stats['seasonal_percentage'] = seasonal_pct
            
            if verbose:
                print(f"Seasonal adjustment magnitude: {seasonal_magnitude:.2f} ({seasonal_pct:.1f}% of mean)")
        
        if plot:
            plt.figure(figsize=(12, 6))
            plt.plot(company_df['date_quarter'], company_df[column], label='Original')
            plt.plot(company_df['date_quarter'], company_df[f'{column}_deseasonalized'], label='Deseasonalized')
            plt.title(f"Deseasonalized {column} for {ticker}")
            plt.legend()
            plt.show()
    else:
        if verbose:
            print(f"RESULT: No seasonality detected for {ticker} in {column}")
        company_df[f'{column}_deseasonalized'] = company_df[column]
    
    if verbose:
        print("-" * 50)
    
    return company_df, stats


def process_all_companies_seasonality(df: pd.DataFrame, 
                                     column: str,
                                     sample_size: Optional[int] = None,
                                     max_lags: int = SEASONALITY_MAX_LAGS,
                                     alpha: float = SEASONALITY_ALPHA,
                                     min_data_points: int = SEASONALITY_MIN_DATA_POINTS,
                                     verbose: bool = True) -> Tuple[pd.DataFrame, Dict]:
    
    if verbose:
        print_section_header(f"Seasonality Processing for {column}")
    
    unique_tickers = df[DEFAULT_TICKER_COLUMN].unique()
    
    if verbose:
        print(f"Total unique tickers in dataset: {format_number(len(unique_tickers))}")
    
    if sample_size:
        unique_tickers = unique_tickers[:sample_size]
        if verbose:
            print(f"Processing sample of {format_number(sample_size)} tickers")
    
    processed_dfs = []
    company_stats = []
    seasonality_summary = {
        'total_processed': 0,
        'seasonal_detected': 0,
        'periods_detected': [],
        'processing_errors': 0,
        'constant_series': 0,
        'too_many_missing': 0
    }
    
    for i, ticker in enumerate(unique_tickers):
        if verbose and (i+1) % 10 == 0:
            print(f"Progress: {format_number(i+1)}/{format_number(len(unique_tickers))} companies processed ({(i+1)/len(unique_tickers)*100:.1f}%)")
        
        company_df = df[df[DEFAULT_TICKER_COLUMN] == ticker].copy()
        
        if len(company_df) < min_data_points:
            if verbose:
                print(f"WARNING: {ticker} has insufficient data points ({len(company_df)}), skipping seasonality detection")
            processed_dfs.append(company_df)
            continue
        
        company_df, stats = process_company_seasonality(
            company_df, column, ticker, max_lags, alpha, min_data_points, plot=False, verbose=False
        )
        
        processed_dfs.append(company_df)
        company_stats.append(stats)
        
        seasonality_summary['total_processed'] += 1
        
        if stats['seasonality_detected']:
            seasonality_summary['seasonal_detected'] += 1
            if stats['period'] and stats['period'] not in seasonality_summary['periods_detected']:
                seasonality_summary['periods_detected'].append(stats['period'])
        
        if stats['processing_status'] == 'constant_series':
            seasonality_summary['constant_series'] += 1
        elif stats['processing_status'] == 'too_many_missing':
            seasonality_summary['too_many_missing'] += 1
    
    processed_df = pd.concat(processed_dfs, ignore_index=True)
    
    seasonal_count = seasonality_summary['seasonal_detected']
    total_count = seasonality_summary['total_processed']
    seasonal_pct = (seasonal_count / total_count * 100) if total_count > 0 else 0
    periods = seasonality_summary['periods_detected']
    
    summary_stats = {
        'column': column,
        'total_tickers_processed': total_count,
        'seasonal_detected': seasonal_count,
        'seasonal_percentage': seasonal_pct,
        'periods_detected': periods,
        'company_stats': company_stats,
        'processing_summary': seasonality_summary
    }
    
    if verbose:
        print_subsection_header("Seasonality Processing Summary")
        print(f"{column.upper()}:")
        print(f"  Companies with seasonality: {format_number(seasonal_count)}/{format_number(total_count)} ({seasonal_pct:.1f}%)")
        print(f"  Detected seasonal periods: {periods}")
        print(f"  Constant series: {format_number(seasonality_summary['constant_series'])}")
        print(f"  Too many missing: {format_number(seasonality_summary['too_many_missing'])}")
    
    return processed_df, summary_stats


def apply_seasonality_processing(df: pd.DataFrame,
                                columns: Union[str, List[str]],
                                sample_size: Optional[int] = None,
                                max_lags: int = SEASONALITY_MAX_LAGS,
                                alpha: float = SEASONALITY_ALPHA,
                                min_data_points: int = SEASONALITY_MIN_DATA_POINTS,
                                save_file: bool = True,
                                output_filename: str = 'processed_data_deseasonalized.csv',
                                replace_original: bool = True,
                                verbose: bool = True) -> Tuple[pd.DataFrame, Dict]:
    
    if verbose:
        print_section_header("Seasonality Processing Pipeline")
    
    if isinstance(columns, str):
        columns = [columns]
    
    df_prepared, prep_stats = prepare_seasonality_data(df, verbose=verbose)
    processed_df = df_prepared.copy()
    
    all_stats = {
        'preparation': prep_stats,
        'column_results': {},
        'overall_summary': {}
    }
    
    for column in columns:
        if verbose:
            print(f"\nProcessing column: {column}")
        
        current_df, column_stats = process_all_companies_seasonality(
            processed_df, column, sample_size, max_lags, alpha, min_data_points, verbose
        )
        
        processed_df = current_df.copy()
        all_stats['column_results'][column] = column_stats
        
        if replace_original:
            deseasonalized_col = f'{column}_deseasonalized'
            if deseasonalized_col in processed_df.columns:
                if verbose:
                    print(f"Replacing original {column} with deseasonalized values")
                processed_df[column] = processed_df[deseasonalized_col]
                processed_df.drop(columns=[deseasonalized_col], inplace=True)
                if verbose:
                    print(f"Dropped intermediate column {deseasonalized_col}")
    
    if prep_stats['temp_column_created'] and 'date_quarter' in processed_df.columns:
        if verbose:
            print("Dropping temporary 'date_quarter' column")
        processed_df.drop(columns=['date_quarter'], inplace=True)
    
    total_seasonal = sum(stats['seasonal_detected'] for stats in all_stats['column_results'].values())
    total_processed = sum(stats['total_tickers_processed'] for stats in all_stats['column_results'].values())
    
    all_stats['overall_summary'] = {
        'columns_processed': len(columns),
        'total_company_column_combinations': total_processed,
        'total_seasonal_detected': total_seasonal,
        'final_shape': processed_df.shape,
        'sample_size_used': sample_size
    }
    
    if save_file:
        processed_df.to_csv(output_filename, index=False)
        if verbose:
            print(f"Saved processed data to {output_filename}")
    
    if verbose:
        print_subsection_header("Overall Processing Summary")
        print(f"Columns processed: {len(columns)}")
        print(f"Total seasonality instances detected: {format_number(total_seasonal)}")
        print(f"Final dataset shape: {processed_df.shape}")
    
    return processed_df, all_stats
