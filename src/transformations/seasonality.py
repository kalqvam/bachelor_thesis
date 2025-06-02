import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from typing import List, Dict, Tuple, Optional, Union
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
                      ticker: str = "", 
                      column: str = "",
                      max_lags: int = SEASONALITY_MAX_LAGS,
                      alpha: float = SEASONALITY_ALPHA,
                      verbose: bool = False) -> Tuple[bool, Optional[int]]:
    
    if len(series) < max_lags + 5:
        return False, None

    variance = np.var(series.dropna())
    variance_threshold = 1e-7

    if variance < variance_threshold:
        return False, None

    try:
        adf_result = adfuller(series.dropna())
        is_stationary = adf_result[1] < alpha
    except:
        return False, None

    from statsmodels.tsa.stattools import acf
    try:
        acf_values = acf(series.dropna(), nlags=max_lags, fft=True)
    except:
        return False, None

    acf_values_for_spikes = acf_values[1:]

    acf_spikes = []
    for i in range(1, len(acf_values_for_spikes)-1):
        lag = i + 1
        if acf_values_for_spikes[i] > acf_values_for_spikes[i-1] and acf_values_for_spikes[i] > acf_values_for_spikes[i+1]:
            acf_spikes.append(lag)

    if len(acf_spikes) >= 2:
        potential_period = acf_spikes[0]
        multiple_spikes = [lag for lag in acf_spikes if lag != potential_period and lag % potential_period == 0]

        if multiple_spikes:
            final_period = potential_period
            if potential_period > 4:
                final_period = potential_period // 2
            return True, final_period

    if len(acf_values) >= 7:
        alternating_count = 0
        min_pairs_to_check = min(6, (len(acf_values) - 1) // 2)

        pct_changes = []
        for i in range(1, len(acf_values)-1):
            change = abs((acf_values[i+1] - acf_values[i]) / acf_values[i]) * 100
            pct_changes.append(change)

        alternating_ratios = []
        for i in range(0, len(pct_changes)-1, 2):
            if pct_changes[i+1] > 0:
                ratio = pct_changes[i] / pct_changes[i+1]
                alternating_ratios.append(ratio)
                if ratio <= 0.51:
                    alternating_count += 1

        if alternating_count >= min_pairs_to_check // 2:
            return True, 2

    return False, None


def remove_seasonality(series: pd.Series, period: int) -> pd.Series:
    if len(series) < period * 2:
        return series

    seasonal_smoothing = period if period % 2 == 1 else period + 1

    try:
        stl = STL(series, period=period, seasonal=seasonal_smoothing, trend=None, robust=True)
        result = stl.fit()
        deseasonalized = result.trend + result.resid
        return deseasonalized
    except Exception as e:
        return series


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
                               plot: bool = False,
                               verbose: bool = False) -> Tuple[pd.DataFrame, Dict]:
    
    company_df = company_df.sort_values('date_quarter')
    ticker = company_df[DEFAULT_TICKER_COLUMN].iloc[0]
    
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

    is_seasonal, period = detect_seasonality(series, ticker=ticker, column=column, verbose=verbose)

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

        deseasonalized = remove_seasonality(series, period)
        company_df[f'{column}_deseasonalized'] = deseasonalized.values

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
        
        if len(company_df) < SEASONALITY_MIN_DATA_POINTS:
            if verbose:
                print(f"WARNING: {ticker} has insufficient data points ({len(company_df)}), skipping seasonality detection")
            processed_dfs.append(company_df)
            continue
        
        company_df_before = company_df.copy()
        company_df, stats = process_company_seasonality(company_df, column, plot=False, verbose=False)
        
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
            processed_df, column, sample_size, verbose
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
