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
