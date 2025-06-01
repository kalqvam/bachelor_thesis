import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('processed_data.csv')

if 'quarter' in df.columns and 'date_quarter' not in df.columns:
    df['date_quarter'] = pd.to_datetime(df['quarter'])
    print("Using 'quarter' as date column and creating 'date_quarter'")
else:
    df['date_quarter'] = pd.to_datetime(df['date_quarter'])
    print("Using existing 'date_quarter' column")

df = df.sort_values(['ticker', 'date_quarter'])

def detect_seasonality(series, max_lags=16, alpha=0.05, ticker="", column=""):
    if len(series) < max_lags + 5:
        print(f"[{ticker}] {column}: Insufficient data points ({len(series)}), skipping seasonality detection")
        return False, None

    variance = np.var(series.dropna())
    variance_threshold = 1e-7

    if variance < variance_threshold:
        print(f"[{ticker}] {column}: Series is effectively constant (variance={variance:.10f}), skipping seasonality detection")
        return False, None

    print(f"\n=== Seasonality Detection for {ticker} ({column}) ===")
    print(f"Time series length: {len(series)}")
    print(f"Series variance: {variance:.10f}")

    adf_result = adfuller(series.dropna())
    is_stationary = adf_result[1] < alpha

    print(f"ADF Test Results:")
    print(f"  ADF Statistic: {adf_result[0]:.4f}")
    print(f"  p-value: {adf_result[1]:.4f}")
    print(f"  Is stationary: {is_stationary}")
    print(f"  Critical Values: {', '.join([f'{k}: {v:.4f}' for k, v in adf_result[4].items()])}")

    from statsmodels.tsa.stattools import acf
    acf_values = acf(series.dropna(), nlags=max_lags, fft=True)

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

    print(f"ACF spikes detected at lags: {acf_spikes}")

    if len(acf_spikes) >= 2:
        potential_period = acf_spikes[0]
        print(f"First ACF spike at lag {potential_period}, considering as potential period")

        multiple_spikes = [lag for lag in acf_spikes if lag != potential_period and lag % potential_period == 0]

        if multiple_spikes:
            print(f"Confirmed seasonality: Found spikes at multiples of {potential_period}: {multiple_spikes}")

            significant_lags = []

            for lag in [potential_period] + multiple_spikes:
                lb_result = acorr_ljungbox(series.dropna(), lags=[lag])

                if isinstance(lb_result, tuple):
                    p_value = lb_result[1][0]
                else:
                    p_value = lb_result['lb_pvalue'].values[0]

                print(f"  Testing period {potential_period} at lag {lag}: Ljung-Box p-value = {p_value:.4f}")

                if p_value < alpha:
                    significant_lags.append(lag)

            if not significant_lags:
                print(f"  WARNING: Period {potential_period} has ACF spikes but no statistically significant lags. Will still apply STL decomposition.")

            final_period = potential_period
            if potential_period > 4:
                final_period = potential_period // 2
                print(f"First spike is at lag > 4, adjusting period from {potential_period} to {final_period}")

            print(f"Detected seasonality with period {final_period} for {ticker}")
            return True, final_period

    if len(acf_values) >= 7:
        print("\nPrimary spike detection method didn't find seasonality.")
        print("Checking for alternating pattern (period = 2) as fallback:")

        alternating_pattern_detected = False
        alternating_count = 0
        min_pairs_to_check = min(6, (len(acf_values) - 1) // 2)

        pct_changes = []
        for i in range(1, len(acf_values)-1):
            change = abs((acf_values[i+1] - acf_values[i]) / acf_values[i]) * 100
            pct_changes.append(change)
            print(f"  % change from lag {i} to {i+1}: {change:.2f}%")

        alternating_ratios = []
        for i in range(0, len(pct_changes)-1, 2):
            if pct_changes[i+1] > 0:
                ratio = pct_changes[i] / pct_changes[i+1]
                alternating_ratios.append(ratio)
                print(f"  Ratio of changes: ({i+1}->{i+2})/({i+2}->{i+3}) = {ratio:.2f}")

                if ratio <= 0.51:
                    alternating_count += 1

        if alternating_count >= min_pairs_to_check // 2:
            print(f"  Alternating pattern detected in {alternating_count}/{len(alternating_ratios)} pairs")
            print(f"  This suggests a period = 2 seasonality")
            print(f"Detected seasonality with period 2 for {ticker} (through alternating pattern)")
            return True, 2
        else:
            print(f"  No strong alternating pattern detected ({alternating_count}/{len(alternating_ratios)} pairs)")

    print(f"No seasonality detected for {ticker} in {column}")
    return False, None

def remove_seasonality(series, period):
    if len(series) < period * 2:
        print(f"WARNING: Time series too short ({len(series)}) for STL with period {period}, skipping")
        return series

    seasonal_smoothing = period if period % 2 == 1 else period + 1

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
        print(f"STL decomposition components:")
        print(f"  Trend range: {trend.min():.2f} to {trend.max():.2f}")
        print(f"  Seasonal range: {seasonal.min():.2f} to {seasonal.max():.2f}")
        print(f"  Residual range: {residual.min():.2f} to {residual.max():.2f}")
        print(f"  Seasonal strength: {seasonal_strength:.2f}%")

        deseasonalized = trend + residual

        return deseasonalized

    except Exception as e:
        print(f"ERROR in STL decomposition: {str(e)}")
        print("Falling back to original series")
        return series


def plot_stl_decomposition(series, period, ticker, column):
    if len(series) < period * 2:
        print(f"Time series too short for STL decomposition visualization")
        return

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

    except Exception as e:
        print(f"Error in STL decomposition visualization: {str(e)}")

def process_company(company_df, column, plot=False):
    company_df = company_df.sort_values('date_quarter')
    ticker = company_df['ticker'].iloc[0]

    print(f"\nProcessing {ticker} for {column}:")
    print(f"Number of data points: {len(company_df)}")

    series = company_df.set_index('date_quarter')[column].astype(float)

    missing_count = series.isna().sum()
    missing_pct = missing_count / len(series) * 100
    print(f"Missing values: {missing_count} ({missing_pct:.1f}%)")

    if missing_pct > 20:
        print(f"Too many missing values for {ticker} {column}, skipping seasonality detection")
        return company_df

    if missing_count > 0:
        series = series.interpolate(method='linear')
        print(f"Filled {missing_count} missing values with linear interpolation")

    min_val = series.min()
    max_val = series.max()

    if (abs(min_val - 0) < 1e-6 and abs(max_val - 0) < 1e-6) or (abs(min_val - 1) < 1e-6 and abs(max_val - 1) < 1e-6):
        print(f"RESULT: Series is constant at {min_val} (likely min-max normalized extreme), skipping seasonality detection")
        company_df[f'{column}_deseasonalized'] = company_df[column]
        return company_df

    is_seasonal, period = detect_seasonality(series, ticker=ticker, column=column)

    if is_seasonal and period is not None:
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
        print(f"Seasonal adjustment magnitude: {seasonal_magnitude:.2f} ({seasonal_pct:.1f}% of mean)")

        if plot:
            plt.figure(figsize=(12, 6))
            plt.plot(company_df['date_quarter'], company_df[column], label='Original')
            plt.plot(company_df['date_quarter'], company_df[f'{column}_deseasonalized'], label='Deseasonalized')
            plt.title(f"Deseasonalized {column} for {ticker}")
            plt.legend()
            plt.show()
    else:
        print(f"RESULT: No seasonality detected for {ticker} in {column}")
        company_df[f'{column}_deseasonalized'] = company_df[column]

    print("-" * 50)
    return company_df

def process_all_companies(df, column, sample_size=None):
    unique_tickers = df['ticker'].unique()
    print(f"Total unique tickers in dataset: {len(unique_tickers)}")

    if sample_size:
        unique_tickers = unique_tickers[:sample_size]
        print(f"Processing sample of {sample_size} tickers")

    processed_dfs = []
    seasonality_summary = {'total': 0, 'seasonal': 0, 'periods': []}

    for i, ticker in enumerate(unique_tickers):
        print(f"\n{'='*70}")
        print(f"Processing company {i+1}/{len(unique_tickers)}: {ticker}")

        company_df = df[df['ticker'] == ticker].copy()

        if len(company_df) < 8:
            print(f"WARNING: {ticker} has insufficient data points ({len(company_df)}), skipping seasonality detection")
            processed_dfs.append(company_df)
            continue

        company_df_before = company_df.copy()
        company_df = process_company(company_df, column)

        seasonality_summary['total'] += 1

        if not company_df[f'{column}_deseasonalized'].equals(company_df_before[column]):
            seasonality_summary['seasonal'] += 1

            is_seasonal = True
            period = None
            first_non_missing = company_df.dropna(subset=[column]).iloc[0]
            series = company_df[company_df['date_quarter'] >= first_non_missing['date_quarter']][column].astype(float)
            is_seasonal, period = detect_seasonality(series, ticker=ticker, column=column)

            if period and period not in seasonality_summary['periods']:
                seasonality_summary['periods'].append(period)

        processed_dfs.append(company_df)

        if (i+1) % 10 == 0 or i+1 == len(unique_tickers):
            print(f"Progress: {i+1}/{len(unique_tickers)} companies processed ({(i+1)/len(unique_tickers)*100:.1f}%)")

    processed_df = pd.concat(processed_dfs)

    print("\n" + "="*30 + " SUMMARY " + "="*30)
    seasonal_count = seasonality_summary['seasonal']
    total_count = seasonality_summary['total']
    seasonal_pct = (seasonal_count / total_count * 100) if total_count > 0 else 0
    periods = seasonality_summary['periods']

    print(f"{column.upper()}:")
    print(f"  Companies with seasonality: {seasonal_count}/{total_count} ({seasonal_pct:.1f}%)")
    print(f"  Detected seasonal periods: {periods}")
    print("="*70)

    return processed_df

def main(columns, sample_size=None, save_file=True, replace_original=True):
    if isinstance(columns, str):
        columns = [columns]

    processed_df = df.copy()

    include_quarter = 'quarter' in processed_df.columns and 'date_quarter' not in df.columns

    for column in columns:
        print(f"\n{'#'*30} Processing column: {column} {'#'*30}")

        current_df = process_all_companies(processed_df, column=column, sample_size=sample_size)
        processed_df = current_df.copy()

        print(f"Processed {len(processed_df['ticker'].unique())} companies for {column}")

        deseasonalized_col = f'{column}_deseasonalized'
        print(f"Added new column: {deseasonalized_col}")

        if replace_original:
            if deseasonalized_col in processed_df.columns:
                print(f"Replacing original {column} with deseasonalized values")
                processed_df[column] = processed_df[deseasonalized_col]
                processed_df.drop(columns=[deseasonalized_col], inplace=True)
                print(f"Dropped intermediate column {deseasonalized_col}")

    if include_quarter and 'date_quarter' in processed_df.columns:
        print("Keeping original 'quarter' column and dropping 'date_quarter'")
        processed_df.drop(columns=['date_quarter'], inplace=True)

    if save_file:
        output_file = 'processed_data_deseasonalized.csv'
        processed_df.to_csv(output_file, index=False)
        print(f"Saved processed data to {output_file}")

    return processed_df
