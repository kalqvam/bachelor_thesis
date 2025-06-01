import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.structural import UnobservedComponents
import warnings
from tqdm import tqdm
import argparse

warnings.filterwarnings('ignore')

def apply_kalman_filter(data_series, sigma2_level, sigma2_obs):
    data = data_series.copy()

    model = UnobservedComponents(
        data,
        level='local level',
        missing='fill'
    )

    initial_params = [sigma2_level, sigma2_obs]
    result = model.fit(initial_params, method='powell', maxiter=100, disp=False)

    smoothed_states = result.smoother_results.smoothed_state
    trend = pd.Series(smoothed_states[0], index=data.index)

    return trend

def process_esg_data(file_path, e_sigma2_level, e_sigma2_obs, s_sigma2_level, s_sigma2_obs,
                     g_sigma2_level, g_sigma2_obs, output_file=None):
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)

    if 'quarter' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['quarter']):
            df['date_quarter'] = pd.to_datetime(df['quarter'])
        else:
            df['date_quarter'] = df['quarter']
    else:
        raise ValueError("Expected 'quarter' column not found in the dataset")

    df.sort_values(['ticker', 'date_quarter'], inplace=True)

    esg_columns = ['environmentalScore', 'socialScore', 'governanceScore']

    parameter_pairs = {
        'environmentalScore': (e_sigma2_level, e_sigma2_obs),
        'socialScore': (s_sigma2_level, s_sigma2_obs),
        'governanceScore': (g_sigma2_level, g_sigma2_obs)
    }

    result_df = df.copy()

    print("Processing companies...")
    for ticker in tqdm(df['ticker'].unique()):
        ticker_data = df[df['ticker'] == ticker].copy()

        for column in esg_columns:
            series = ticker_data[column]

            if series.isna().all():
                continue

            sigma2_level, sigma2_obs = parameter_pairs[column]

            try:
                trend = apply_kalman_filter(series, sigma2_level, sigma2_obs)

                result_idx = result_df[result_df['ticker'] == ticker].index
                result_df.loc[result_idx, column] = trend.values

            except Exception as e:
                print(f"Error processing {ticker} for {column}: {e}")

    print("Processing completed!")

    if output_file:
        result_df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")

    return result_df
