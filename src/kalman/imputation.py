import pandas as pd
import numpy as np
from typing import Optional
from tqdm.auto import tqdm
from statsmodels.tsa.statespace.structural import UnobservedComponents

from ..utils import print_subsection_header


def apply_esg_kalman_imputation(df: pd.DataFrame,
                              sigma2_level_e: float,
                              sigma2_obs_e: float,
                              sigma2_level_s: float,
                              sigma2_obs_s: float,
                              sigma2_level_g: float,
                              sigma2_obs_g: float,
                              output_file: Optional[str] = None,
                              verbose: bool = True) -> pd.DataFrame:
    if verbose:
        print_subsection_header("Applying ESG Kalman Filter")
    
    result_df = df.copy()
    
    esg_columns = ['environmentalScore', 'socialScore', 'governanceScore']
    parameter_pairs = {
        'environmentalScore': (sigma2_level_e, sigma2_obs_e),
        'socialScore': (sigma2_level_s, sigma2_obs_s),
        'governanceScore': (sigma2_level_g, sigma2_obs_g)
    }
    
    if verbose:
        print("Processing companies...")
    
    for ticker in tqdm(df['ticker'].unique(), disable=not verbose):
        ticker_data = df[df['ticker'] == ticker]
        
        for column in esg_columns:
            if column not in ticker_data.columns:
                continue
            
            series = ticker_data[column]
            
            if series.dropna().size < 10:
                continue
            
            sigma2_level, sigma2_obs = parameter_pairs[column]
            
            try:
                model = UnobservedComponents(series.values, level='local level')
                initial_params = [sigma2_level, sigma2_obs]
                result = model.fit(initial_params, method='powell', maxiter=100, disp=False)
                
                smoothed_states = result.smoother_results.smoothed_state[0]
                
                ticker_indices = result_df[result_df['ticker'] == ticker].index
                
                for i, idx in enumerate(ticker_indices):
                    if i < len(smoothed_states):
                        result_df.loc[idx, column] = smoothed_states[i]
                
            except Exception as e:
                if verbose:
                    print(f"Error processing {ticker} for {column}: {e}")
    
    if output_file:
        result_df.to_csv(output_file, index=False)
        if verbose:
            print(f"Results saved to {output_file}")
    
    if verbose:
        print("Kalman filter imputation completed!")
    
    return result_df
