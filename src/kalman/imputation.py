import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from tqdm.auto import tqdm
from datetime import datetime
import os

from .parameter_tuning import fit_kalman_model, evaluate_white_noise_residuals
from ..utils import (
    DEFAULT_OUTPUT_DIR, OUTPUT_TIMESTAMP_FORMAT, 
    print_subsection_header, format_number, ensure_directory_exists
)


def extract_kalman_components(series: pd.Series,
                            sigma2_level: float,
                            sigma2_obs: float) -> Optional[Dict]:
    clean_data = series.dropna()
    
    if len(clean_data) < 10:
        return None
    
    try:
        clean_data_values = clean_data.values
        model = UnobservedComponents(clean_data_values, level='local level')
        initial_params = [sigma2_level, sigma2_obs]
        result = model.fit(initial_params, method='powell', maxiter=100, disp=False)
        
        filter_results = result.filter_results
        innovations = filter_results.forecasts_error[0]
        is_white_noise, _, _ = evaluate_white_noise_residuals(innovations)
        
        states = result.smoother_results.smoothed_state[0]
        
        full_smoothed = pd.Series(index=series.index, dtype=float)
        clean_indices = clean_data.index
        
        for i, idx in enumerate(clean_indices):
            if i < len(states):
                full_smoothed.loc[idx] = states[i]
        
        full_smoothed = full_smoothed.interpolate(method='linear')
        
        components = []
        for idx in series.index:
            original = series.loc[idx]
            trend = full_smoothed.loc[idx] if not pd.isna(full_smoothed.loc[idx]) else np.nan
            residual = original - trend if not pd.isna(original) and not pd.isna(trend) else np.nan
            
            components.append({
                'index': idx,
                'original': original,
                'trend': trend,
                'residual': residual
            })
        
        return {
            'components': components,
            'full_smoothed': full_smoothed,
            'is_white_noise': is_white_noise,
            'model_result': result
        }
        
    except Exception:
        return None


def apply_kalman_to_ticker(ticker_data: pd.DataFrame,
                         column: str,
                         params: Dict,
                         date_column: str = 'date_quarter') -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
    if column not in ticker_data.columns:
        return None, None
    
    ticker = ticker_data['ticker'].iloc[0]
    series = ticker_data[column]
    
    if series.dropna().size < 10:
        return None, None
    
    result = extract_kalman_components(
        series, params['sigma2_level'], params['sigma2_obs']
    )
    
    if result is None:
        return None, None
    
    components_data = []
    for comp in result['components']:
        row_data = ticker_data.loc[comp['index']].copy()
        components_data.append({
            'ticker': ticker,
            'date_quarter': row_data[date_column],
            'column': column,
            'original': comp['original'],
            'trend': comp['trend'],
            'residual': comp['residual'],
            'is_white_noise': result['is_white_noise']
        })
    
    components_df = pd.DataFrame(components_data)
    smoothed_series = result['full_smoothed']
    
    return components_df, smoothed_series


def process_all_tickers(df: pd.DataFrame,
                       column: str,
                       params: Dict,
                       verbose: bool = True) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    if verbose:
        print(f"Processing {column} for all tickers...")
    
    df_prepared = df.copy()
    if 'date_quarter' not in df_prepared.columns:
        if 'quarter' in df_prepared.columns:
            df_prepared['date_quarter'] = pd.to_datetime(df_prepared['quarter'])
        else:
            raise ValueError("No suitable date column found")
    
    all_components = []
    updated_df = df_prepared.copy()
    successful_tickers = 0
    
    for ticker in tqdm(df_prepared['ticker'].unique(), desc=f"Processing {column}"):
        ticker_data = df_prepared[df_prepared['ticker'] == ticker].sort_values('date_quarter')
        
        components_df, smoothed_series = apply_kalman_to_ticker(ticker_data, column, params)
        
        if components_df is not None and smoothed_series is not None:
            all_components.append(components_df)
            
            ticker_mask = updated_df['ticker'] == ticker
            ticker_indices = updated_df[ticker_mask].index
            
            for idx in ticker_indices:
                if idx in smoothed_series.index and not pd.isna(smoothed_series.loc[idx]):
                    updated_df.loc[idx, column] = smoothed_series.loc[idx]
            
            successful_tickers += 1
    
    if all_components:
        components_result = pd.concat(all_components, ignore_index=True)
        
        if verbose:
            print(f"Successfully processed {successful_tickers} tickers")
            print(f"Generated {len(components_result)} component observations")
        
        return components_result, updated_df
    else:
        if verbose:
            print(f"No components extracted for {column}")
        return None, df_prepared


def create_components_dataframe(results: List[pd.DataFrame]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()
    
    combined_df = pd.concat(results, ignore_index=True)
    
    combined_df['date_quarter'] = pd.to_datetime(combined_df['date_quarter'])
    combined_df = combined_df.sort_values(['ticker', 'column', 'date_quarter'])
    
    return combined_df


def apply_kalman_imputation(df: pd.DataFrame,
                          optimal_params: pd.DataFrame,
                          save_results: bool = True,
                          output_dir: str = DEFAULT_OUTPUT_DIR,
                          verbose: bool = True) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Dict]:
    if verbose:
        print_subsection_header("Applying Kalman Filter Imputation")
    
    if optimal_params is None or optimal_params.empty:
        if verbose:
            print("No optimal parameters provided")
        return None, None, {'error': 'no_parameters'}
    
    all_components = []
    updated_df = df.copy()
    processing_stats = {}
    
    for _, row in optimal_params.iterrows():
        column = row['column']
        params = {
            'sigma2_level': row['sigma2_level'],
            'sigma2_obs': row['sigma2_obs']
        }
        
        components_df, column_updated_df = process_all_tickers(updated_df, column, params, verbose)
        
        if components_df is not None and column_updated_df is not None:
            all_components.append(components_df)
            updated_df = column_updated_df
            
            original_missing = df[column].isna().sum()
            final_missing = updated_df[column].isna().sum()
            imputed_count = original_missing - final_missing
            
            processing_stats[column] = {
                'tickers_processed': components_df['ticker'].nunique(),
                'observations_generated': len(components_df),
                'original_missing': original_missing,
                'final_missing': final_missing,
                'values_imputed': imputed_count,
                'white_noise_percentage': (components_df['is_white_noise'].sum() / len(components_df) * 100) if len(components_df) > 0 else 0
            }
        else:
            processing_stats[column] = {'error': 'no_components_extracted'}
    
    if all_components:
        final_components_df = create_components_dataframe(all_components)
        
        summary_stats = {
            'total_columns_processed': len(optimal_params),
            'successful_columns': len(all_components),
            'total_component_observations': len(final_components_df),
            'unique_tickers': final_components_df['ticker'].nunique() if not final_components_df.empty else 0,
            'processing_details': processing_stats
        }
        
        if save_results:
            output_path = save_kalman_results(final_components_df, optimal_params, output_dir, verbose)
            summary_stats['output_path'] = output_path
        
        if verbose:
            print(f"\nImputation Summary:")
            print(f"Processed {len(optimal_params)} columns")
            print(f"Generated {len(final_components_df)} component observations")
            print(f"Covered {final_components_df['ticker'].nunique()} unique tickers")
            
            for col, stats in processing_stats.items():
                if 'values_imputed' in stats:
                    print(f"{col}: imputed {stats['values_imputed']} missing values")
        
        return final_components_df, updated_df, summary_stats
    else:
        if verbose:
            print("No components extracted for any column")
        return None, df.copy(), {'error': 'no_results'}


def save_kalman_results(components_df: pd.DataFrame,
                       params_df: pd.DataFrame,
                       output_dir: str = DEFAULT_OUTPUT_DIR,
                       verbose: bool = True) -> str:
    output_path = ensure_directory_exists(output_dir)
    timestamp = datetime.now().strftime(OUTPUT_TIMESTAMP_FORMAT)
    
    components_file = output_path / f"kalman_components_{timestamp}.csv"
    params_file = output_path / f"kalman_parameters_{timestamp}.csv"
    
    components_df.to_csv(components_file, index=False)
    params_df.to_csv(params_file, index=False)
    
    if verbose:
        print(f"Components saved to: {components_file}")
        print(f"Parameters saved to: {params_file}")
    
    return str(output_path)


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
    
    df_prepared = df.copy()
    
    if 'quarter' in df_prepared.columns:
        if not pd.api.types.is_datetime64_any_dtype(df_prepared['quarter']):
            df_prepared['date_quarter'] = pd.to_datetime(df_prepared['quarter'])
        else:
            df_prepared['date_quarter'] = df_prepared['quarter']
    else:
        raise ValueError("Expected 'quarter' column not found")
    
    df_prepared.sort_values(['ticker', 'date_quarter'], inplace=True)
    
    esg_columns = ['environmentalScore', 'socialScore', 'governanceScore']
    parameter_pairs = {
        'environmentalScore': (sigma2_level_e, sigma2_obs_e),
        'socialScore': (sigma2_level_s, sigma2_obs_s),
        'governanceScore': (sigma2_level_g, sigma2_obs_g)
    }
    
    result_df = df_prepared.copy()
    
    if verbose:
        print("Processing companies...")
    
    for ticker in tqdm(df_prepared['ticker'].unique()):
        ticker_data = df_prepared[df_prepared['ticker'] == ticker].copy()
        
        for column in esg_columns:
            if column not in ticker_data.columns:
                continue
            
            series = ticker_data[column]
            
            if series.isna().all():
                continue
            
            sigma2_level, sigma2_obs = parameter_pairs[column]
            
            try:
                model_result, _ = fit_kalman_model(series, sigma2_level, sigma2_obs)
                
                if model_result is not None:
                    states = model_result.smoother_results.smoothed_state
                    clean_indices = series.dropna().index
                    
                    result_idx = result_df[result_df['ticker'] == ticker].index
                    
                    for i, idx in enumerate(clean_indices):
                        if i < len(states[0]) and idx in result_idx:
                            result_df.loc[idx, column] = states[0][i]
                
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
