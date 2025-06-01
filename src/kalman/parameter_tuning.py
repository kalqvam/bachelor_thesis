import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from tqdm.auto import tqdm
import optuna
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.stats.diagnostic import acorr_ljungbox

from ..utils import (
    ESG_COLUMNS, KALMAN_DEFAULT_TRIALS, KALMAN_MIN_CONSECUTIVE_OBS,
    KALMAN_SIGMA_RANGE, print_subsection_header, format_number
)


def prepare_kalman_dataset(df: pd.DataFrame, 
                         min_consecutive_observations: int = KALMAN_MIN_CONSECUTIVE_OBS) -> Tuple[pd.DataFrame, List[str]]:
    df_prepared = df.copy()
    
    if 'quarter' in df_prepared.columns:
        if not pd.api.types.is_datetime64_any_dtype(df_prepared['quarter']):
            df_prepared['date_quarter'] = pd.to_datetime(df_prepared['quarter'])
        else:
            df_prepared['date_quarter'] = df_prepared['quarter']
    else:
        raise ValueError("Expected 'quarter' column not found in the dataset")

    df_prepared.sort_values(['ticker', 'date_quarter'], inplace=True)
    
    valid_tickers = []
    esg_columns = [col for col in ESG_COLUMNS if col in df_prepared.columns]
    
    for ticker in df_prepared['ticker'].unique():
        ticker_data = df_prepared[df_prepared['ticker'] == ticker].sort_values('date_quarter')
        
        has_enough_data = True
        for col in esg_columns:
            non_missing = ~ticker_data[col].isna()
            
            consecutive_counts = []
            count = 0
            for val in non_missing:
                if val:
                    count += 1
                else:
                    if count > 0:
                        consecutive_counts.append(count)
                    count = 0
            if count > 0:
                consecutive_counts.append(count)
            
            if not consecutive_counts or max(consecutive_counts) < min_consecutive_observations:
                has_enough_data = False
                break
        
        if has_enough_data:
            valid_tickers.append(ticker)
    
    df_filtered = df_prepared[df_prepared['ticker'].isin(valid_tickers)].copy()
    
    return df_filtered, valid_tickers


def validate_kalman_data(df: pd.DataFrame, columns: List[str]) -> Dict[str, any]:
    validation_stats = {
        'total_tickers': df['ticker'].nunique(),
        'column_coverage': {},
        'data_quality': {},
        'warnings': []
    }
    
    for column in columns:
        if column not in df.columns:
            validation_stats['warnings'].append(f"Column {column} not found")
            continue
        
        non_missing = df[column].notna().sum()
        total_obs = len(df)
        coverage_rate = non_missing / total_obs if total_obs > 0 else 0
        
        validation_stats['column_coverage'][column] = {
            'observations': non_missing,
            'coverage_rate': coverage_rate
        }
        
        if coverage_rate < 0.5:
            validation_stats['warnings'].append(f"Low coverage for {column}: {coverage_rate:.1%}")
    
    return validation_stats


def fit_kalman_model(series: pd.Series, 
                    sigma2_level: float, 
                    sigma2_obs: float, 
                    get_innovations: bool = False) -> Tuple[Optional[object], Optional[np.ndarray]]:
    clean_data = series.dropna()
    
    if len(clean_data) < 10:
        return None, None
    
    try:
        clean_data_values = clean_data.values
        model = UnobservedComponents(clean_data_values, level='local level')
        initial_params = [sigma2_level, sigma2_obs]
        result = model.fit(initial_params, method='powell', maxiter=100, disp=False)
        
        if get_innovations:
            filter_results = result.filter_results
            innovations = filter_results.forecasts_error[0]
            return result, innovations
        else:
            return result, None
            
    except Exception:
        return None, None


def calculate_harvey_likelihood(series: pd.Series, model_result: object) -> float:
    filter_results = model_result.filter_results
    innovations = filter_results.forecasts_error[0]
    forecast_error_variances = filter_results.forecasts_error_cov[0, 0, :]
    
    n = len(innovations)
    if n == 0:
        return -np.inf
    
    term1 = -n / 2 * np.log(2 * np.pi)
    term2 = -0.5 * np.sum(np.log(forecast_error_variances))
    term3 = -0.5 * np.sum(innovations**2 / forecast_error_variances)
    
    return term1 + term2 + term3


def evaluate_white_noise_residuals(innovations: np.ndarray, 
                                 alpha: float = 0.05) -> Tuple[bool, float, List[float]]:
    if len(innovations) < 5:
        return False, np.nan, []
    
    try:
        lb_test = acorr_ljungbox(innovations, lags=[1, 2, 3, 4], return_df=True)
        p_values = lb_test['lb_pvalue'].values
        
        is_white_noise = all(p > alpha for p in p_values)
        lb_stat = lb_test['lb_stat'].iloc[-1]
        
        return is_white_noise, lb_stat, p_values.tolist()
    except Exception:
        return False, np.nan, []


def create_optimization_objective(df: pd.DataFrame, 
                                column: str, 
                                valid_tickers: List[str]) -> callable:
    def objective(trial):
        sigma2_level = trial.suggest_float("sigma2_level", *KALMAN_SIGMA_RANGE, log=True)
        sigma2_obs = trial.suggest_float("sigma2_obs", *KALMAN_SIGMA_RANGE, log=True)
        
        all_likelihoods = []
        successful_fits = 0
        
        for ticker in valid_tickers:
            ticker_data = df[df['ticker'] == ticker].sort_values('date_quarter')
            series = ticker_data[column]
            
            if series.dropna().size < 10:
                continue
            
            model_result, _ = fit_kalman_model(series, sigma2_level, sigma2_obs)
            
            if model_result is not None:
                log_likelihood = calculate_harvey_likelihood(series, model_result)
                all_likelihoods.append(log_likelihood)
                successful_fits += 1
        
        if not all_likelihoods or successful_fits < 0.5 * len(valid_tickers):
            return -float('inf')
        
        return sum(all_likelihoods)
    
    return objective


def optimize_parameters_for_column(df: pd.DataFrame,
                                 column: str,
                                 valid_tickers: List[str],
                                 n_trials: int = KALMAN_DEFAULT_TRIALS,
                                 verbose: bool = True) -> Optional[Dict]:
    if verbose:
        print_subsection_header(f"Optimizing Parameters for {column}")
    
    if column not in df.columns:
        if verbose:
            print(f"Column {column} not found")
        return None
    
    objective_func = create_optimization_objective(df, column, valid_tickers)
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_func, n_trials=n_trials)
    
    if np.isneginf(study.best_value):
        if verbose:
            print(f"No valid parameters found for {column}")
        return None
    
    best_params = study.best_params
    result = {
        'column': column,
        'sigma2_level': best_params['sigma2_level'],
        'sigma2_obs': best_params['sigma2_obs'],
        'log_likelihood': study.best_value
    }
    
    if verbose:
        print(f"Best parameters for {column}:")
        print(f"  sigma2_level: {result['sigma2_level']:.6f}")
        print(f"  sigma2_obs: {result['sigma2_obs']:.6f}")
        print(f"  log_likelihood: {result['log_likelihood']:.4f}")
    
    return result


def validate_optimal_parameters(df: pd.DataFrame,
                               column: str,
                               valid_tickers: List[str],
                               params: Dict,
                               verbose: bool = True) -> Dict:
    if verbose:
        print(f"Validating parameters for {column}...")
    
    sigma2_level = params['sigma2_level']
    sigma2_obs = params['sigma2_obs']
    
    white_noise_count = 0
    total_evaluated = 0
    all_pvalues = []
    
    for ticker in tqdm(valid_tickers, desc="Testing white noise", disable=not verbose):
        ticker_data = df[df['ticker'] == ticker].sort_values('date_quarter')
        series = ticker_data[column]
        
        if series.dropna().size < 10:
            continue
        
        model_result, innovations = fit_kalman_model(
            series, sigma2_level, sigma2_obs, get_innovations=True
        )
        
        if model_result is not None and innovations is not None:
            is_white_noise, _, p_values = evaluate_white_noise_residuals(innovations)
            
            if p_values:
                all_pvalues.extend(p_values)
            
            if is_white_noise:
                white_noise_count += 1
            total_evaluated += 1
    
    white_noise_pct = (white_noise_count / total_evaluated * 100) if total_evaluated > 0 else 0
    avg_pvalue = np.mean(all_pvalues) if all_pvalues else np.nan
    
    validation_results = {
        'white_noise_percentage': white_noise_pct,
        'white_noise_count': white_noise_count,
        'total_evaluated': total_evaluated,
        'average_pvalue': avg_pvalue
    }
    
    if verbose:
        print(f"White noise validation: {white_noise_pct:.1f}% ({white_noise_count}/{total_evaluated})")
        if not np.isnan(avg_pvalue):
            print(f"Average p-value: {avg_pvalue:.4f}")
    
    return validation_results


def optimize_all_esg_parameters(df: pd.DataFrame,
                               columns: Optional[List[str]] = None,
                               min_consecutive_observations: int = KALMAN_MIN_CONSECUTIVE_OBS,
                               n_trials: int = KALMAN_DEFAULT_TRIALS,
                               max_companies: Optional[int] = None,
                               verbose: bool = True) -> Tuple[Optional[pd.DataFrame], Dict]:
    if verbose:
        print_subsection_header("ESG Parameter Optimization")
    
    df_prepared, valid_tickers = prepare_kalman_dataset(df, min_consecutive_observations)
    
    if not valid_tickers:
        if verbose:
            print("No valid tickers found with sufficient data")
        return None, {'error': 'no_valid_tickers'}
    
    if max_companies and max_companies < len(valid_tickers):
        valid_tickers = valid_tickers[:max_companies]
        if verbose:
            print(f"Limited analysis to {max_companies} companies")
    
    if columns is None:
        columns = [col for col in ESG_COLUMNS if col in df_prepared.columns]
    
    optimal_params = []
    validation_stats = {}
    
    for column in columns:
        params = optimize_parameters_for_column(
            df_prepared, column, valid_tickers, n_trials, verbose
        )
        
        if params:
            validation = validate_optimal_parameters(
                df_prepared, column, valid_tickers, params, verbose
            )
            params.update(validation)
            optimal_params.append(params)
            validation_stats[column] = validation
    
    if optimal_params:
        params_df = pd.DataFrame(optimal_params)
        
        summary_stats = {
            'total_columns_processed': len(columns),
            'successful_optimizations': len(optimal_params),
            'valid_tickers_used': len(valid_tickers),
            'validation_stats': validation_stats
        }
        
        if verbose:
            print(f"\nOptimization Summary:")
            print(f"Processed {len(columns)} columns")
            print(f"Successful optimizations: {len(optimal_params)}")
            print(f"Used {len(valid_tickers)} tickers")
        
        return params_df, summary_stats
    else:
        if verbose:
            print("No successful parameter optimizations")
        return None, {'error': 'no_successful_optimizations'}
