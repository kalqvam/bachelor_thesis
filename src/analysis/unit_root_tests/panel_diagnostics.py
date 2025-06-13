import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from typing import Dict, List, Optional, Tuple, Any

from ..utils import print_subsection_header, format_number


def pesaran_cd_test(data: pd.DataFrame, variable: str) -> Dict[str, Any]:
    pivot_data = data.pivot(index='date', columns='ticker', values=variable)

    min_periods = 5
    valid_cols = pivot_data.columns[pivot_data.count() >= min_periods]
    pivot_data = pivot_data[valid_cols]

    N = len(valid_cols)
    T = pivot_data.count().min()

    if N <= 1:
        return {'cd_stat': np.nan, 'p_value': np.nan, 'error': 'Not enough cross-sections'}

    corr_matrix = pivot_data.corr()

    sum_corr = 0
    count = 0

    for i in range(N):
        for j in range(i+1, N):
            if not np.isnan(corr_matrix.iloc[i, j]):
                sum_corr += corr_matrix.iloc[i, j]
                count += 1

    if count == 0:
        return {'cd_stat': np.nan, 'p_value': np.nan, 'error': 'No valid correlations'}

    cd_stat = np.sqrt(2 * T / (N * (N - 1))) * sum_corr
    p_value = 2 * (1 - stats.norm.cdf(abs(cd_stat)))

    return {'cd_stat': cd_stat, 'p_value': p_value, 'N': N, 'T': T}


def adf_test(x: np.ndarray, trend: bool = False, max_lags: Optional[int] = None) -> Dict[str, Any]:
    x = np.array(x)

    valid_indices = ~np.isnan(x)
    if not np.all(valid_indices):
        x = x[valid_indices]

    if len(x) < 10:
        return {'adf_stat': np.nan, 'p_value': np.nan, 'error': f'Not enough observations (only {len(x)} valid points)'}

    if np.std(x) < 1e-10:
        return {'adf_stat': np.nan, 'p_value': np.nan, 'error': 'Series has no variation'}

    try:
        dx = np.diff(x)

        if max_lags is None:
            max_lags = min(int(12 * (len(x)/100)**(1/4)), int(len(x)/3))
            max_lags = max(1, max_lags)

        best_aic = np.inf
        best_lag = 0
        result = {}

        for lag in range(max_lags + 1):
            if len(dx) - lag < 3:
                continue

            y = dx[lag:]

            X_list = [x[lag:-1]]

            if trend:
                X_list.append(np.arange(1, len(y) + 1))

            for p in range(1, lag + 1):
                if lag-p < 0 or -p >= len(dx):
                    continue
                X_list.append(dx[lag-p:-p])

            try:
                if len(X_list) > 1:
                    X = np.column_stack(X_list)
                else:
                    X = X_list[0].reshape(-1, 1)

                X = sm.add_constant(X)

                model = sm.OLS(y, X).fit()

                aic = model.aic

                if aic < best_aic:
                    best_aic = aic
                    best_lag = lag
                    result['adf_stat'] = model.params[1] / model.bse[1]
                    result['p_value'] = model.pvalues[1]
                    result['coefficients'] = model.params
                    result['std_errors'] = model.bse
                    result['residuals'] = model.resid
                    result['lag'] = lag
                    result['aic'] = aic
            except Exception as e:
                continue

        if 'adf_stat' not in result:
            return {'adf_stat': np.nan, 'p_value': np.nan, 'error': 'Regression failed for all lags'}

        return result
    except Exception as e:
        return {'adf_stat': np.nan, 'p_value': np.nan, 'error': f'Exception in ADF test: {str(e)}'}


def panel_adf_test(data_panel: np.ndarray, trend: bool = False) -> Dict[str, Any]:
    N, T = data_panel.shape

    adf_stats = []
    p_values = []

    for i in range(N):
        series = data_panel[i, :]
        adf_result = adf_test(series, trend=trend)

        if 'error' not in adf_result:
            adf_stats.append(adf_result['adf_stat'])
            p_values.append(adf_result['p_value'])

    adf_stats = np.array(adf_stats)
    p_values = np.array(p_values)

    valid_adf = ~np.isnan(adf_stats)
    adf_stats = adf_stats[valid_adf]
    p_values = p_values[valid_adf]

    if len(adf_stats) == 0:
        return {'Pb_stat': np.nan, 'p_value': np.nan, 'error': 'No valid ADF statistics'}

    Pb = np.mean(adf_stats)

    Pb_p_value = np.nan

    k = 1 if not trend else 2
    mu = -1.5 if not trend else -2.02
    sigma = 0.867 if not trend else 0.55

    std_adf = (adf_stats - mu) / sigma
    std_Pb = np.sum(std_adf) / np.sqrt(len(std_adf))

    return {
        'Pb_stat': Pb,
        'std_Pb_stat': std_Pb,
        'p_value': Pb_p_value,
        'individual_adf': adf_stats.tolist(),
        'N_valid': len(adf_stats),
        'N_total': N
    }


def validate_panel_data(data: pd.DataFrame, variables: List[str], verbose: bool = True) -> Dict[str, Any]:
    validation_results = {
        'valid_variables': [],
        'invalid_variables': [],
        'missing_variables': [],
        'data_quality': {}
    }
    
    if verbose:
        print_subsection_header("Validating Panel Data")
        print(f"Checking {len(variables)} variables")
    
    for variable in variables:
        if variable not in data.columns:
            validation_results['missing_variables'].append(variable)
            if verbose:
                print(f"✗ {variable}: Column not found")
            continue
        
        if not pd.api.types.is_numeric_dtype(data[variable]):
            validation_results['invalid_variables'].append(variable)
            if verbose:
                print(f"✗ {variable}: Non-numeric data type")
            continue
        
        missing_rate = data[variable].isna().sum() / len(data)
        
        if missing_rate > 0.8:
            validation_results['invalid_variables'].append(variable)
            validation_results['data_quality'][variable] = {
                'missing_rate': missing_rate,
                'reason': 'too_many_missing'
            }
            if verbose:
                print(f"✗ {variable}: Too many missing values ({missing_rate:.1%})")
            continue
        
        validation_results['valid_variables'].append(variable)
        validation_results['data_quality'][variable] = {
            'missing_rate': missing_rate,
            'observations': data[variable].notna().sum()
        }
        if verbose:
            print(f"✓ {variable}: Valid ({missing_rate:.1%} missing)")
    
    if verbose:
        print(f"Validation complete: {len(validation_results['valid_variables'])}/{len(variables)} variables valid")
    
    return validation_results


def prepare_panel_data(data: pd.DataFrame, variable: str, verbose: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    dupes = data.duplicated(subset=['ticker', 'date'])
    if dupes.any():
        data = data.drop_duplicates(subset=['ticker', 'date'])
        if verbose:
            print(f"Removed {dupes.sum()} duplicate ticker-date combinations")
    
    pivot_data = data.pivot(index='date', columns='ticker', values=variable)
    pivot_data = pivot_data.sort_index()
    
    if verbose:
        print(f"Created pivot table: {pivot_data.shape[0]} periods × {pivot_data.shape[1]} entities")
    
    pivot_data = pivot_data.fillna(method='ffill').fillna(method='bfill')
    
    missing_rate = pivot_data.isna().sum().sum() / (pivot_data.shape[0] * pivot_data.shape[1])
    
    if verbose:
        print(f"Missing data rate after forward/backward fill: {missing_rate:.2%}")
    
    if missing_rate > 0.2:
        if verbose:
            print(f"Error: Too many missing values ({missing_rate:.2%} > 20%)")
        return None, {
            'error': 'Too many missing values',
            'missing_rate': missing_rate
        }
    
    valid_cols = pivot_data.columns[pivot_data.count() >= 10]
    pivot_data = pivot_data[valid_cols]
    
    if verbose:
        print(f"Retained {len(valid_cols)} entities with ≥10 observations")
    
    if len(valid_cols) < 5:
        if verbose:
            print(f"Error: Not enough entities with sufficient data ({len(valid_cols)} < 5)")
        return None, {
            'error': 'Not enough entities with sufficient data',
            'valid_entities': len(valid_cols)
        }
    
    stats = {
        'N': pivot_data.shape[1],
        'T': pivot_data.shape[0],
        'missing_rate': missing_rate,
        'entities_retained': len(valid_cols)
    }
    
    return pivot_data, stats
