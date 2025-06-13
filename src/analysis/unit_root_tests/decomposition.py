import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from typing import Dict, Tuple, Any

from ...utils import print_subsection_header


def bai_ng_ic(X: np.ndarray, max_factors: int = 25, cumulative_var_threshold: float = 0.85) -> Tuple[int, Dict[str, Any]]:
    N, T = X.shape

    has_nans = np.isnan(X).any()
    if has_nans:
        X = np.nan_to_num(X, nan=0.0)

    try:
        X_demean = X - np.mean(X, axis=1, keepdims=True)
    except Exception as e:
        X_demean = X

    actual_max = min(max_factors, min(N, T) - 1)

    try:
        if T > N:
            cov_matrix = np.dot(X_demean, X_demean.T) / T
            eigvals, eigvecs = np.linalg.eigh(cov_matrix)
            idx = np.argsort(eigvals)[::-1]
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]

            total_var = np.sum(eigvals)

            cum_var_ratio = np.cumsum(eigvals) / total_var
            var_threshold_r = np.argmax(cum_var_ratio >= cumulative_var_threshold) + 1
            var_threshold_r = min(var_threshold_r, actual_max)

            IC1 = []
            IC2 = []

            V0 = total_var

            for r in range(1, actual_max + 1):
                explained_var = np.sum(eigvals[:r])
                V_r = total_var - explained_var

                V_r = V_r / total_var

                penalty1 = r * (N + T) / (N * T) * np.log(N * T / (N + T))
                penalty2 = r * (N + T) / (N * T) * np.log(min(N, T))

                ic1_value = np.log(V_r) + penalty1
                ic2_value = np.log(V_r) + penalty2

                IC1.append(ic1_value)
                IC2.append(ic2_value)
        else:
            pca = PCA(n_components=actual_max)
            pca.fit(X_demean.T)

            cum_var_ratio = np.cumsum(pca.explained_variance_ratio_)
            var_threshold_r = np.argmax(cum_var_ratio >= cumulative_var_threshold) + 1
            var_threshold_r = min(var_threshold_r, actual_max)

            IC1 = []
            IC2 = []

            V0 = np.mean((X_demean)**2)

            for r in range(1, actual_max + 1):
                components_r = pca.components_[:r]

                transformed = pca.transform(X_demean.T)[:, :r]

                X_hat = np.dot(transformed, components_r).T

                resid = X_demean - X_hat
                V_r = np.mean(resid**2)

                penalty1 = r * (N + T) / (N * T) * np.log(N * T / (N + T))
                penalty2 = r * (N + T) / (N * T) * np.log(min(N, T))

                ic1_value = np.log(V_r) + penalty1
                ic2_value = np.log(V_r) + penalty2

                IC1.append(ic1_value)
                IC2.append(ic2_value)

        optimal_r1 = np.nanargmin(IC1) + 1 if not np.all(np.isnan(IC1)) else 1
        optimal_r2 = np.nanargmin(IC2) + 1 if not np.all(np.isnan(IC2)) else 1

        max_reasonable = min(N, T) - 3

        optimal_r = optimal_r1
        selection_method = "IC1"

        if optimal_r1 == actual_max and optimal_r2 < actual_max:
            optimal_r = optimal_r2
            selection_method = "IC2"
        elif optimal_r1 == actual_max and optimal_r2 == actual_max:
            optimal_r = var_threshold_r
            selection_method = "Cumulative variance threshold"

        if optimal_r > max_reasonable and optimal_r2 > max_reasonable:
            optimal_r = var_threshold_r
            selection_method = "Cumulative variance threshold (after IC exceeded max reasonable)"

            if optimal_r > max_reasonable:
                optimal_r = max_reasonable
                selection_method = "Max reasonable (fallback)"
        elif optimal_r > max_reasonable:
            optimal_r = max_reasonable

        if optimal_r > max_reasonable:
            optimal_r = max_reasonable

        return optimal_r, {
            'IC1': IC1,
            'IC2': IC2,
            'optimal_r1': optimal_r1,
            'optimal_r2': optimal_r2,
            'var_threshold_r': var_threshold_r,
            'cum_var_ratio': cum_var_ratio.tolist(),
            'selection_method': selection_method
        }

    except Exception as e:
        optimal_r = min(3, min(N, T) // 5)
        return optimal_r, {
            'IC1': [],
            'IC2': [],
            'optimal_r1': optimal_r,
            'optimal_r2': optimal_r,
            'var_threshold_r': optimal_r,
            'cum_var_ratio': [],
            'selection_method': "Error fallback"
        }


def panic_decomposition(data_panel: pd.DataFrame, cumulative_var_threshold: float = 0.85, 
                       verbose: bool = True) -> Dict[str, Any]:
    nan_percentage = data_panel.isna().sum().sum() / (data_panel.shape[0] * data_panel.shape[1])
    
    if verbose:
        print_subsection_header("PANIC Factor Decomposition")
        print(f"Panel contains {nan_percentage:.2%} missing values")

    temp_data = data_panel.copy()
    if nan_percentage > 0:
        temp_data = temp_data.fillna(method='ffill').fillna(method='bfill')
        if temp_data.isna().sum().sum() > 0:
            col_means = temp_data.mean()
            temp_data = temp_data.fillna(col_means)
            if verbose:
                print(f"Filled remaining NaNs with column means")

    X = temp_data.to_numpy().T
    N, T = X.shape

    if verbose:
        print(f"Decomposing panel with N={N} entities, T={T} time periods")

    X_demean = X - np.nanmean(X, axis=1, keepdims=True)

    max_factors = min(25, min(N, T) - 1)
    try:
        n_factors, ic_info = bai_ng_ic(X_demean, max_factors=max_factors, cumulative_var_threshold=cumulative_var_threshold)
        if verbose:
            print(f"Bai-Ng information criteria selected {n_factors} factors using {ic_info['selection_method']}")
            print(f"Information criteria: IC1 suggested {ic_info['optimal_r1']}, IC2 suggested {ic_info['optimal_r2']}")
            print(f"Cumulative variance threshold ({cumulative_var_threshold}) suggested {ic_info.get('var_threshold_r', 'N/A')}")
    except Exception as e:
        n_factors = min(3, min(N, T) // 3)
        if verbose:
            print(f"Error in factor selection: {str(e)}")
            print(f"Falling back to {n_factors} factors")

    try:
        pca = PCA(n_components=n_factors)
        factors = pca.fit_transform(X_demean.T).T
        loadings = pca.components_.T

        explained_var = np.sum(pca.explained_variance_ratio_)
        if verbose:
            print(f"Factors explain {explained_var:.2%} of total variance")

        common_component = np.dot(loadings, factors)

        idiosyncratic_component = X_demean - common_component

        if (np.isnan(common_component).any() or np.isnan(idiosyncratic_component).any()) and verbose:
            print("Warning: Decomposition contains NaN values")
    except Exception as e:
        if verbose:
            print(f"PCA decomposition failed: {str(e)}")
            print("Attempting fallback decomposition method...")

        try:
            X_std = (X_demean - np.nanmean(X_demean, axis=1, keepdims=True)) / (np.nanstd(X_demean, axis=1, keepdims=True) + 1e-10)

            X_std = np.nan_to_num(X_std, nan=0.0)

            C = np.dot(X_std, X_std.T) / T

            eigvals, eigvecs = np.linalg.eigh(C)

            idx = np.argsort(eigvals)[::-1]
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]

            explained_var_ratio = eigvals / np.sum(eigvals)
            cumulative_var = np.cumsum(explained_var_ratio)

            n_factors = np.argmax(cumulative_var >= cumulative_var_threshold) + 1
            n_factors = min(n_factors, 3)

            loadings = eigvecs[:, :n_factors]

            factors = np.dot(loadings.T, X_std)

            common_component = np.dot(loadings, factors)

            idiosyncratic_component = X_demean - common_component

            common_component = np.nan_to_num(common_component, nan=0.0)
            idiosyncratic_component = np.nan_to_num(idiosyncratic_component, nan=0.0)

            explained_var = cumulative_var[n_factors-1]

        except Exception as e:
            X_demean = np.nan_to_num(X_demean, nan=0.0)

            n_factors = 1

            factors = np.mean(X_demean, axis=0, keepdims=True)
            loadings = np.ones((N, 1))

            common_component = np.outer(np.mean(X_demean, axis=1), np.mean(X_demean, axis=0))
            idiosyncratic_component = X_demean - common_component

            var_common = np.var(common_component)
            var_total = np.var(X_demean)
            explained_var = min(var_common / (var_total + 1e-10), 0.5)

    idio_stds = np.std(idiosyncratic_component, axis=1)
    zero_var_count = np.sum(idio_stds < 1e-10)

    result = {
        'common_factors': factors,
        'factor_loadings': loadings,
        'common_component': common_component,
        'idiosyncratic_component': idiosyncratic_component,
        'n_factors': n_factors,
        'explained_variance_ratio': pca.explained_variance_ratio_ if 'pca' in locals() else np.array([explained_var]),
        'data_panel': data_panel,
        'entities': data_panel.columns.tolist(),
        'times': data_panel.index.tolist()
    }

    return result


def extract_common_factors(X: np.ndarray, n_factors: int) -> Tuple[np.ndarray, np.ndarray]:
    X_demean = X - np.mean(X, axis=1, keepdims=True)
    
    pca = PCA(n_components=n_factors)
    factors = pca.fit_transform(X_demean.T).T
    loadings = pca.components_.T
    
    return factors, loadings


def calculate_factor_loadings(X: np.ndarray, factors: np.ndarray) -> np.ndarray:
    N, T = X.shape
    r = factors.shape[0]
    
    loadings = np.zeros((N, r))
    
    for i in range(N):
        y = X[i, :]
        X_reg = factors.T
        
        valid_idx = ~np.isnan(y)
        if np.sum(valid_idx) > r:
            y_valid = y[valid_idx]
            X_valid = X_reg[valid_idx, :]
            
            try:
                loadings[i, :] = np.linalg.lstsq(X_valid, y_valid, rcond=None)[0]
            except:
                loadings[i, :] = np.zeros(r)
    
    return loadings
