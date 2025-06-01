import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_ljungbox
from scipy import stats
import statsmodels.api as sm
from itertools import product
from sklearn.decomposition import PCA
from scipy.stats import norm

def test_panel_unit_roots(data, variables_to_test=None, n_bootstraps=999, trend=False, block_size=None, cumulative_var_threshold=0.85):
    if variables_to_test is None:
        variables_to_test = data.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['sector_num', 'ESGRiskRating_ordered']
        variables_to_test = [col for col in variables_to_test if col not in exclude_cols]

    results = {}
    tickers = data['ticker'].unique()

    for variable in variables_to_test:
        print(f"\n{'-'*80}\nTesting variable: {variable}\n{'-'*80}")
        var_results = {}

        cd_result = pesaran_cd_test(data, variable)
        var_results['pesaran_cd'] = cd_result
        print(f"Pesaran CD test p-value: {cd_result['p_value']:.4f}")

        pivot_data = data.pivot(index='date', columns='ticker', values=variable)
        pivot_data = pivot_data.sort_index()

        pivot_data = pivot_data.fillna(method='ffill').fillna(method='bfill')

        if pivot_data.isna().sum().sum() / (pivot_data.shape[0] * pivot_data.shape[1]) > 0.2:
            print(f"Warning: Too many missing values in {variable}, skipping PANIC test")
            var_results['panic_test'] = {
                'error': 'Too many missing values'
            }
            results[variable] = var_results
            continue

        spec_type = "with trend" if trend else "without trend"
        print(f"PANIC specification: {spec_type}")

        valid_cols = pivot_data.columns[pivot_data.count() >= 10]
        pivot_data = pivot_data[valid_cols]

        if len(valid_cols) < 5:
            print(f"Warning: Not enough entities with sufficient data for {variable}, skipping PANIC test")
            var_results['panic_test'] = {
                'error': 'Not enough entities with sufficient data'
            }
            results[variable] = var_results
            continue

        N, T = pivot_data.shape[1], pivot_data.shape[0]

        if block_size is None:
            block_size = max(int(np.sqrt(T)), 2)
            print(f"Auto-selected block size: {block_size}")

        print(f"Using unified bootstrap with {n_bootstraps} replications.")
        print(f"Using cumulative variance threshold of {cumulative_var_threshold} for factor selection fallback.")

        panic_result = panic_test(
            pivot_data,
            n_bootstraps=n_bootstraps,
            trend=trend,
            block_size=block_size,
            cumulative_var_threshold=cumulative_var_threshold
        )

        print(f"PANIC results:")
        print(f"  Common components test (Pa): {panic_result['Pa_stat']:.4f}, bootstrap p-value: {panic_result['Pa_p_value']:.4f}")
        print(f"  Idiosyncratic components test (Pb): {panic_result['Pb_stat']:.4f}, bootstrap p-value: {panic_result['Pb_p_value']:.4f}")
        print(f"  Pooled test (Pc): {panic_result['Pc_stat']:.4f}, bootstrap p-value: {panic_result['Pc_p_value']:.4f}")

        var_results['panic_test'] = panic_result

        results[variable] = var_results

    return results

def pesaran_cd_test(data, variable):
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

def bai_ng_ic(X, max_factors=25, cumulative_var_threshold=0.85):
    N, T = X.shape

    print(f"Computing Bai-Ng criteria with max {max_factors} factors (N={N}, T={T})")

    has_nans = np.isnan(X).any()
    if has_nans:
        print("Warning: Data contains NaN values, replacing with zeros for factor selection")
        X = np.nan_to_num(X, nan=0.0)

    try:
        X_demean = X - np.mean(X, axis=1, keepdims=True)
    except Exception as e:
        print(f"Error in demeaning: {str(e)}")
        X_demean = X

    actual_max = min(max_factors, min(N, T) - 1)
    print(f"Testing up to {actual_max} factors")

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
            print(f"Cumulative variance threshold {cumulative_var_threshold} reached at r={var_threshold_r}")

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

                print(f"  r={r}: V_r={V_r:.4f}, IC1={ic1_value:.4f}, IC2={ic2_value:.4f}, cum_var={cum_var_ratio[r-1]:.4f}")
        else:
            pca = PCA(n_components=actual_max)
            pca.fit(X_demean.T)

            cum_var_ratio = np.cumsum(pca.explained_variance_ratio_)
            var_threshold_r = np.argmax(cum_var_ratio >= cumulative_var_threshold) + 1
            var_threshold_r = min(var_threshold_r, actual_max)
            print(f"Cumulative variance threshold {cumulative_var_threshold} reached at r={var_threshold_r}")

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

                print(f"  r={r}: V_r={V_r:.4f}, IC1={ic1_value:.4f}, IC2={ic2_value:.4f}, cum_var={cum_var_ratio[r-1]:.4f}")

        optimal_r1 = np.nanargmin(IC1) + 1 if not np.all(np.isnan(IC1)) else 1
        optimal_r2 = np.nanargmin(IC2) + 1 if not np.all(np.isnan(IC2)) else 1

        max_reasonable = min(N, T) - 3

        optimal_r = optimal_r1
        selection_method = "IC1"

        if optimal_r1 == actual_max and optimal_r2 < actual_max:
            optimal_r = optimal_r2
            selection_method = "IC2"
            print(f"ICp1 selected maximum factors, using ICp2 selection instead: {optimal_r}")
        elif optimal_r1 == actual_max and optimal_r2 == actual_max:
            optimal_r = var_threshold_r
            selection_method = "Cumulative variance threshold"
            print(f"Both IC1 and IC2 selected maximum factors, using cumulative variance threshold ({cumulative_var_threshold}): {optimal_r}")

        if optimal_r > max_reasonable and optimal_r2 > max_reasonable:
            old_r = optimal_r
            optimal_r = var_threshold_r
            selection_method = "Cumulative variance threshold (after IC exceeded max reasonable)"
            print(f"Both IC1 ({optimal_r1}) and IC2 ({optimal_r2}) exceeded max reasonable factors ({max_reasonable}), using cumulative variance threshold: {optimal_r}")

            if optimal_r > max_reasonable:
                print(f"Cumulative variance threshold also exceeded max reasonable factors, capping at {max_reasonable}")
                optimal_r = max_reasonable
                selection_method = "Max reasonable (fallback)"
        elif optimal_r > max_reasonable:
            old_r = optimal_r
            optimal_r = max_reasonable
            print(f"Selected factors ({old_r}) > max reasonable factors, capping at {max_reasonable}")

        if optimal_r > max_reasonable:
            old_r = optimal_r
            optimal_r = max_reasonable
            print(f"Selected factors ({old_r}) > 1/3 of min(N,T), capping at {max_reasonable}")

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
        print(f"Error in Bai-Ng criteria calculation: {str(e)}")
        optimal_r = min(3, min(N, T) // 5)
        print(f"Using fallback factor count: {optimal_r}")
        return optimal_r, {
            'IC1': [],
            'IC2': [],
            'optimal_r1': optimal_r,
            'optimal_r2': optimal_r,
            'var_threshold_r': optimal_r,
            'cum_var_ratio': [],
            'selection_method': "Error fallback"
        }

def panic_decomposition(data_panel, cumulative_var_threshold=0.85):
    nan_percentage = data_panel.isna().sum().sum() / (data_panel.shape[0] * data_panel.shape[1])
    print(f"Panel contains {nan_percentage:.2%} missing values")

    temp_data = data_panel.copy()
    if nan_percentage > 0:
        temp_data = temp_data.fillna(method='ffill').fillna(method='bfill')
        if temp_data.isna().sum().sum() > 0:
            col_means = temp_data.mean()
            temp_data = temp_data.fillna(col_means)
            print(f"Filled remaining NaNs with column means")

    X = temp_data.to_numpy().T
    N, T = X.shape

    print(f"Decomposing panel with N={N} entities, T={T} time periods")

    X_demean = X - np.nanmean(X, axis=1, keepdims=True)

    max_factors = min(25, min(N, T) - 1)
    try:
        n_factors, ic_info = bai_ng_ic(X_demean, max_factors=max_factors, cumulative_var_threshold=cumulative_var_threshold)
        print(f"Bai-Ng information criteria selected {n_factors} factors using {ic_info['selection_method']}")
        print(f"Information criteria: IC1 suggested {ic_info['optimal_r1']}, IC2 suggested {ic_info['optimal_r2']}")
        print(f"Cumulative variance threshold ({cumulative_var_threshold}) suggested {ic_info.get('var_threshold_r', 'N/A')}")
    except Exception as e:
        print(f"Error in factor selection: {str(e)}")
        n_factors = min(3, min(N, T) // 3)
        print(f"Falling back to {n_factors} factors")

    try:
        pca = PCA(n_components=n_factors)
        factors = pca.fit_transform(X_demean.T).T
        loadings = pca.components_.T

        explained_var = np.sum(pca.explained_variance_ratio_)
        print(f"Factors explain {explained_var:.2%} of total variance")

        common_component = np.dot(loadings, factors)

        idiosyncratic_component = X_demean - common_component

        if (np.isnan(common_component).any() or np.isnan(idiosyncratic_component).any()):
            print("Warning: Decomposition contains NaN values")
    except Exception as e:
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

            print(f"Fallback method selected {n_factors} factors explaining {cumulative_var[n_factors-1]:.2%} variance")

            loadings = eigvecs[:, :n_factors]

            factors = np.dot(loadings.T, X_std)

            common_component = np.dot(loadings, factors)

            idiosyncratic_component = X_demean - common_component

            common_component = np.nan_to_num(common_component, nan=0.0)
            idiosyncratic_component = np.nan_to_num(idiosyncratic_component, nan=0.0)

            explained_var = cumulative_var[n_factors-1]

        except Exception as e:
            print(f"Fallback decomposition also failed: {str(e)}")
            print("Using last resort decomposition (not theoretically sound)")

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
    if zero_var_count > 0:
        print(f"Warning: {zero_var_count} entities have near-zero variance in idiosyncratic component")

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

def adf_test(x, trend=False, max_lags=None):
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
                print(f"Lag {lag} failed: {str(e)}")
                continue

        if 'adf_stat' not in result:
            return {'adf_stat': np.nan, 'p_value': np.nan, 'error': 'Regression failed for all lags'}

        return result
    except Exception as e:
        return {'adf_stat': np.nan, 'p_value': np.nan, 'error': f'Exception in ADF test: {str(e)}'}

def panel_adf_test(data_panel, trend=False):
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

def moving_block_bootstrap(x, block_size, sample_size=None):
    x = np.asarray(x)
    if sample_size is None:
        sample_size = len(x)

    valid_indices = ~np.isnan(x)
    valid_count = np.sum(valid_indices)

    if valid_count / len(x) < 0.5:
        print(f"Warning: Series contains {len(x) - valid_count}/{len(x)} NaN values")

    if valid_count < 3:
        print(f"Warning: Not enough valid observations for bootstrap ({valid_count})")
        return np.full(sample_size, np.nan)

    valid_x = x[valid_indices]

    if len(valid_x) <= block_size:
        if len(valid_x) <= 1:
            return np.full(sample_size, np.nan)
        else:
            indices = np.random.choice(len(valid_x), size=sample_size, replace=True)
            return valid_x[indices]

    try:
        n_blocks = int(np.ceil(sample_size / block_size))

        max_start = len(valid_x) - block_size + 1
        block_starts = np.random.choice(max_start, size=n_blocks, replace=True)

        indices = np.concatenate([np.arange(start, start + block_size) for start in block_starts])

        indices = indices[:sample_size]

        indices = np.clip(indices, 0, len(valid_x) - 1)

        return valid_x[indices]
    except Exception as e:
        print(f"Error in moving block bootstrap: {str(e)}")
        indices = np.random.choice(len(valid_x), size=sample_size, replace=True)
        return valid_x[indices]

def unified_bootstrap(decomp_result, n_bootstraps=999, block_size=None, trend=False):
    common_factors = decomp_result['common_factors']
    factor_loadings = decomp_result['factor_loadings']
    idiosyncratic = decomp_result['idiosyncratic_component']

    r, T = common_factors.shape
    N = idiosyncratic.shape[0]

    print(f"Starting bootstrap with r={r} factors, N={N} entities, T={T} time periods")

    if block_size is None:
        block_size = max(int(np.sqrt(T)), 2)

    print("Testing original common factors...")
    factor_adf_stats = []
    factor_adf_results = []
    for i in range(r):
        adf_result = adf_test(common_factors[i, :], trend=trend)
        factor_adf_results.append(adf_result)
        if 'error' not in adf_result:
            factor_adf_stats.append(adf_result['adf_stat'])
        else:
            print(f"  Warning: Factor {i} ADF test failed: {adf_result.get('error', 'Unknown error')}")
            factor_adf_stats.append(np.nan)

    valid_factor_adfs = [x for x in factor_adf_stats if not np.isnan(x)]
    print(f"Found {len(valid_factor_adfs)}/{r} valid factor ADF statistics")

    print("Testing idiosyncratic components...")
    idio_adf_stats = []
    for i in range(N):
        series = idiosyncratic[i, :]
        if np.isnan(series).all() or len(series) < 5:
            continue

        adf_result = adf_test(series, trend=trend)
        if 'error' not in adf_result:
            idio_adf_stats.append(adf_result['adf_stat'])

    print(f"Found {len(idio_adf_stats)}/{N} valid idiosyncratic component ADF statistics")

    if len(idio_adf_stats) > 0:
        Pb_stat = np.nanmean(idio_adf_stats)
        print(f"Pb statistic (mean of idiosyncratic ADFs): {Pb_stat:.4f}")
    else:
        Pb_stat = np.nan
        print("Warning: No valid idiosyncratic ADF statistics, Pb_stat is NaN")

    if len(valid_factor_adfs) > 0:
        Pa_stat = np.nanmin(valid_factor_adfs)
        print(f"Pa statistic (min of factor ADFs): {Pa_stat:.4f}")
    else:
        Pa_stat = np.nan
        print("Warning: No valid factor ADF statistics, Pa_stat is NaN")

    if not np.isnan(Pa_stat) and not np.isnan(Pb_stat):
        weight_common = r / (r + 1)
        weight_idio = 1 / (r + 1)
        Pc_stat = weight_common * Pa_stat + weight_idio * Pb_stat
        print(f"Pc statistic (weighted average): {Pc_stat:.4f}")
    else:
        Pc_stat = np.nan
        print("Warning: Cannot calculate Pc statistic due to NaN in Pa or Pb")

    bootstrap_Pa = []
    bootstrap_Pb = []
    bootstrap_Pc = []

    successful_bootstraps = 0
    failed_bootstraps = 0

    print(f"Performing {n_bootstraps} bootstrap iterations...")
    for b in range(n_bootstraps):
        if b > 0 and b % 100 == 0:
            print(f"  Completed {b} bootstrap iterations ({successful_bootstraps} successful)")

        try:
            boot_idiosyncratic = np.zeros_like(idiosyncratic)
            for i in range(N):
                if np.isnan(idiosyncratic[i, :]).all():
                    boot_idiosyncratic[i, :] = np.nan
                    continue

                valid_series = idiosyncratic[i, :]
                valid_indices = ~np.isnan(valid_series)
                if np.sum(valid_indices) < 5:
                    boot_idiosyncratic[i, :] = np.nan
                    continue

                valid_values = valid_series[valid_indices]

                boot_values = moving_block_bootstrap(
                    valid_values, block_size, sample_size=len(valid_values)
                )

                boot_series = np.full_like(valid_series, np.nan)
                boot_series[valid_indices] = boot_values
                boot_idiosyncratic[i, :] = boot_series

            boot_idio_adf_stats = []
            for i in range(N):
                series = boot_idiosyncratic[i, :]
                if np.isnan(series).all() or np.sum(~np.isnan(series)) < 5:
                    continue

                boot_adf_result = adf_test(series, trend=trend)
                if 'error' not in boot_adf_result:
                    boot_idio_adf_stats.append(boot_adf_result['adf_stat'])

            boot_factor_adf_stats = factor_adf_stats.copy()

            if len(boot_idio_adf_stats) > 0:
                boot_Pb = np.nanmean(boot_idio_adf_stats)
            else:
                boot_Pb = np.nan
                continue

            boot_Pa = Pa_stat

            if not np.isnan(boot_Pa) and not np.isnan(boot_Pb):
                boot_Pc = weight_common * boot_Pa + weight_idio * boot_Pb
            else:
                boot_Pc = np.nan
                continue

            if not np.isnan(boot_Pa):
                bootstrap_Pa.append(boot_Pa)
            if not np.isnan(boot_Pb):
                bootstrap_Pb.append(boot_Pb)
            if not np.isnan(boot_Pc):
                bootstrap_Pc.append(boot_Pc)

            successful_bootstraps += 1

        except Exception as e:
            failed_bootstraps += 1
            if failed_bootstraps <= 5:
                print(f"  Bootstrap iteration {b} failed: {str(e)}")
            continue

    print(f"Bootstrap complete: {successful_bootstraps} successful, {failed_bootstraps} failed")

    if not np.isnan(Pa_stat):
        Pa_p_values = [r.get('p_value', 1.0) for r in factor_adf_results if 'error' not in r]
        Pa_p_value = min(Pa_p_values) if Pa_p_values else np.nan
        print(f"Pa p-value (from original tests): {Pa_p_value:.4f}")
    else:
        Pa_p_value = np.nan

    if len(bootstrap_Pb) > 0 and not np.isnan(Pb_stat):
        Pb_p_value = np.mean(np.array(bootstrap_Pb) <= Pb_stat)
        print(f"Pb p-value (from {len(bootstrap_Pb)} bootstrap samples): {Pb_p_value:.4f}")
    else:
        Pb_p_value = np.nan
        print("Warning: Cannot calculate Pb p-value (no valid bootstrap samples)")

    if len(bootstrap_Pc) > 0 and not np.isnan(Pc_stat):
        Pc_p_value = np.mean(np.array(bootstrap_Pc) <= Pc_stat)
        print(f"Pc p-value (from {len(bootstrap_Pc)} bootstrap samples): {Pc_p_value:.4f}")
    else:
        Pc_p_value = np.nan
        print("Warning: Cannot calculate Pc p-value (no valid bootstrap samples)")

    return {
        'Pa_stat': Pa_stat,
        'Pb_stat': Pb_stat,
        'Pc_stat': Pc_stat,
        'Pa_p_value': Pa_p_value,
        'Pb_p_value': Pb_p_value,
        'Pc_p_value': Pc_p_value,
        'bootstrap_Pa': [Pa_stat] if not np.isnan(Pa_stat) else [],
        'bootstrap_Pb': bootstrap_Pb,
        'bootstrap_Pc': bootstrap_Pc,
        'bootstrap_samples': successful_bootstraps,
        'total_attempted': n_bootstraps,
        'failed_bootstraps': failed_bootstraps,
        'r': r,
        'N': N,
        'T': T,
        'block_size': block_size
    }

def check_panel_stationarity(data, variables=None, significance=0.05, trend=True, n_bootstraps=199, cumulative_var_threshold=0.85):
    results = test_panel_unit_roots(
        data=data,
        variables_to_test=variables,
        n_bootstraps=n_bootstraps,
        trend=trend,
        cumulative_var_threshold=cumulative_var_threshold
    )

    summary = {}

    for variable, result in results.items():
        panic_result = result.get('panic_test', {})

        if 'error' in panic_result:
            summary[variable] = {
                'stationary': None,
                'error': panic_result['error']
            }
            continue

        p_common = panic_result.get('Pa_p_value', 1)
        p_idio = panic_result.get('Pb_p_value', 1)
        p_pooled = panic_result.get('Pc_p_value', 1)

        common_stationary = p_common < significance if not np.isnan(p_common) else None
        idio_stationary = p_idio < significance if not np.isnan(p_idio) else None
        pooled_stationary = p_pooled < significance if not np.isnan(p_pooled) else None

        summary[variable] = {
            'common_stationary': common_stationary,
            'idiosyncratic_stationary': idio_stationary,
            'pooled_stationary': pooled_stationary,
            'n_factors': panic_result.get('n_factors'),
            'Pa_stat': panic_result.get('Pa_stat'),
            'Pa_p_value': p_common,
            'Pb_stat': panic_result.get('Pb_stat'),
            'Pb_p_value': p_idio,
            'Pc_stat': panic_result.get('Pc_stat'),
            'Pc_p_value': p_pooled
        }

    return summary

def panic_test(data_panel, n_bootstraps=199, trend=False, block_size=None, cumulative_var_threshold=0.85):
    print("Decomposing panel data into common factors and idiosyncratic components...")
    decomp_result = panic_decomposition(data_panel, cumulative_var_threshold=cumulative_var_threshold)

    print(f"Running unified bootstrap with fixed factor structure ({n_bootstraps} replications)...")
    bootstrap_result = unified_bootstrap(
        decomp_result,
        n_bootstraps=n_bootstraps,
        block_size=block_size,
        trend=trend
    )

    result = {
        'Pa_stat': bootstrap_result['Pa_stat'],
        'Pb_stat': bootstrap_result['Pb_stat'],
        'Pc_stat': bootstrap_result['Pc_stat'],
        'Pa_p_value': bootstrap_result['Pa_p_value'],
        'Pb_p_value': bootstrap_result['Pb_p_value'],
        'Pc_p_value': bootstrap_result['Pc_p_value'],

        'bootstrap_samples': bootstrap_result['bootstrap_samples'],
        'total_attempted': bootstrap_result['total_attempted'],

        'n_factors': decomp_result['n_factors'],
        'N': bootstrap_result['N'],
        'T': bootstrap_result['T'],
        'trend': trend,
        'block_size': bootstrap_result['block_size'],

        'explained_variance_ratio': decomp_result['explained_variance_ratio'].tolist(),

        'method_details': {
            'bootstrap_approach': 'Fixed factor structure with idiosyncratic resampling',
            'inference': 'Bootstrap-based (no distributional assumptions)',
            'decomposition': 'PANIC (Bai & Ng, 2004, 2010)',
            'factor_selection': 'Bai-Ng information criteria with cumulative variance threshold fallback',
            'cumulative_var_threshold': cumulative_var_threshold
        }
    }

    return result

def main():
    file_path = 'raw_ratios.csv'
    data = pd.read_csv(file_path)

    dupes = data.duplicated(subset=['ticker', 'date'])
    if dupes.any():
        print(f"Warning: Found {dupes.sum()} duplicate ticker-date combinations. Removing duplicates.")
        data = data.drop_duplicates(subset=['ticker', 'date'])

    variables_to_test = [
        'cashAndCashEquivalents_to_totalAssets_ratio',
        'ln_totalAssets',
        'totalDebt_to_totalAssets_ratio',
        'ebitda_to_revenue_ratio'
    ]

    use_trend = True
    n_bootstraps = 449
    block_size = None
    cumulative_var_threshold = 0.95

    trend_desc = "with trend" if use_trend else "without trend"
    print(f"Running PANIC panel unit root tests {trend_desc}")
    print(f"Using {n_bootstraps} bootstrap replications")
    print(f"Using cumulative variance threshold of {cumulative_var_threshold} for factor selection fallback")

    results = test_panel_unit_roots(
        data=data,
        variables_to_test=variables_to_test,
        n_bootstraps=n_bootstraps,
        trend=use_trend,
        block_size=block_size,
        cumulative_var_threshold=cumulative_var_threshold
    )

    print("\n" + "="*80)
    print("SUMMARY OF PANEL UNIT ROOT TESTS")
    print("="*80)

    for variable, result in results.items():
        print(f"\nVariable: {variable}")

        cd_result = result.get('pesaran_cd', {})
        print(f"Pesaran CD test: stat={cd_result.get('cd_stat', 'N/A'):.4f}, p-value={cd_result.get('p_value', 'N/A'):.4f}")

        panic_result = result.get('panic_test', {})

        if 'error' in panic_result:
            print(f"PANIC test failed: {panic_result['error']}")
            continue

        print(f"Common factor unit root test (Pa):")
        print(f"  Statistic: {panic_result.get('Pa_stat', 'N/A'):.4f}")
        print(f"  Bootstrap p-value: {panic_result.get('Pa_p_value', 'N/A'):.4f}")
        print(f"  Interpretation: {'Non-stationary common factors' if panic_result.get('Pa_p_value', 1) > 0.05 else 'Stationary common factors'}")

        print(f"Idiosyncratic component unit root test (Pb):")
        print(f"  Statistic: {panic_result.get('Pb_stat', 'N/A'):.4f}")
        print(f"  Bootstrap p-value: {panic_result.get('Pb_p_value', 'N/A'):.4f}")
        print(f"  Interpretation: {'Non-stationary idiosyncratic components' if panic_result.get('Pb_p_value', 1) > 0.05 else 'Stationary idiosyncratic components'}")

        print(f"Pooled unit root test (Pc):")
        print(f"  Statistic: {panic_result.get('Pc_stat', 'N/A'):.4f}")
        print(f"  Bootstrap p-value: {panic_result.get('Pc_p_value', 'N/A'):.4f}")
        print(f"  Interpretation: {'Non-stationary panel data' if panic_result.get('Pc_p_value', 1) > 0.05 else 'Stationary panel data'}")

        print(f"Number of factors: {panic_result.get('n_factors', 'N/A')}")
        print(f"Explained variance by factors: {sum(panic_result.get('explained_variance_ratio', [0])):.2%}")
        print(f"N = {panic_result.get('N', 'N/A')}, T = {panic_result.get('T', 'N/A')}")
        print(f"Bootstrap samples: {panic_result.get('bootstrap_samples', 0)}/{panic_result.get('total_attempted', 0)}")

        print("-"*80)
