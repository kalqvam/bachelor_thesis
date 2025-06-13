import numpy as np
from typing import Dict, List, Tuple, Any

from .panel_diagnostics import adf_test


def moving_block_bootstrap(x: np.ndarray, block_size: int, sample_size: int = None) -> np.ndarray:
    x = np.asarray(x)
    if sample_size is None:
        sample_size = len(x)

    valid_indices = ~np.isnan(x)
    valid_count = np.sum(valid_indices)

    if valid_count / len(x) < 0.5:
        return np.full(sample_size, np.nan)

    if valid_count < 3:
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
        indices = np.random.choice(len(valid_x), size=sample_size, replace=True)
        return valid_x[indices]


def unified_bootstrap(decomp_result: Dict[str, Any], n_bootstraps: int = 999, block_size: int = None, trend: bool = False) -> Dict[str, Any]:
    common_factors = decomp_result['common_factors']
    factor_loadings = decomp_result['factor_loadings']
    idiosyncratic = decomp_result['idiosyncratic_component']

    r, T = common_factors.shape
    N = idiosyncratic.shape[0]

    if block_size is None:
        block_size = max(int(np.sqrt(T)), 2)

    factor_adf_stats = []
    factor_adf_results = []
    for i in range(r):
        adf_result = adf_test(common_factors[i, :], trend=trend)
        factor_adf_results.append(adf_result)
        if 'error' not in adf_result:
            factor_adf_stats.append(adf_result['adf_stat'])
        else:
            factor_adf_stats.append(np.nan)

    valid_factor_adfs = [x for x in factor_adf_stats if not np.isnan(x)]

    idio_adf_stats = []
    for i in range(N):
        series = idiosyncratic[i, :]
        if np.isnan(series).all() or len(series) < 5:
            continue

        adf_result = adf_test(series, trend=trend)
        if 'error' not in adf_result:
            idio_adf_stats.append(adf_result['adf_stat'])

    if len(idio_adf_stats) > 0:
        Pb_stat = np.nanmean(idio_adf_stats)
    else:
        Pb_stat = np.nan

    if len(valid_factor_adfs) > 0:
        Pa_stat = np.nanmin(valid_factor_adfs)
    else:
        Pa_stat = np.nan

    if not np.isnan(Pa_stat) and not np.isnan(Pb_stat):
        weight_common = r / (r + 1)
        weight_idio = 1 / (r + 1)
        Pc_stat = weight_common * Pa_stat + weight_idio * Pb_stat
    else:
        Pc_stat = np.nan

    bootstrap_Pa = []
    bootstrap_Pb = []
    bootstrap_Pc = []

    successful_bootstraps = 0
    failed_bootstraps = 0

    for b in range(n_bootstraps):
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
            continue

    if not np.isnan(Pa_stat):
        Pa_p_values = [r.get('p_value', 1.0) for r in factor_adf_results if 'error' not in r]
        Pa_p_value = min(Pa_p_values) if Pa_p_values else np.nan
    else:
        Pa_p_value = np.nan

    if len(bootstrap_Pb) > 0 and not np.isnan(Pb_stat):
        Pb_p_value = np.mean(np.array(bootstrap_Pb) <= Pb_stat)
    else:
        Pb_p_value = np.nan

    if len(bootstrap_Pc) > 0 and not np.isnan(Pc_stat):
        Pc_p_value = np.mean(np.array(bootstrap_Pc) <= Pc_stat)
    else:
        Pc_p_value = np.nan

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


def bootstrap_critical_values(bootstrap_statistics: List[float], alpha_levels: List[float] = [0.01, 0.05, 0.10]) -> Dict[float, float]:
    if not bootstrap_statistics:
        return {alpha: np.nan for alpha in alpha_levels}
    
    bootstrap_array = np.array(bootstrap_statistics)
    critical_values = {}
    
    for alpha in alpha_levels:
        critical_values[alpha] = np.percentile(bootstrap_array, (1 - alpha) * 100)
    
    return critical_values


def bootstrap_pvalues(observed_statistic: float, bootstrap_statistics: List[float]) -> float:
    if np.isnan(observed_statistic) or not bootstrap_statistics:
        return np.nan
    
    bootstrap_array = np.array(bootstrap_statistics)
    
    p_value = np.mean(bootstrap_array <= observed_statistic)
    
    return p_value
