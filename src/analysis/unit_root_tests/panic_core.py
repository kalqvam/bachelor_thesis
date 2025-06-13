import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any

from .panel_diagnostics import pesaran_cd_test, validate_panel_data, prepare_panel_data
from .decomposition import panic_decomposition
from .bootstrap import unified_bootstrap
from ...utils import print_section_header, print_subsection_header, format_number


def panic_test(data_panel: pd.DataFrame, n_bootstraps: int = 199, trend: bool = False, 
               block_size: Optional[int] = None, cumulative_var_threshold: float = 0.85) -> Dict[str, Any]:
    
    decomp_result = panic_decomposition(data_panel, cumulative_var_threshold=cumulative_var_threshold)
    
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


def test_panel_unit_roots(data: pd.DataFrame, variables_to_test: Optional[List[str]] = None, 
                         n_bootstraps: int = 999, trend: bool = False, block_size: Optional[int] = None, 
                         cumulative_var_threshold: float = 0.85) -> Dict[str, Any]:
    
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
        
        pivot_data, prep_stats = prepare_panel_data(data, variable)
        
        if pivot_data is None:
            print(f"Warning: Cannot prepare data for {variable}: {prep_stats.get('error', 'Unknown error')}")
            var_results['panic_test'] = {
                'error': prep_stats.get('error', 'Data preparation failed')
            }
            results[variable] = var_results
            continue
        
        N, T = prep_stats['N'], prep_stats['T']
        
        if block_size is None:
            auto_block_size = max(int(np.sqrt(T)), 2)
            print(f"Auto-selected block size: {auto_block_size}")
        else:
            auto_block_size = block_size
        
        spec_type = "with trend" if trend else "without trend"
        print(f"PANIC specification: {spec_type}")
        print(f"Using unified bootstrap with {n_bootstraps} replications.")
        print(f"Using cumulative variance threshold of {cumulative_var_threshold} for factor selection fallback.")
        
        panic_result = panic_test(
            pivot_data,
            n_bootstraps=n_bootstraps,
            trend=trend,
            block_size=auto_block_size,
            cumulative_var_threshold=cumulative_var_threshold
        )
        
        print(f"PANIC results:")
        print(f"  Common components test (Pa): {panic_result['Pa_stat']:.4f}, bootstrap p-value: {panic_result['Pa_p_value']:.4f}")
        print(f"  Idiosyncratic components test (Pb): {panic_result['Pb_stat']:.4f}, bootstrap p-value: {panic_result['Pb_p_value']:.4f}")
        print(f"  Pooled test (Pc): {panic_result['Pc_stat']:.4f}, bootstrap p-value: {panic_result['Pc_p_value']:.4f}")
        
        var_results['panic_test'] = panic_result
        results[variable] = var_results
    
    return results


def check_panel_stationarity(data: pd.DataFrame, variables: Optional[List[str]] = None, 
                            significance: float = 0.05, trend: bool = True, n_bootstraps: int = 199, 
                            cumulative_var_threshold: float = 0.85) -> Dict[str, Any]:
    
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


def print_panic_results(results: Dict[str, Any], test_name: str = "Panel Unit Root Tests") -> None:
    print_section_header(test_name)
    
    for variable, result in results.items():
        print(f"\nVariable: {variable}")
        
        cd_result = result.get('pesaran_cd', {})
        if cd_result:
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


def format_panic_summary(results: Dict[str, Any]) -> str:
    summary_lines = []
    
    summary_lines.append("PANIC Panel Unit Root Test Summary")
    summary_lines.append("=" * 50)
    
    for variable, result in results.items():
        panic_result = result.get('panic_test', {})
        
        if 'error' in panic_result:
            summary_lines.append(f"{variable}: ERROR - {panic_result['error']}")
            continue
        
        pa_p = panic_result.get('Pa_p_value', np.nan)
        pb_p = panic_result.get('Pb_p_value', np.nan)
        pc_p = panic_result.get('Pc_p_value', np.nan)
        
        common_stat = "Stationary" if pa_p < 0.05 else "Non-stationary"
        idio_stat = "Stationary" if pb_p < 0.05 else "Non-stationary"
        pooled_stat = "Stationary" if pc_p < 0.05 else "Non-stationary"
        
        summary_lines.append(f"{variable}:")
        summary_lines.append(f"  Common factors: {common_stat} (p={pa_p:.3f})")
        summary_lines.append(f"  Idiosyncratic: {idio_stat} (p={pb_p:.3f})")
        summary_lines.append(f"  Pooled: {pooled_stat} (p={pc_p:.3f})")
        summary_lines.append(f"  Factors: {panic_result.get('n_factors', 'N/A')}")
        summary_lines.append("")
    
    return "\n".join(summary_lines)
