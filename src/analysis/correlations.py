import pandas as pd
import numpy as np
from scipy import stats
from typing import List, Dict, Optional, Tuple

from ..utils import (
    DEFAULT_TICKER_COLUMN, print_subsection_header, print_section_header,
    format_number, print_correlation_matrix
)


def calculate_correlation_matrices(df: pd.DataFrame, 
                                 variables: List[str], 
                                 ticker_column: str = DEFAULT_TICKER_COLUMN,
                                 include_pvalues: bool = True,
                                 verbose: bool = True) -> Dict[str, Dict]:
    
    if verbose:
        print_subsection_header("Calculating Correlation Matrices")
    
    missing_vars = [var for var in variables if var not in df.columns]
    if missing_vars:
        if verbose:
            print(f"Warning: The following variables are not in the dataframe: {missing_vars}")
        variables = [var for var in variables if var in df.columns]

    if not variables:
        raise ValueError("No valid variables provided")

    if verbose:
        print(f"Analyzing correlations for {len(variables)} variables")

    overall_corr = df[variables].corr()
    overall_pvals = None

    if include_pvalues:
        overall_pvals = pd.DataFrame(np.zeros((len(variables), len(variables))),
                                  columns=variables, index=variables)
        for i, var1 in enumerate(variables):
            for j, var2 in enumerate(variables):
                if i != j: 
                    corr, p_val = stats.pearsonr(df[var1].dropna(), df[var2].dropna())
                    overall_pvals.iloc[i, j] = p_val

    company_means = df.groupby(ticker_column)[variables].mean()
    between_corr = company_means.corr()
    between_pvals = None

    if include_pvalues:
        between_pvals = pd.DataFrame(np.zeros((len(variables), len(variables))),
                                  columns=variables, index=variables)
        for i, var1 in enumerate(variables):
            for j, var2 in enumerate(variables):
                if i != j:
                    corr, p_val = stats.pearsonr(company_means[var1].dropna(), company_means[var2].dropna())
                    between_pvals.iloc[i, j] = p_val

    company_means_expanded = pd.DataFrame()
    for var in variables:
        temp_means = df.groupby(ticker_column)[var].mean().reset_index()
        temp_means.columns = [ticker_column, f'{var}_mean']
        if company_means_expanded.empty:
            company_means_expanded = temp_means
        else:
            company_means_expanded = company_means_expanded.merge(temp_means, on=ticker_column, how='outer')

    merged_df = df.merge(company_means_expanded, on=ticker_column, how='left')

    within_df = pd.DataFrame()
    overall_means = df[variables].mean()

    for var in variables:
        within_df[var] = df[var] - merged_df[f'{var}_mean'] + overall_means[var]

    within_corr = within_df.corr()
    within_pvals = None

    if include_pvalues:
        within_pvals = pd.DataFrame(np.zeros((len(variables), len(variables))),
                                 columns=variables, index=variables)
        for i, var1 in enumerate(variables):
            for j, var2 in enumerate(variables):
                if i != j:
                    corr, p_val = stats.pearsonr(within_df[var1].dropna(), within_df[var2].dropna())
                    within_pvals.iloc[i, j] = p_val

    results = {
        'overall': {'corr': overall_corr, 'pvals': overall_pvals},
        'between': {'corr': between_corr, 'pvals': between_pvals},
        'within': {'corr': within_corr, 'pvals': within_pvals}
    }

    if verbose:
        print(f"Correlation analysis completed for {len(variables)} variables")
        print(f"Generated overall, between-company, and within-company correlation matrices")

    return results


def print_correlation_matrices(correlation_matrices: Dict[str, Dict], 
                             sig_level: float = 0.05,
                             verbose: bool = True) -> None:
    
    if verbose:
        print_section_header("Correlation Analysis Results")
    
    for corr_type, matrices in correlation_matrices.items():
        if verbose:
            print_subsection_header(f"{corr_type.capitalize()} Correlation Matrix")
        
        print_correlation_matrix(matrices['corr'], f"{corr_type.capitalize()} Correlations")

        if matrices['pvals'] is not None:
            if verbose:
                print(f"\n{corr_type.capitalize()} P-values:")
            print(matrices['pvals'].round(3))

            if verbose:
                print(f"\n{corr_type.capitalize()} Significance (* p<0.1, ** p<0.05, *** p<0.01):")
            
            significance = pd.DataFrame('', index=matrices['corr'].index, columns=matrices['corr'].columns)

            for i in range(len(matrices['pvals'])):
                for j in range(len(matrices['pvals'])):
                    if i != j:
                        if matrices['pvals'].iloc[i, j] <= 0.01:
                            significance.iloc[i, j] = '***'
                        elif matrices['pvals'].iloc[i, j] <= 0.05:
                            significance.iloc[i, j] = '**'
                        elif matrices['pvals'].iloc[i, j] <= 0.1:
                            significance.iloc[i, j] = '*'

            print(significance)
        
        if verbose:
            print()


def analyze_correlation_patterns(correlation_matrices: Dict[str, Dict],
                               threshold: float = 0.7,
                               verbose: bool = True) -> Dict[str, Dict]:
    
    if verbose:
        print_subsection_header("Correlation Pattern Analysis")
    
    patterns = {}
    
    for corr_type, matrices in correlation_matrices.items():
        corr_matrix = matrices['corr']
        p_values = matrices['pvals']
        
        high_correlations = []
        significant_correlations = []
        
        for i in range(len(corr_matrix)):
            for j in range(i+1, len(corr_matrix)):
                var1 = corr_matrix.index[i]
                var2 = corr_matrix.columns[j]
                corr_val = corr_matrix.iloc[i, j]
                
                if abs(corr_val) >= threshold:
                    high_correlations.append({
                        'var1': var1,
                        'var2': var2,
                        'correlation': corr_val
                    })
                
                if p_values is not None:
                    p_val = p_values.iloc[i, j]
                    if p_val <= 0.05:
                        significant_correlations.append({
                            'var1': var1,
                            'var2': var2,
                            'correlation': corr_val,
                            'p_value': p_val
                        })
        
        patterns[corr_type] = {
            'high_correlations': high_correlations,
            'significant_correlations': significant_correlations,
            'avg_abs_correlation': abs(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]).mean()
        }
        
        if verbose:
            print(f"{corr_type.capitalize()} patterns:")
            print(f"  High correlations (|r| >= {threshold}): {len(high_correlations)}")
            print(f"  Significant correlations (p <= 0.05): {len(significant_correlations)}")
            print(f"  Average absolute correlation: {patterns[corr_type]['avg_abs_correlation']:.3f}")
    
    return patterns


def run_correlation_analysis(df: pd.DataFrame, 
                           variables: List[str],
                           ticker_column: str = DEFAULT_TICKER_COLUMN,
                           print_matrices: bool = True, 
                           include_pvalues: bool = True, 
                           sig_level: float = 0.05,
                           analyze_patterns: bool = True,
                           pattern_threshold: float = 0.7,
                           verbose: bool = True) -> Tuple[Dict[str, Dict], Optional[Dict[str, Dict]]]:
    
    if verbose:
        print_section_header("Panel Correlation Analysis")
        print(f"Running correlation analysis for variables: {variables}")

    correlation_matrices = calculate_correlation_matrices(
        df, variables, ticker_column, include_pvalues, verbose
    )

    if print_matrices:
        print_correlation_matrices(correlation_matrices, sig_level, verbose)

    patterns = None
    if analyze_patterns:
        patterns = analyze_correlation_patterns(
            correlation_matrices, pattern_threshold, verbose
        )

    return correlation_matrices, patterns
