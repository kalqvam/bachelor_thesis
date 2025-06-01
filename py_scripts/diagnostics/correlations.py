import pandas as pd
import numpy as np
from scipy import stats

def calculate_correlation_matrices(df, variables, include_pvalues=True):
    missing_vars = [var for var in variables if var not in df.columns]
    if missing_vars:
        print(f"Warning: The following variables are not in the dataframe: {missing_vars}")
        variables = [var for var in variables if var in df.columns]

    if not variables:
        raise ValueError("No valid variables provided")

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

    company_means = df.groupby('ticker')[variables].mean()
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
        temp_means = df.groupby('ticker')[var].mean().reset_index()
        temp_means.columns = ['ticker', f'{var}_mean']
        if company_means_expanded.empty:
            company_means_expanded = temp_means
        else:
            company_means_expanded = company_means_expanded.merge(temp_means, on='ticker', how='outer')

    merged_df = df.merge(company_means_expanded, on='ticker', how='left')

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

    return {
        'overall': {'corr': overall_corr, 'pvals': overall_pvals},
        'between': {'corr': between_corr, 'pvals': between_pvals},
        'within': {'corr': within_corr, 'pvals': within_pvals}
    }

def print_correlation_matrices(correlation_matrices, sig_level=0.05):
    for corr_type, matrices in correlation_matrices.items():
        print(f"\n=== {corr_type.capitalize()} Correlation Matrix ===")
        print(matrices['corr'].round(3))

        if matrices['pvals'] is not None:
            print(f"\n=== {corr_type.capitalize()} P-values ===")
            print(matrices['pvals'].round(3))

            print(f"\n=== {corr_type.capitalize()} Significance ===")
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
        print("\n")

def run_correlation_analysis(df, variables, print_matrices=True, include_pvalues=True, sig_level=0.05):
    print(f"Running correlation analysis for variables: {variables}")

    correlation_matrices = calculate_correlation_matrices(df, variables, include_pvalues)

    if print_matrices:
        print_correlation_matrices(correlation_matrices, sig_level)

    return correlation_matrices
