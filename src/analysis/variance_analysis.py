import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple

from ..utils import (
    DEFAULT_TICKER_COLUMN, DEFAULT_VARIANCE_THRESHOLD, print_subsection_header,
    print_section_header, format_number, print_processing_stats
)


def analyze_panel_variables(df: pd.DataFrame, 
                          columns: List[str], 
                          id_column: str = DEFAULT_TICKER_COLUMN, 
                          variance_threshold: float = DEFAULT_VARIANCE_THRESHOLD,
                          verbose: bool = True) -> Dict[str, Dict]:
    
    if verbose:
        print_subsection_header("Panel Variable Analysis")
    
    results = {}

    valid_columns = [col for col in columns if col in df.columns and col != id_column]
    if not valid_columns:
        if verbose:
            print(f"None of the provided columns {columns} exist in the dataframe")
        return {}

    companies = df[id_column].unique()
    n_companies = len(companies)

    if verbose:
        print(f"Analyzing {len(valid_columns)} variables across {format_number(n_companies)} companies")
        print(f"Variance threshold: {variance_threshold}")

    for column in valid_columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            if verbose:
                print(f"Skipping non-numeric column: {column}")
            continue

        if verbose:
            print(f"Processing variable: {column}")

        variable_results = {
            'overall_stats': {},
            'company_level': {},
            'variance_analysis': {},
            'cv_percentiles': {},
            'near_zero_variance_companies': []
        }

        overall_data = df[column].dropna()
        variable_results['overall_stats'] = {
            'mean': overall_data.mean(),
            'median': overall_data.median(),
            'std_dev': overall_data.std(),
            'min': overall_data.min(),
            'max': overall_data.max(),
            'cv': overall_data.std() / overall_data.mean() if overall_data.mean() != 0 else np.nan,
            'count': len(overall_data),
            'missing': df[column].isna().sum()
        }

        company_stats = []
        company_variances = []
        company_cvs = []
        near_zero_variance_companies = []

        for company in companies:
            company_data = df[df[id_column] == company][column].dropna()

            if len(company_data) <= 1:
                continue

            mean = company_data.mean()
            std_dev = company_data.std()
            variance = std_dev ** 2
            cv = std_dev / mean if mean != 0 else np.nan

            company_stats.append({
                'company': company,
                'mean': mean,
                'median': company_data.median(),
                'std_dev': std_dev,
                'variance': variance,
                'cv': cv,
                'min': company_data.min(),
                'max': company_data.max(),
                'count': len(company_data)
            })

            if variance <= variance_threshold:
                near_zero_variance_companies.append({
                    'company': company,
                    'variance': variance,
                    'mean': mean,
                    'std_dev': std_dev,
                    'observations': len(company_data)
                })

            if not np.isnan(variance):
                company_variances.append(variance)

            if not np.isnan(cv):
                company_cvs.append(cv)

        variable_results['company_level'] = pd.DataFrame(company_stats)

        if company_variances:
            variable_results['variance_analysis'] = {
                'average_variance': np.mean(company_variances),
                'median_variance': np.median(company_variances),
                'min_variance': np.min(company_variances),
                'max_variance': np.max(company_variances)
            }
        else:
            variable_results['variance_analysis'] = {
                'average_variance': np.nan,
                'median_variance': np.nan,
                'min_variance': np.nan,
                'max_variance': np.nan
            }

        if company_cvs:
            percentiles = np.arange(0, 101, 10)
            cv_percentiles = np.percentile(company_cvs, percentiles)

            variable_results['cv_percentiles'] = {
                f'{p}%': cv_percentiles[i] for i, p in enumerate(percentiles)
            }
        else:
            variable_results['cv_percentiles'] = {
                f'{p}%': np.nan for p in np.arange(0, 101, 10)
            }

        variable_results['near_zero_variance_companies'] = near_zero_variance_companies

        results[column] = variable_results

        if verbose:
            print(f"  Companies with near-zero variance: {len(near_zero_variance_companies)}")
            print(f"  Average variance: {variable_results['variance_analysis']['average_variance']:.6f}")

    return results


def remove_near_zero_companies(df: pd.DataFrame, 
                              results: Dict[str, Dict], 
                              id_column: str = DEFAULT_TICKER_COLUMN,
                              verbose: bool = True) -> Tuple[pd.DataFrame, Dict]:
    
    if verbose:
        print_subsection_header("Removing Near-Zero Variance Companies")
    
    original_shape = df.shape
    companies_to_remove = set()

    for variable, analysis in results.items():
        near_zero_list = analysis['near_zero_variance_companies']
        for company_data in near_zero_list:
            companies_to_remove.add(company_data['company'])

    if companies_to_remove:
        filtered_df = df[~df[id_column].isin(companies_to_remove)].copy()
        if verbose:
            print(f"Removed {len(companies_to_remove)} companies with near-zero variance")
            print(f"Companies removed: {sorted(list(companies_to_remove))[:10]}{'...' if len(companies_to_remove) > 10 else ''}")
    else:
        filtered_df = df.copy()
        if verbose:
            print("No companies with near-zero variance found")

    rows_removed = original_shape[0] - filtered_df.shape[0]
    retention_rate = (filtered_df.shape[0] / original_shape[0]) * 100 if original_shape[0] > 0 else 0

    stats = {
        'original_shape': original_shape,
        'final_shape': filtered_df.shape,
        'companies_removed': len(companies_to_remove),
        'rows_removed': rows_removed,
        'retention_rate': retention_rate,
        'removed_companies': list(companies_to_remove)
    }

    if verbose:
        print_processing_stats(stats, "Near-Zero Variance Removal")

    return filtered_df, stats


def print_panel_analysis_report(results: Dict[str, Dict], 
                              show_near_zero: bool = True,
                              variance_threshold: float = DEFAULT_VARIANCE_THRESHOLD,
                              verbose: bool = True) -> None:
    
    if not results:
        if verbose:
            print("No results to display")
        return

    if verbose:
        print_section_header("Panel Variable Analysis Report")

    for variable, analysis in results.items():
        if verbose:
            print_subsection_header(f"Variable: {variable}")

        overall = analysis['overall_stats']
        if verbose:
            print("Overall Panel Statistics:")
            for stat, value in overall.items():
                if isinstance(value, (float, int)) and not pd.isna(value):
                    if stat in ['count', 'missing']:
                        print(f"  {stat.replace('_', ' ').title()}: {format_number(int(value))}")
                    else:
                        print(f"  {stat.replace('_', ' ').title()}: {value:.4f}")
                else:
                    print(f"  {stat.replace('_', ' ').title()}: {value}")

        variance_stats = analysis['variance_analysis']
        if verbose:
            print("\nVariance Analysis Across Companies:")
            for stat, value in variance_stats.items():
                if isinstance(value, (float, int)) and not pd.isna(value):
                    print(f"  {stat.replace('_', ' ').title()}: {value:.6f}")
                else:
                    print(f"  {stat.replace('_', ' ').title()}: {value}")

        cv_percentiles = analysis['cv_percentiles']
        if verbose:
            print("\nCoefficient of Variation Percentiles:")
            for percentile, value in cv_percentiles.items():
                if isinstance(value, (float, int)) and not pd.isna(value):
                    print(f"  {percentile}: {value:.4f}")
                else:
                    print(f"  {percentile}: {value}")

        if not analysis['company_level'].empty:
            company_stats = analysis['company_level']
            if verbose:
                print(f"\nCompany-Level Summary (Total companies analyzed: {len(company_stats)}):")

            numeric_cols = ['mean', 'median', 'std_dev', 'variance', 'cv', 'min', 'max', 'count']
            existing_cols = [col for col in numeric_cols if col in company_stats.columns]
            
            if existing_cols:
                summary = company_stats[existing_cols].describe(percentiles=[.25, .5, .75])
                if verbose:
                    print(summary)

            if show_near_zero and analysis['near_zero_variance_companies']:
                near_zero = analysis['near_zero_variance_companies']
                if verbose:
                    print(f"\nCompanies with Near-Zero Variance (threshold: {variance_threshold}):")
                    print(f"Total companies with near-zero variance: {len(near_zero)}")

                if len(near_zero) > 0:
                    if verbose:
                        print(f"\n{'Company':20} {'Variance':15} {'Mean':15} {'Std Dev':15} {'Observations':12}")
                        print("-" * 77)
                    for company_data in near_zero[:10]:
                        if verbose:
                            print(f"{str(company_data['company']):20} {company_data['variance']:15.8f} "
                                f"{company_data['mean']:15.4f} {company_data['std_dev']:15.8f} "
                                f"{company_data['observations']:12d}")

        if verbose:
            print()

    if verbose:
        print("Analysis complete.")


def analyze_panel_dataset(df: pd.DataFrame, 
                        columns: List[str], 
                        id_column: str = DEFAULT_TICKER_COLUMN, 
                        variance_threshold: float = DEFAULT_VARIANCE_THRESHOLD, 
                        print_report: bool = True, 
                        show_near_zero: bool = True, 
                        remove_near_zero: bool = False, 
                        save_filtered: bool = False, 
                        filtered_file_path: str = 'filtered_panel_data.csv',
                        verbose: bool = True) -> Tuple[Dict[str, Dict], pd.DataFrame]:
    
    if verbose:
        print_section_header("Panel Dataset Variance Analysis")
        print(f"Analyzing {len(columns)} variables with variance threshold: {variance_threshold}")

    results = analyze_panel_variables(df, columns, id_column, variance_threshold, verbose)

    if remove_near_zero:
        df_filtered, removal_stats = remove_near_zero_companies(df, results, id_column, verbose)

        if save_filtered:
            df_filtered.to_csv(filtered_file_path, index=False)
            if verbose:
                print(f"Filtered dataset saved to: {filtered_file_path}")
    else:
        df_filtered = df.copy()

    if print_report:
        print_panel_analysis_report(results, show_near_zero, variance_threshold, verbose)

        if remove_near_zero:
            if verbose:
                print_section_header("Companies Removed Due to Near-Zero Variance")

            for column in columns:
                if column in results:
                    near_zero = results[column]['near_zero_variance_companies']
                    companies = [item['company'] for item in near_zero]
                    if verbose:
                        print(f"Variable: {column} - {len(companies)} companies removed")
                    if companies:
                        display_companies = companies[:20]
                        if verbose:
                            print(", ".join(str(company) for company in display_companies))
                        if len(companies) > 20 and verbose:
                            print(f"...and {len(companies) - 20} more")

            if verbose:
                print(f"Original dataset shape: {df.shape}")
                print(f"Filtered dataset shape: {df_filtered.shape}")

    return results, df_filtered
