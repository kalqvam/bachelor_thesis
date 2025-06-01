import pandas as pd
import numpy as np

_global_variance_threshold = 1e-5

def analyze_panel_variables(df, columns, id_column='ticker', variance_threshold=1e-5):
    results = {}

    valid_columns = [col for col in columns if col in df.columns and col != id_column]
    if not valid_columns:
        print(f"None of the provided columns {columns} exist in the dataframe")
        return {}

    companies = df[id_column].unique()
    n_companies = len(companies)

    for column in valid_columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            print(f"Skipping non-numeric column: {column}")
            continue

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

    return results

def remove_near_zero_companies(df, results, id_column='ticker'):
    companies_to_remove = set()

    for variable, analysis in results.items():
        near_zero_list = analysis['near_zero_variance_companies']
        for company_data in near_zero_list:
            companies_to_remove.add(company_data['company'])

    if companies_to_remove:
        filtered_df = df[~df[id_column].isin(companies_to_remove)].copy()
        return filtered_df
    else:
        return df.copy()

def print_panel_analysis_report(results, show_near_zero=True):
    if not results:
        print("No results to display")
        return

    for variable, analysis in results.items():
        print("\n" + "="*80)
        print(f"ANALYSIS FOR VARIABLE: {variable}")
        print("="*80)

        print("\nOVERALL PANEL STATISTICS:")
        print("-"*50)
        overall = analysis['overall_stats']
        for stat, value in overall.items():
            print(f"{stat.replace('_', ' ').title():20}: {value:,.4f}" if isinstance(value, (float, int)) and not pd.isna(value) else f"{stat.replace('_', ' ').title():20}: {value}")

        print("\nVARIANCE ANALYSIS ACROSS COMPANIES:")
        print("-"*50)
        variance = analysis['variance_analysis']
        for stat, value in variance.items():
            print(f"{stat.replace('_', ' ').title():20}: {value:,.4f}" if isinstance(value, (float, int)) and not pd.isna(value) else f"{stat.replace('_', ' ').title():20}: {value}")

        print("\nCOEFFICIENT OF VARIATION PERCENTILES:")
        print("-"*50)
        cv_percentiles = analysis['cv_percentiles']
        for percentile, value in cv_percentiles.items():
            print(f"{percentile:10}: {value:,.4f}" if isinstance(value, (float, int)) and not pd.isna(value) else f"{percentile:10}: {value}")

        if not analysis['company_level'].empty:
            company_stats = analysis['company_level']
            print(f"\nCOMPANY-LEVEL SUMMARY (Total companies analyzed: {len(company_stats)}):")
            print("-"*50)

            numeric_cols = ['mean', 'median', 'std_dev', 'variance', 'cv', 'min', 'max', 'count']
            summary = company_stats[numeric_cols].describe(percentiles=[.25, .5, .75])
            print(summary)

            if show_near_zero and analysis['near_zero_variance_companies']:
                near_zero = analysis['near_zero_variance_companies']
                print(f"\nCOMPANIES WITH NEAR-ZERO VARIANCE (threshold: {_global_variance_threshold}):")
                print("-"*70)
                print(f"Total companies with near-zero variance: {len(near_zero)}")

                if len(near_zero) > 0:
                    print(f"\n{'Company':20} {'Variance':15} {'Mean':15} {'Std Dev':15} {'Observations':12}")
                    print("-"*77)
                    for company_data in near_zero:
                        print(f"{str(company_data['company']):20} {company_data['variance']:15.8f} "
                              f"{company_data['mean']:15.4f} {company_data['std_dev']:15.8f} "
                              f"{company_data['observations']:12d}")

    print("\nAnalysis complete.")

def analyze_panel_dataset(df, columns, id_column='ticker', variance_threshold=1e-5, print_report=True, show_near_zero=True, remove_near_zero=False, save_filtered=False, filtered_file_path='filtered_panel_data.csv'):
    global _global_variance_threshold
    _global_variance_threshold = variance_threshold

    results = analyze_panel_variables(df, columns, id_column, variance_threshold)

    if remove_near_zero:
        df_filtered = remove_near_zero_companies(df, results, id_column)

        if save_filtered:
            df_filtered.to_csv(filtered_file_path, index=False)
            print(f"\nFiltered dataset saved to: {filtered_file_path}")
    else:
        df_filtered = df.copy()

    if print_report:
        print_panel_analysis_report(results, show_near_zero)

        if remove_near_zero:
            print("\n" + "="*80)
            print("COMPANIES REMOVED DUE TO NEAR-ZERO VARIANCE")
            print("="*80)

            for column in columns:
                if column in results:
                    near_zero = results[column]['near_zero_variance_companies']
                    companies = [item['company'] for item in near_zero]
                    print(f"\nVariable: {column} - {len(companies)} companies removed")
                    if companies:
                        print(", ".join(str(company) for company in companies[:20]))
                        if len(companies) > 20:
                            print(f"...and {len(companies) - 20} more")

            print(f"\nOriginal dataset shape: {df.shape}")
            print(f"Filtered dataset shape: {df_filtered.shape}")

    return results, df_filtered
