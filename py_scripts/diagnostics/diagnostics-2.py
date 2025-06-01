import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import random
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100

def load_data(file_path):
    df = pd.read_csv(file_path)

    df['date'] = pd.to_datetime(df['date'])

    df = df.sort_values(['ticker', 'date'])

    print(f"Data loaded with {df.shape[0]} rows and {df.shape[1]} columns")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Number of unique companies: {df['ticker'].nunique()}")

    return df

def sample_companies(df, n=5):
    unique_companies = df['ticker'].unique()
    sample = random.sample(list(unique_companies), min(n, len(unique_companies)))
    return sample

def plot_company_time_series(df, companies, column, title, ylabel, normalize=False, plot_type='line'):
    plt.figure(figsize=(14, 8))

    for company in companies:
        company_data = df[df['ticker'] == company].sort_values('date')

        if len(company_data) == 0:
            print(f"No data for company {company}")
            continue

        values = company_data[column].values

        if normalize and len(values) > 0:
            values = values / values[0] if values[0] != 0 else values

        if plot_type == 'line':
            plt.plot(company_data['date'], values, marker='o', label=f"{company}")
        else:
            plt.bar(company_data['date'], values, alpha=0.7, label=f"{company}")

    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.show()

def plot_overall_time_series(df, column, title, ylabel, aggregation='sum'):
    if aggregation == 'sum':
        agg_data = df.groupby('date')[column].sum().reset_index()
        agg_label = 'Sum'
    elif aggregation == 'mean':
        agg_data = df.groupby('date')[column].mean().reset_index()
        agg_label = 'Mean'
    elif aggregation == 'median':
        agg_data = df.groupby('date')[column].median().reset_index()
        agg_label = 'Median'
    else:
        raise ValueError("Aggregation must be 'sum', 'mean', or 'median'")

    plt.figure(figsize=(14, 8))
    plt.plot(agg_data['date'], agg_data[column], marker='o', linewidth=2, label=agg_label)

    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.show()

def plot_mean_median_time_series(df, column, title, ylabel):
    mean_data = df.groupby('date')[column].mean().reset_index()

    plt.figure(figsize=(14, 8))
    plt.plot(mean_data['date'], mean_data[column],
             linestyle='-', marker='o', linewidth=2, label='Mean')

    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.show()

def plot_histograms(df, columns, title, bins=30):
    plt.figure(figsize=(14, 8))

    for column in columns:
        if column in df.columns:
            sns.histplot(df[column].dropna(), bins=bins, alpha=0.5, kde=True, label=column)

    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_categorical_histogram(df, column, title):
    if column not in df.columns:
        print(f"Column {column} not found in the dataframe")
        return

    category_counts = df[column].value_counts().sort_index()

    plt.figure(figsize=(14, 8))
    plt.bar(category_counts.index, category_counts.values)
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.grid(True, axis='y')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def calculate_xtsum(df, columns):
    result = []

    for column in columns:
        if column not in df.columns:
            print(f"Column {column} not found in the dataframe")
            continue

        overall_mean = df[column].mean()
        overall_std = df[column].std()
        overall_min = df[column].min()
        overall_max = df[column].max()
        n_obs = df[column].count()
        overall_skewness = df[column].skew()
        overall_kurtosis = df[column].kurt()
        overall_cv = (overall_std / overall_mean) * 100 if overall_mean != 0 else np.nan

        company_means = df.groupby('ticker')[column].mean()
        between_std = company_means.std()
        between_min = company_means.min()
        between_max = company_means.max()
        n_companies = company_means.count()
        between_skewness = company_means.skew()
        between_kurtosis = company_means.kurt()
        between_cv = (between_std / company_means.mean()) * 100 if company_means.mean() != 0 else np.nan

        company_means_expanded = df.merge(
            pd.DataFrame({'ticker': company_means.index, f'{column}_mean': company_means.values}),
            on='ticker', how='left'
        )

        within_values = df[column] - company_means_expanded[f'{column}_mean'] + overall_mean
        within_std = within_values.std()
        within_min = within_values.min()
        within_max = within_values.max()
        within_skewness = within_values.skew()
        within_kurtosis = within_values.kurt()
        within_cv = (within_std / within_values.mean()) * 100 if within_values.mean() != 0 else np.nan

        result.append({
            'Variable': column,
            'Mean': overall_mean,
            'Std Dev': overall_std,
            'CV (%)': overall_cv,
            'Min': overall_min,
            'Max': overall_max,
            'Skewness': overall_skewness,
            'Kurtosis': overall_kurtosis,
            'Observations': n_obs,
            #'Between Std Dev': between_std,
            'Between CV (%)': between_cv,
            'Between Min': between_min,
            'Between Max': between_max,
            'Between Skewness': between_skewness,
            'Between Kurtosis': between_kurtosis,
            #'Within Std Dev': within_std,
            'Within CV (%)': within_cv,
            'Within Min': within_min,
            'Within Max': within_max,
            'Within Skewness': within_skewness,
            'Within Kurtosis': within_kurtosis,
        })

    return pd.DataFrame(result)

def run_all_diagnostics(file_path):
    print("Loading data...")
    df = load_data(file_path)

    print("\nSampling companies...")
    sampled_companies = sample_companies(df, n=5)
    print(f"Sampled companies: {sampled_companies}")

    print("\n=== Company-Level Diagnostics ===")
    """
    plot_company_time_series(
        df, sampled_companies, 'ebitda_to_revenue_ratio',
        'EBITDA to Revenue Ratio',
        'EBITDA to Revenue Ratio'
    )

    plot_mean_median_time_series(df, 'ebitda_to_revenue_ratio', 'EBITDA to Revenue Ratio Over Time', 'EBITDA to Revenue Ratio')
    # Uncomment the following code as needed for your analysis

    plot_histograms(df, ['column1', 'column2', 'column3'], 'Distribution of Key Metrics', bins=50)

    plot_overall_time_series(df, 'ebitda_to_revenue_ratio', 'EBITDA to Revenue Ratio Over Time', 'EBITDA to Revenue Ratio', "mean")
    """
    xtsum_results = calculate_xtsum(df, [
                                         #'ebitda',
                                         #'totalAssets', 'totalDebt', 'cashAndCashEquivalents',
                                         #'ebitda_to_revenue_ratio',
                                         #'totalDebt_to_totalAssets_ratio',
                                         #'cashAndCashEquivalents_to_totalAssets_ratio',
                                         #'environmentalScore',
                                         #'socialScore',
                                         #'governanceScore'
                                         'totalAssets',
                                         ])
    print(xtsum_results)
