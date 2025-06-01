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

    if 'sector_num' in df.columns and not pd.api.types.is_numeric_dtype(df['sector_num']):
        print("Warning: sector_num column is not numeric. This code assumes numeric sector values.")

    print(f"Data loaded with {df.shape[0]} rows and {df.shape[1]} columns")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Number of unique companies: {df['ticker'].nunique()}")
    if 'sector_num' in df.columns:
        print(f"Number of unique sectors: {df['sector_num'].nunique()}")
        print(f"Sector values: {sorted(df['sector_num'].unique())}")

    return df

def sample_companies(df, n=5):
    if 'sector_num' in df.columns:
        sectors = df['sector_num'].unique()
        sample = []

        for sector in sectors[:min(n, len(sectors))]:
            companies_in_sector = df[df['sector_num'] == sector]['ticker'].unique()
            if len(companies_in_sector) > 0:
                sample.append(random.choice(companies_in_sector))

        if len(sample) < n:
            remaining_companies = set(df['ticker'].unique()) - set(sample)
            sample.extend(random.sample(list(remaining_companies), min(n - len(sample), len(remaining_companies))))

        return sample[:n]
    else:
        unique_companies = df['ticker'].unique()
        return random.sample(list(unique_companies), min(n, len(unique_companies)))

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
            if 'sector_num' in company_data.columns:
                sector_label = f"Sector {int(company_data['sector_num'].iloc[0])}"
                plt.plot(company_data['date'], values, marker='o', label=f"{company} ({sector_label})")
            else:
                plt.plot(company_data['date'], values, marker='o', label=f"{company}")
        else:
            if 'sector_num' in company_data.columns:
                sector_label = f"Sector {int(company_data['sector_num'].iloc[0])}"
                plt.bar(company_data['date'], values, alpha=0.7, label=f"{company} ({sector_label})")
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

def plot_sector_time_series(df, column, title, ylabel, aggregation='sum'):
    if 'sector_num' not in df.columns:
        print("Error: sector_num column not found in the dataframe")
        return

    if aggregation == 'sum':
        sector_data = df.groupby(['sector_num', 'date'])[column].sum().reset_index()
    elif aggregation == 'mean':
        sector_data = df.groupby(['sector_num', 'date'])[column].mean().reset_index()
    elif aggregation == 'median':
        sector_data = df.groupby(['sector_num', 'date'])[column].median().reset_index()
    else:
        raise ValueError("Aggregation must be 'sum', 'mean', or 'median'")

    plt.figure(figsize=(14, 10))

    for sector in sorted(sector_data['sector_num'].unique()):
        sector_subset = sector_data[sector_data['sector_num'] == sector]
        plt.plot(sector_subset['date'], sector_subset[column], marker='o', label=f"Sector {int(sector)}")

    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel(ylabel)
    plt.legend(loc='best', bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.show()

def plot_sector_mean_median_time_series(df, column, title, ylabel):
    if 'sector_num' not in df.columns:
        print("Error: sector_num column not found in the dataframe")
        return

    sector_mean = df.groupby(['sector_num', 'date'])[column].mean().reset_index()

    plt.figure(figsize=(14, 10))

    for sector in sorted(sector_mean['sector_num'].unique()):
        sector_mean_subset = sector_mean[sector_mean['sector_num'] == sector]
        plt.plot(sector_mean_subset['date'], sector_mean_subset[column],
                 linestyle='-', marker='o', label=f"Sector {int(sector)} (Mean)")

    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel(ylabel)
    plt.legend(loc='best', bbox_to_anchor=(1, 1))
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

def calculate_sector_summary_statistics(df, columns):
    if 'sector_num' not in df.columns:
        print("Error: sector_num column not found in the dataframe")
        return None

    result = []

    for sector in sorted(df['sector_num'].unique()):
        sector_data = df[df['sector_num'] == sector]
        sector_stats = {'Sector': int(sector), 'Companies': sector_data['ticker'].nunique()}

        for column in columns:
            if column in df.columns:
                values = sector_data[column].dropna()

                if len(values) > 0:
                    sector_stats[f'{column}_mean'] = values.mean()
                    sector_stats[f'{column}_median'] = values.median()
                    sector_stats[f'{column}_min'] = values.min()
                    sector_stats[f'{column}_max'] = values.max()
                    sector_stats[f'{column}_std'] = values.std()

                    if len(values) > 2:
                        sector_stats[f'{column}_skewness'] = stats.skew(values)
                        sector_stats[f'{column}_kurtosis'] = stats.kurtosis(values)
                    else:
                        sector_stats[f'{column}_skewness'] = np.nan
                        sector_stats[f'{column}_kurtosis'] = np.nan
                else:
                    sector_stats[f'{column}_mean'] = np.nan
                    sector_stats[f'{column}_median'] = np.nan
                    sector_stats[f'{column}_min'] = np.nan
                    sector_stats[f'{column}_max'] = np.nan
                    sector_stats[f'{column}_std'] = np.nan
                    sector_stats[f'{column}_skewness'] = np.nan
                    sector_stats[f'{column}_kurtosis'] = np.nan

        result.append(sector_stats)

    return pd.DataFrame(result)

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
            'Min': overall_min,
            'Max': overall_max,
            'Skewness': overall_skewness,
            'Kurtosis': overall_kurtosis,
            'Observations': n_obs,
            'Between Std Dev': between_std,
            'Between Min': between_min,
            'Between Max': between_max,
            'Between Skewness': between_skewness,
            'Between Kurtosis': between_kurtosis,
            'Within Std Dev': within_std,
            'Within Min': within_min,
            'Within Max': within_max,
            'Within Skewness': within_skewness,
            'Within Kurtosis': within_kurtosis,
        })

    return pd.DataFrame(result)

def run_all_diagnostics(file_path):
    print("Loading data...")
    df = load_data(file_path)

    print("\nSampling companies from different sectors...")
    sampled_companies = sample_companies(df, n=5)
    print(f"Sampled companies: {sampled_companies}")

    print(calculate_sector_summary_statistics(df, ['ebitda_to_revenue_ratio', 'environmentalScore', 'socialScore', 'governanceScore']))
