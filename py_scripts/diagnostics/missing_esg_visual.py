import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_esg_missing_data(file_path='decircused_data.csv'):
    print(f"Loading panel dataset from {file_path}...")
    df = pd.read_csv(file_path)

    print(f"Dataset shape: {df.shape}")
    print(f"Number of unique companies: {df['ticker'].nunique()}")
    print("\nMissing data overview:")
    print(f"Missing ESG scores: {df['environmentalScore'].isna().sum()} / {len(df)} rows")

    print("\nCalculating missing rates per company...")

    company_missing_rates = {}
    for ticker in df['ticker'].unique():
        company_data = df[df['ticker'] == ticker]
        total_rows = len(company_data)
        missing_rows = company_data['environmentalScore'].isna().sum()
        missing_rate = missing_rows / total_rows
        company_missing_rates[ticker] = {
            'total_observations': total_rows,
            'missing_rows': missing_rows,
            'missing_rate': missing_rate
        }

    missing_df = pd.DataFrame.from_dict(company_missing_rates, orient='index')
    missing_df.reset_index(inplace=True)
    missing_df.rename(columns={'index': 'ticker'}, inplace=True)

    csv_output = 'esg_missing_data_rates.csv'
    missing_df.to_csv(csv_output, index=False)
    print(f"Missing-rate DataFrame saved to {csv_output}")

    print("\nMissing rate summary statistics:")
    print(missing_df['missing_rate'].describe())

    plt.style.use('classic')
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12

    avg_obs_per_company = missing_df['total_observations'].mean()

    plt.figure(figsize=(12, 8), facecolor='white')
    ax = plt.gca()
    ax.set_facecolor('white')

    if missing_df['total_observations'].std() < 0.5:
        possible_bins = int(missing_df['total_observations'].iloc[0]) + 1
        sns.histplot(
            data=missing_df,
            x='missing_rate',
            bins=possible_bins,
            kde=False,
            color='gray',
            edgecolor=None
        )
    else:
        num_bins = min(20, len(missing_df) // 5)
        sns.histplot(
            data=missing_df,
            x='missing_rate',
            bins=num_bins,
            kde=False,
            color='gray',
            edgecolor=None
        )

    plt.title('')
    plt.xlabel('Missing Data Rate', fontsize=12)
    plt.ylabel('Number of Companies', fontsize=12)
    plt.gca().xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f'{x:.0%}')
    )
    plt.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_edgecolor('gray')
    ax.spines['left'].set_edgecolor('gray')
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.tick_params(axis='both', which='both', length=0)

    output_path = 'esg_missing_data_histogram.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nHistogram saved to {output_path}")
    plt.show()

    stats = {
        'num_companies': df['ticker'].nunique(),
        'avg_missing_rate': missing_df['missing_rate'].mean(),
        'median_missing_rate': missing_df['missing_rate'].median(),
        'overall_missing_rate': df['environmentalScore'].isna().sum() / len(df)
    }

    return missing_df, stats
