import pandas as pd
import numpy as np

def process_financial_data(file_path, consecutive_threshold=3):
    df = pd.read_csv(file_path)

    df['quarter'] = pd.to_datetime(df['quarter'])
    df['year'] = df['quarter'].dt.year
    df['quarter_num'] = df['quarter'].dt.quarter

    columns_to_analyze = ['ebitda', 'revenue', 'cashAndCashEquivalents', 'totalDebt', 'totalAssets']

    companies_to_remove = set()

    def find_consecutive_zeros(series):
        is_zero = (series == 0)

        if not any(is_zero):
            return []

        runs = []
        run_length = 0

        for i, zero in enumerate(is_zero):
            if zero:
                run_length += 1
            elif run_length > 0:
                runs.append(run_length)
                run_length = 0

        if run_length > 0:
            runs.append(run_length)

        return runs

    for ticker in df['ticker'].unique():
        ticker_data = df[df['ticker'] == ticker].sort_values('quarter')

        if len(ticker_data) < 2:
            continue

        for col in columns_to_analyze:
            if col not in ticker_data.columns:
                continue

            consecutive_runs = find_consecutive_zeros(ticker_data[col])

            if any(run_length > consecutive_threshold for run_length in consecutive_runs):
                companies_to_remove.add(ticker)
                break

    filtered_df = df[~df['ticker'].isin(companies_to_remove)]

    stats = {
        'original_company_count': len(df['ticker'].unique()),
        'removed_company_count': len(companies_to_remove),
        'remaining_company_count': len(filtered_df['ticker'].unique()),
        'companies_removed': list(companies_to_remove)
    }

    processed_df = filtered_df.copy()

    for ticker in processed_df['ticker'].unique():
        ticker_mask = processed_df['ticker'] == ticker
        ticker_indices = processed_df[ticker_mask].index

        for col in columns_to_analyze:
            if col in processed_df.columns:
                if (processed_df.loc[ticker_indices, col] == 0).any():
                    series = processed_df.loc[ticker_indices, col].replace(0, np.nan)
                    interpolated = series.interpolate(method='linear')
                    processed_df.loc[ticker_indices, col] = interpolated

    print(f"Original number of companies: {stats['original_company_count']}")
    print(f"Companies removed due to exceeding {consecutive_threshold} consecutive zeros: {stats['removed_company_count']}")
    print(f"Remaining companies after filtering: {stats['remaining_company_count']}")

    return processed_df, stats
