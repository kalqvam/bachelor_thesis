import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def process_outliers_in_dataset(file_path, threshold=3):
    df = pd.read_csv(file_path)

    print(f"Original dataset shape: {df.shape}")
    print(f"Number of unique tickers: {df['ticker'].nunique()}")

    df['quarter'] = pd.to_datetime(df['quarter'])
    df['year'] = df['quarter'].dt.year
    df['quarter_num'] = df['quarter'].dt.quarter

    excluded_periods = [
        (2020, 1), (2020, 2),
        (2022, 1), (2022, 2)
    ]

    def is_excluded_period(row):
        return (row['year'], row['quarter_num']) in excluded_periods

    excluded_mask = df.apply(is_excluded_period, axis=1)
    print(f"Number of rows in excluded periods: {excluded_mask.sum()}")

    columns_to_analyze = ['ebitda', 'revenue', 'cashAndCashEquivalents', 'totalDebt', 'totalAssets']

    tickers_to_remove = set()

    stats = {
        'total_outliers': {col: 0 for col in columns_to_analyze},
        'single_spikes': {col: 0 for col in columns_to_analyze},
        'fixed_outliers': {col: 0 for col in columns_to_analyze},
        'tickers_removed': 0
    }

    for ticker in df['ticker'].unique():
        ticker_data = df[df['ticker'] == ticker].copy()

        non_excluded_data = ticker_data[~ticker_data.apply(is_excluded_period, axis=1)]
        if len(non_excluded_data) < 2:
            continue

        has_non_single_spike_outlier = False

        for column in columns_to_analyze:
            column_mean = non_excluded_data[column].mean()
            column_std = non_excluded_data[column].std()

            if column_std == 0 or np.isnan(column_std):
                continue

            outlier_mask = abs(ticker_data[column] - column_mean) > threshold * column_std
            outlier_indices = ticker_data[outlier_mask].index.tolist()

            stats['total_outliers'][column] += len(outlier_indices)

            for idx in outlier_indices:
                if is_excluded_period(df.loc[idx]):
                    continue

                sorted_ticker_data = ticker_data.sort_values('quarter')
                outlier_pos = sorted_ticker_data.index.get_loc(idx)

                is_single = True

                sorted_indices = sorted_ticker_data.index.tolist()
                current_pos = sorted_indices.index(idx)

                has_enough_neighbors = (current_pos >= 2) and (current_pos < len(sorted_indices) - 2)

                if not has_enough_neighbors:
                    is_single = False
                else:
                    neighbor_indices = [
                        sorted_indices[current_pos - 2],
                        sorted_indices[current_pos - 1],
                        sorted_indices[current_pos + 1],
                        sorted_indices[current_pos + 2]
                    ]

                    for n_idx in neighbor_indices:
                        if abs(df.loc[n_idx, column] - column_mean) > 1 * column_std:
                            is_single = False
                            break

                if is_single:
                    stats['single_spikes'][column] += 1

                    prev_idx = sorted_indices[current_pos - 1]
                    next_idx = sorted_indices[current_pos + 1]

                    prev_val = df.loc[prev_idx, column]
                    next_val = df.loc[next_idx, column]

                    df.loc[idx, column] = (prev_val + next_val) / 2
                    stats['fixed_outliers'][column] += 1
                else:
                    has_non_single_spike_outlier = True

        if has_non_single_spike_outlier:
            tickers_to_remove.add(ticker)

    if tickers_to_remove:
        df_filtered = df[~df['ticker'].isin(tickers_to_remove)]
        stats['tickers_removed'] = len(tickers_to_remove)
    else:
        df_filtered = df

    print("\nOutlier Processing Summary:")
    print(f"Total tickers processed: {df['ticker'].nunique()}")
    print(f"Tickers removed: {stats['tickers_removed']} ({len(tickers_to_remove) / df['ticker'].nunique() * 100:.2f}%)")
    print(f"Tickers removed: {', '.join(sorted(tickers_to_remove))}")

    for column in columns_to_analyze:
        print(f"\n{column.upper()} Statistics:")
        print(f"  Total outliers detected: {stats['total_outliers'][column]}")
        print(f"  Single spike outliers: {stats['single_spikes'][column]}")
        print(f"  Fixed by neighbor averaging: {stats['fixed_outliers'][column]}")

    print(f"\nFinal dataset shape: {df_filtered.shape}")
    print(f"Final number of unique tickers: {df_filtered['ticker'].nunique()}")

    return df_filtered, tickers_to_remove
