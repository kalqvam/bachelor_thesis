import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

def convert_quarter_year_to_datetime(quarter_year):
    if pd.isna(quarter_year) or not isinstance(quarter_year, str):
        return None

    match = re.search(r'Q(\d)-(\d{4})', quarter_year)
    if match:
        quarter, year = match.groups()
        month = int(quarter) * 3
        return datetime(int(year), month, 1)
    return None

def check_consecutive_missing(series, threshold=9):
    max_consecutive = 0
    current_streak = 0

    for value in series:
        if pd.isna(value):
            current_streak += 1
            max_consecutive = max(max_consecutive, current_streak)
            if max_consecutive >= threshold:
                return True
        else:
            current_streak = 0

    return False

def clean_esg_dataset(file_path, lower_year, upper_year, consecutive_missing_threshold=9,
                      columns_to_remove=None, output_path=None, verbose=True):
    if verbose:
        print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)

    if verbose:
        print(f"Original dataset shape: {df.shape}")

    stats = {
        'original_shape': df.shape,
        'duplicate_periods_count': 0,
        'duplicate_rows_removed': 0,
        'incomplete_tickers_removed': 0,
        'consecutive_missing_removed': 0,
        'final_shape': None
    }

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    if verbose:
        print("\nStep 1: Handling tickers with duplicate periods...")

    duplicates = df.duplicated(subset=['ticker', 'quarter_year'], keep=False)
    duplicate_rows = df[duplicates].copy()

    if not duplicate_rows.empty:
        if verbose:
            print(f"Found {duplicate_rows['ticker'].nunique()} tickers with duplicate quarter_year entries.")

        stats['duplicate_periods_count'] = duplicate_rows['ticker'].nunique()

        duplicate_groups = duplicate_rows.groupby(['ticker', 'quarter_year']).size()
        stats['duplicate_rows_removed'] = duplicate_groups.sum() - duplicate_groups.shape[0]

        df_no_dups = df.drop_duplicates(subset=['ticker', 'quarter_year'], keep='last')

        if verbose:
            print(f"Kept the latest observation for each duplicate period based on 'date' column.")
            print(f"Removed {df.shape[0] - df_no_dups.shape[0]} duplicate rows.")
            print(f"Dataset shape after handling duplicates: {df_no_dups.shape}")

        df = df_no_dups
    else:
        if verbose:
            print("No tickers with duplicate quarter_year entries found.")

    if verbose:
        print(f"\nStep 2: Filtering by year range ({lower_year} to {upper_year})...")

    df['year'] = df['quarter_year'].apply(
        lambda x: int(re.search(r'Q\d-(\d{4})', x).group(1)) if isinstance(x, str) else np.nan
    )
    df_filtered = df[(df['year'] >= lower_year) & (df['year'] <= upper_year)].copy()

    if verbose:
        print(f"Dataset shape after year filtering: {df_filtered.shape}")

    if verbose:
        print("\nStep 3: Converting quarter_year to datetime...")

    df_filtered.loc[:, 'quarter'] = df_filtered['quarter_year'].apply(convert_quarter_year_to_datetime)

    if verbose:
        print("\nStep 4: Checking for complete time series...")

    expected_observations = (upper_year - lower_year + 1) * 4
    ticker_counts = df_filtered['ticker'].value_counts()
    valid_tickers = ticker_counts[ticker_counts == expected_observations].index.tolist()

    if verbose:
        print(f"Expected observations per ticker: {expected_observations}")
        print(f"Number of tickers with complete data: {len(valid_tickers)} out of {len(ticker_counts)}")

    stats['incomplete_tickers_removed'] = df_filtered[~df_filtered['ticker'].isin(valid_tickers)].shape[0]

    df_filtered = df_filtered[df_filtered['ticker'].isin(valid_tickers)].copy()

    if verbose:
        print(f"Dataset shape after removing incomplete tickers: {df_filtered.shape}")

    if verbose:
        print(f"\nStep 5: Checking for tickers with {consecutive_missing_threshold}+ consecutive missing environmental scores...")

    if 'environmentalScore' not in df_filtered.columns:
        if verbose:
            print("Warning: 'environmentalScore' column not found. Skipping consecutive missing check.")
    else:
        consecutive_missing_tickers = []

        for ticker in tqdm(df_filtered['ticker'].unique(), desc="Checking consecutive missing values") if verbose else df_filtered['ticker'].unique():
            ticker_data = df_filtered[df_filtered['ticker'] == ticker].sort_values('quarter')

            if check_consecutive_missing(ticker_data['environmentalScore'], consecutive_missing_threshold):
                consecutive_missing_tickers.append(ticker)

        if consecutive_missing_tickers:
            if verbose:
                print(f"Found {len(consecutive_missing_tickers)} tickers with {consecutive_missing_threshold}+ consecutive missing environmental scores.")
                print(f"Tickers with consecutive missing values: {', '.join(consecutive_missing_tickers[:10])}{'...' if len(consecutive_missing_tickers) > 10 else ''}")

            stats['consecutive_missing_removed'] = df_filtered[df_filtered['ticker'].isin(consecutive_missing_tickers)].shape[0]

            df_filtered = df_filtered[~df_filtered['ticker'].isin(consecutive_missing_tickers)].copy()

            if verbose:
                print(f"Dataset shape after removing tickers with consecutive missing values: {df_filtered.shape}")
        else:
            if verbose:
                print(f"No tickers found with {consecutive_missing_threshold}+ consecutive missing environmental scores.")

    if columns_to_remove:
        if verbose:
            print(f"\nStep 6: Removing columns: {columns_to_remove}")

        cols_to_drop = [col for col in columns_to_remove if col in df_filtered.columns]
        if cols_to_drop:
            df_filtered.drop(columns=cols_to_drop, inplace=True)

        if verbose:
            print(f"Dataset shape after column removal: {df_filtered.shape}")
    else:
        if verbose:
            print("\nStep 6: No columns specified for removal")

    if 'year' in df_filtered.columns:
        df_filtered = df_filtered.drop(columns=['year'])
      
    if 'quarter' in df_filtered.columns:
        cols = df_filtered.columns.tolist()
        cols.remove('ticker')
        cols.remove('quarter')
        df_filtered = df_filtered[['ticker', 'quarter'] + cols]
        df_filtered = df_filtered.sort_values(['ticker', 'quarter'])

    if output_path:
        df_filtered.to_csv(output_path, index=False)
        if verbose:
            print(f"\nCleaned dataset saved to {output_path}")

    stats['final_shape'] = df_filtered.shape

    if verbose:
        print("\nCleaning Summary Statistics:")
        print(f"Original dataset: {stats['original_shape'][0]} rows, {stats['original_shape'][1]} columns")
        print(f"Tickers with duplicate periods: {stats['duplicate_periods_count']}")
        print(f"Duplicate rows removed: {stats['duplicate_rows_removed']}")
        print(f"Rows removed due to incomplete time series: {stats['incomplete_tickers_removed']}")
        print(f"Rows removed due to consecutive missing values: {stats['consecutive_missing_removed']}")
        print(f"Final dataset: {stats['final_shape'][0]} rows, {stats['final_shape'][1]} columns")
        print(f"Rows removed total: {stats['original_shape'][0] - stats['final_shape'][0]}")

        retention_pct = (stats['final_shape'][0] / stats['original_shape'][0]) * 100
        print(f"Percentage of data retained: {retention_pct:.2f}%")

    return df_filtered, stats
