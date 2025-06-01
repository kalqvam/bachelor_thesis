import pandas as pd
import numpy as np

def filter_companies_by_multiple_columns(file_path, column_filters, output_path=None, verbose=True):
    if verbose:
        print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)

    if verbose:
        print(f"Original dataset shape: {df.shape}")
        print(f"Number of unique companies: {df['ticker'].nunique()}")

    stats = {
        'original_shape': df.shape,
        'original_companies': df['ticker'].nunique(),
        'removed_companies': {},
        'total_removed_companies': 0,
        'removed_rows': 0,
        'final_shape': None,
        'final_companies': None,
        'column_stats': {}
    }

    for column_name in column_filters.keys():
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found in the dataset.")

    all_companies_to_remove = set()

    for column_name, filter_values in column_filters.items():
        min_value = filter_values.get('min_value')
        max_value = filter_values.get('max_value')

        companies_to_remove_for_column = set()

        if verbose:
            print(f"\nFiltering companies based on {column_name} values...")
            if min_value is not None:
                print(f"Minimum acceptable value: {min_value}")
            if max_value is not None:
                print(f"Maximum acceptable value: {max_value}")

        for ticker in df['ticker'].unique():
            ticker_data = df[df['ticker'] == ticker]

            should_remove = False
            if min_value is not None and (ticker_data[column_name] < min_value).any():
                should_remove = True

            if not should_remove and max_value is not None and (ticker_data[column_name] > max_value).any():
                should_remove = True

            if should_remove:
                companies_to_remove_for_column.add(ticker)
                all_companies_to_remove.add(ticker)

        stats['removed_companies'][column_name] = companies_to_remove_for_column
        stats['column_stats'][column_name] = {
            'min_value': min_value,
            'max_value': max_value,
            'companies_removed': len(companies_to_remove_for_column)
        }

        if verbose:
            print(f"Found {len(companies_to_remove_for_column)} companies with {column_name} values outside the specified range.")
            if companies_to_remove_for_column:
                sample = list(companies_to_remove_for_column)[:10]
                print(f"Sample of removed companies: {', '.join(sample)}{'...' if len(companies_to_remove_for_column) > 10 else ''}")

    stats['total_removed_companies'] = len(all_companies_to_remove)

    if verbose:
        print(f"\nTotal companies removed across all filters: {stats['total_removed_companies']}")

    rows_before = df.shape[0]
    df_filtered = df[~df['ticker'].isin(all_companies_to_remove)].copy()
    stats['removed_rows'] = rows_before - df_filtered.shape[0]

    stats['final_shape'] = df_filtered.shape
    stats['final_companies'] = df_filtered['ticker'].nunique()

    if verbose:
        print(f"\nFiltering results:")
        print(f"Rows removed: {stats['removed_rows']}")
        print(f"Final dataset shape: {stats['final_shape'][0]} rows, {stats['final_shape'][1]} columns")
        print(f"Final number of companies: {stats['final_companies']}")

        retention_pct_rows = (stats['final_shape'][0] / stats['original_shape'][0]) * 100
        retention_pct_companies = (stats['final_companies'] / stats['original_companies']) * 100
        print(f"Percentage of rows retained: {retention_pct_rows:.2f}%")
        print(f"Percentage of companies retained: {retention_pct_companies:.2f}%")

        print("\nColumn-specific filtering results:")
        for col, col_stats in stats['column_stats'].items():
            print(f"  {col}:")
            print(f"    Min value: {col_stats['min_value']}")
            print(f"    Max value: {col_stats['max_value']}")
            print(f"    Companies removed: {col_stats['companies_removed']}")

    if output_path:
        df_filtered.to_csv(output_path, index=False)
        if verbose:
            print(f"\nFiltered dataset saved to {output_path}")

    return df_filtered, stats
