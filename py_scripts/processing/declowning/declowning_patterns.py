import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def clean_financial_data(input_file, output_file=None):
    print(f"Reading data from {input_file}...")
    df = pd.read_csv(input_file)

    df_original = df.copy()

    df['quarter'] = pd.to_datetime(df['quarter'])

    df['clown_flags'] = ""

    companies_to_remove = set()

    print(f"Processing {df['ticker'].nunique()} unique tickers...")

    for ticker in df['ticker'].unique():
        company_data = df[df['ticker'] == ticker].copy()
        company_data = company_data.sort_values('quarter')

        if len(company_data) <= 2:
            continue

        if len(company_data) >= 3 and 'ebitda' in company_data.columns:
            ebitda_series = company_data['ebitda'].values
            date_series = company_data['quarter'].values

            original_indices = company_data.index.values

            if len(ebitda_series) >= 5:
                for i in range(2, len(ebitda_series) - 2):
                    if (pd.isna(ebitda_series[i-2]) or pd.isna(ebitda_series[i-1]) or
                        pd.isna(ebitda_series[i]) or pd.isna(ebitda_series[i+1]) or
                        pd.isna(ebitda_series[i+2])):
                        continue

                    prev_prev = ebitda_series[i-2]
                    prev = ebitda_series[i-1]
                    current = ebitda_series[i]
                    next_val = ebitda_series[i+1]
                    next_next = ebitda_series[i+2]

                    current_sign_positive = current > 0
                    neighbors_sign_positive = [val > 0 for val in [prev_prev, prev, next_val, next_next]]

                    consistent_opposite_signs = all(sign != current_sign_positive for sign in neighbors_sign_positive)

                    if consistent_opposite_signs:
                        abs_current = abs(current)

                        magnitudes_similar = (
                            abs_current <= 1.75 * abs(prev) and abs_current >= abs(prev) / 1.75 and
                            abs_current <= 1.75 * abs(next_val) and abs_current >= abs(next_val) / 1.75
                        )

                        if magnitudes_similar:
                            logical_progression = False

                            if current_sign_positive:
                                logical_progression = abs(prev) >= abs(next_val)
                            else:
                                logical_progression = abs(prev) <= abs(next_val)

                            if logical_progression:
                                idx = original_indices[i]
                                df.at[idx, 'ebitda'] = -current
                                df.at[idx, 'clown_flags'] += "reversed_ebitda;"
                                print(f"🤡 Fixed reversed EBITDA sign for {ticker} at {date_series[i]}: {current} -> {-current} (5-quarter check)")

            for i in range(1, len(ebitda_series) - 1):
                if pd.isna(ebitda_series[i]) or pd.isna(ebitda_series[i-1]) or pd.isna(ebitda_series[i+1]):
                    continue

                current = df.loc[original_indices[i], 'ebitda']
                prev = df.loc[original_indices[i-1], 'ebitda']
                next_val = df.loc[original_indices[i+1], 'ebitda']

                if current == 0 or prev == 0 or next_val == 0:
                    continue

                if (current > 0 and prev > 0 and next_val > 0) or (current < 0 and prev < 0 and next_val < 0):
                    ratio_prev = abs(current / prev)
                    ratio_next = abs(current / next_val)

                    if (9 <= ratio_prev <= 11) and (9 <= ratio_next <= 11):
                        idx = original_indices[i]
                        corrected_value = current / 10
                        df.at[idx, 'ebitda'] = corrected_value
                        df.at[idx, 'clown_flags'] += "10x_ebitda_large;"
                        print(f"🤡 Fixed 10x too large EBITDA for {ticker} at {date_series[i]}: {current} -> {corrected_value}")

                    if (0.09 <= ratio_prev <= 0.11) and (0.09 <= ratio_next <= 0.11):
                        idx = original_indices[i]
                        corrected_value = current * 10
                        df.at[idx, 'ebitda'] = corrected_value
                        df.at[idx, 'clown_flags'] += "10x_ebitda_small;"
                        print(f"🤡 Fixed 10x too small EBITDA for {ticker} at {date_series[i]}: {current} -> {corrected_value}")

        if len(company_data) >= 4 and 'revenue' in company_data.columns and 'ebitda' in company_data.columns:
            company_data = df[df['ticker'] == ticker].copy()
            company_data = company_data.sort_values('quarter')

            company_data['revenue_pct_change'] = company_data['revenue'].pct_change().abs()
            high_volatility_periods = company_data['revenue_pct_change'] > 1.0

            if high_volatility_periods.sum() >= 2:
                high_vol_idxs = company_data[high_volatility_periods].index

                ebitda_stable_during_rev_jumps = True
                for idx in high_vol_idxs:
                    pos = company_data.index.get_loc(idx)
                    if pos > 0:
                        rev_change = company_data.loc[idx, 'revenue_pct_change']
                        ebitda_pct_change = abs((company_data['ebitda'].iloc[pos] - company_data['ebitda'].iloc[pos-1]) /
                                                company_data['ebitda'].iloc[pos-1] if company_data['ebitda'].iloc[pos-1] != 0 else 0)

                        if pd.notna(ebitda_pct_change) and ebitda_pct_change >= 0.5 * rev_change:
                            ebitda_stable_during_rev_jumps = False
                            break

                if ebitda_stable_during_rev_jumps and high_volatility_periods.sum() >= 2:
                    companies_to_remove.add(ticker)
                    print(f"🤡 Detected gross profit instead of revenue pattern for {ticker} - marked for removal")

    if companies_to_remove:
        print(f"Removing {len(companies_to_remove)} companies with inconsistent revenue patterns...")
        df_before = len(df)
        df = df[~df['ticker'].isin(companies_to_remove)]
        print(f"Removed {df_before - len(df)} rows from dataset")

    df['clown_flags'] = df['clown_flags'].replace("", np.nan)

    reversed_ebitda_count = df['clown_flags'].str.contains('reversed_ebitda', na=False).sum()
    ebitda_10x_large_count = df['clown_flags'].str.contains('10x_ebitda_large', na=False).sum()
    ebitda_10x_small_count = df['clown_flags'].str.contains('10x_ebitda_small', na=False).sum()

    print("\n===== Clown Elimination Results =====")
    print(f"Reversed EBITDA signs fixed: {reversed_ebitda_count}")
    print(f"10x too large EBITDA values fixed: {ebitda_10x_large_count}")
    print(f"10x too small EBITDA values fixed: {ebitda_10x_small_count}")
    print(f"Companies removed due to revenue/gross profit confusion: {len(companies_to_remove)}")
    print(f"Total clown flags: {df['clown_flags'].notna().sum()}")
    print(f"Percentage of rows modified: {df['clown_flags'].notna().sum() / len(df) * 100:.2f}%")

    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"cleaned_financial_data_{timestamp}.csv"

    df.to_csv(output_file, index=False)
    print(f"\nCleaned data saved to {output_file}")


    return df

if __name__ == "__main__":
    input_file = "panel_data_processed.csv"
    output_file = "declowned_panel_data.csv"

    cleaned_df = clean_financial_data(input_file, output_file)

    print(f"\nClown removal completed! Cleaned data saved to {output_file}")
