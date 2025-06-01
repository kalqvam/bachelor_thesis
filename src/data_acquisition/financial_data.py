import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm
from typing import Optional, Dict, List
from datetime import datetime

from .api_client import ApiManager
from ..utils import (
    API_BATCH_SIZE, FINANCIAL_COLUMNS, 
    print_progress_update, format_number
)


def filter_by_year(df: pd.DataFrame, start_year: int) -> Optional[pd.DataFrame]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None

    df['year'] = pd.to_datetime(df['date']).dt.year
    filtered_df = df[df['year'] >= start_year].copy()
    filtered_df.drop('year', axis=1, inplace=True)

    if filtered_df.empty:
        return None

    return filtered_df


def merge_financial_data(income_df: pd.DataFrame, balance_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if income_df is None or not isinstance(income_df, pd.DataFrame) or income_df.empty:
        return None
    if balance_df is None or not isinstance(balance_df, pd.DataFrame) or balance_df.empty:
        return None

    income_df['date'] = pd.to_datetime(income_df['date'])
    balance_df['date'] = pd.to_datetime(balance_df['date'])

    merged_df = pd.merge(income_df, balance_df, on='date', how='inner')

    for col in merged_df.columns:
        if '_x' in col or '_y' in col:
            base_col = col.split('_')[0]
            cols_to_check = [f"{base_col}_x", f"{base_col}_y"]
            cols_present = [c for c in cols_to_check if c in merged_df.columns]

            if cols_present:
                merged_df[base_col] = None
                for c in cols_present:
                    merged_df[base_col] = merged_df[base_col].combine_first(merged_df[c])

                merged_df.drop(cols_present, axis=1, inplace=True)

    merged_df = merged_df.sort_values('date', ascending=False)

    merged_df['quarter'] = None
    merged_df['quarter_year'] = None

    for idx, row in merged_df.iterrows():
        date = row['date']
        year = date.year
        month = date.month

        if 'period' in row and pd.notna(row['period']):
            period = row['period']

            if period == 'Q4' and month in [1, 2, 3]:
                year_to_use = year - 1
            elif period == 'Q1' and month == 12:
                year_to_use = year + 1
            else:
                year_to_use = year

            quarter = period
        else:
            if month in [1, 2, 3]:
                quarter = 'Q1'
            elif month in [4, 5, 6]:
                quarter = 'Q2'
            elif month in [7, 8, 9]:
                quarter = 'Q3'
            else:
                quarter = 'Q4'

            year_to_use = year

        merged_df.at[idx, 'quarter'] = quarter
        merged_df.at[idx, 'quarter_year'] = f"{quarter}-{year_to_use}"

    return merged_df


def get_financial_data_for_ticker(ticker: str, api_manager: ApiManager, start_year: int) -> Optional[pd.DataFrame]:
    try:
        income_url = f"https://financialmodelingprep.com/api/v3/income-statement/{ticker}?period=quarter&apikey={api_manager.api_key}"
        income_data = api_manager.get(income_url)

        if not income_data:
            print(f"No income data for {ticker}")
            return None

        balance_url = f"https://financialmodelingprep.com/api/v3/balance-sheet-statement/{ticker}?period=quarter&apikey={api_manager.api_key}"
        balance_data = api_manager.get(balance_url)

        if not balance_data:
            print(f"No balance sheet data for {ticker}")
            return None

        required_income_fields = ['date', 'period', 'revenue', 'ebitda']
        filtered_income_data = []
        for item in income_data:
            filtered_item = {field: item.get(field) for field in required_income_fields if field in item}
            filtered_income_data.append(filtered_item)

        required_balance_fields = ['date', 'period', 'totalAssets', 'totalDebt', 'cashAndCashEquivalents', 'totalCurrentLiabilities', 'totalCurrentAssets']
        filtered_balance_data = []
        for item in balance_data:
            filtered_item = {field: item.get(field) for field in required_balance_fields if field in item}
            filtered_balance_data.append(filtered_item)

        income_df = pd.DataFrame(filtered_income_data)
        balance_df = pd.DataFrame(filtered_balance_data)

        if income_df.empty or balance_df.empty:
            print(f"Empty dataframe for {ticker}")
            return None

        income_df = filter_by_year(income_df, start_year)
        balance_df = filter_by_year(balance_df, start_year)

        if income_df is None or balance_df is None:
            return None

        merged_df = merge_financial_data(income_df, balance_df)
        return merged_df

    except Exception as e:
        print(f"Error processing financial data for {ticker}: {e}")
        return None


def get_financial_data_parallel(tickers: List[str], api_manager: ApiManager, start_year: int, max_workers: int = 10) -> Dict[str, Optional[pd.DataFrame]]:
    results = {}

    def process_ticker(ticker: str):
        return ticker, get_financial_data_for_ticker(ticker, api_manager, start_year)

    batch_size = API_BATCH_SIZE

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        batch_results = {}

        print(f"Processing batch {i//batch_size + 1}/{(len(tickers) + batch_size - 1)//batch_size}, tickers {i+1}-{min(i+batch_size, len(tickers))}")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []

            for ticker in batch:
                futures.append(executor.submit(process_ticker, ticker))

            for future in tqdm(futures, total=len(futures), desc=f"Batch {i//batch_size + 1}"):
                ticker, data = future.result()
                batch_results[ticker] = data

        results.update(batch_results)

        if i + batch_size < len(tickers):
            print(f"Completed batch of {len(batch)} tickers. Taking a pause to respect rate limits...")
            time.sleep(22)

    return results


def validate_financial_data(df: pd.DataFrame) -> Dict[str, any]:
    validation_stats = {
        'total_rows': len(df),
        'missing_data': {},
        'data_ranges': {},
        'warnings': []
    }

    required_columns = ['date', 'revenue', 'ebitda', 'totalAssets']
    missing_required = [col for col in required_columns if col not in df.columns]
    
    if missing_required:
        validation_stats['warnings'].append(f"Missing required columns: {missing_required}")

    for col in FINANCIAL_COLUMNS:
        if col in df.columns:
            missing_count = df[col].isna().sum()
            missing_pct = (missing_count / len(df)) * 100
            validation_stats['missing_data'][col] = {
                'count': missing_count,
                'percentage': missing_pct
            }

            numeric_data = df[col].dropna()
            if len(numeric_data) > 0:
                validation_stats['data_ranges'][col] = {
                    'min': numeric_data.min(),
                    'max': numeric_data.max(),
                    'mean': numeric_data.mean(),
                    'negative_count': (numeric_data < 0).sum()
                }

    if 'revenue' in df.columns and 'ebitda' in df.columns:
        revenue_zero = (df['revenue'] == 0).sum()
        ebitda_revenue_ratio = df['ebitda'] / df['revenue']
        extreme_ratios = ((ebitda_revenue_ratio > 2) | (ebitda_revenue_ratio < -1)).sum()
        
        validation_stats['business_logic'] = {
            'zero_revenue_count': revenue_zero,
            'extreme_ebitda_ratios': extreme_ratios
        }

    return validation_stats


def print_financial_data_summary(financial_data: Dict[str, Optional[pd.DataFrame]]):
    successful_tickers = sum(1 for data in financial_data.values() if data is not None)
    failed_tickers = len(financial_data) - successful_tickers
    
    print(f"\nFinancial Data Collection Summary:")
    print(f"Total tickers processed: {format_number(len(financial_data))}")
    print(f"Successful: {format_number(successful_tickers)} ({successful_tickers/len(financial_data)*100:.1f}%)")
    print(f"Failed: {format_number(failed_tickers)} ({failed_tickers/len(financial_data)*100:.1f}%)")

    if successful_tickers > 0:
        total_observations = sum(len(data) for data in financial_data.values() if data is not None)
        avg_observations = total_observations / successful_tickers
        print(f"Total financial observations: {format_number(total_observations)}")
        print(f"Average observations per ticker: {avg_observations:.1f}")

    failed_ticker_names = [ticker for ticker, data in financial_data.items() if data is None]
    if failed_ticker_names and len(failed_ticker_names) <= 10:
        print(f"Failed tickers: {', '.join(failed_ticker_names)}")
    elif len(failed_ticker_names) > 10:
        print(f"Failed tickers (first 10): {', '.join(failed_ticker_names[:10])}...")

    return {
        'total_tickers': len(financial_data),
        'successful_tickers': successful_tickers,
        'failed_tickers': failed_tickers,
        'success_rate': successful_tickers/len(financial_data) if len(financial_data) > 0 else 0
    }
