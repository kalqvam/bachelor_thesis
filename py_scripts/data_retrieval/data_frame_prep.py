import pandas as pd
import requests
import time
import datetime
from tqdm.auto import tqdm
import numpy as np
import os
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor

class ApiManager:
    def __init__(self, api_key, requests_per_minute=300):
        self.api_key = api_key
        self.rate_limit = requests_per_minute
        self.cache = {}
        self.call_times = []
        self.lock = threading.Lock()
        self.requests_count = 0
        self.esg_request_counter = 0
        self.profile_request_counter = 0

    def _check_rate_limit(self):
        now = time.time()
        with self.lock:
            self.call_times = [t for t in self.call_times if now - t < 60]
            self.requests_count += 1
            if len(self.call_times) < self.rate_limit:
                self.call_times.append(now)
                return 0
            else:
                oldest_call = min(self.call_times)
                wait_time = 60 - (now - oldest_call) + 0.5
                return wait_time

    def get(self, url):
        if url in self.cache:
            return self.cache[url]

        wait_time = self._check_rate_limit()
        if wait_time > 0:
            time.sleep(wait_time)

        max_retries = 3
        for retry in range(max_retries):
            try:
                response = requests.get(url)

                if response.status_code == 200:
                    data = response.json()
                    self.cache[url] = data
                    return data
                elif response.status_code == 429:
                    retry_delay = (retry + 1) * 3
                    print(f"Rate limit hit (429). Waiting {retry_delay}s before retry {retry+1}/{max_retries}")
                    time.sleep(retry_delay)
                else:
                    if retry < max_retries - 1:
                        print(f"API Error: {response.status_code} for {url}, retrying...")
                        time.sleep(1)
                    else:
                        print(f"API Error: {response.status_code} for {url}, giving up after {max_retries} attempts")
                        return None
            except Exception as e:
                if retry < max_retries - 1:
                    print(f"Request Exception: {e}, retrying...")
                    time.sleep(1)
                else:
                    print(f"Request Exception: {e}, giving up after {max_retries} attempts")
                    return None

        return None

    def get_esg_data(self, ticker):
        url = f"https://financialmodelingprep.com/api/v4/esg-environmental-social-governance-data?symbol={ticker}&apikey={self.api_key}"

        if url in self.cache:
            return self.cache[url]

        with self.lock:
            self.esg_request_counter += 1
            if self.esg_request_counter % 5 == 0:
                time.sleep(1)

        wait_time = self._check_rate_limit()
        if wait_time > 0:
            time.sleep(wait_time)

        try:
            response = requests.get(url)

            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    esg_time_series = []
                    for item in data:
                        esg_entry = {
                            'date': item.get('date', None),
                            'environmentalScore': item.get('environmentalScore', np.nan),
                            'socialScore': item.get('socialScore', np.nan),
                            'governanceScore': item.get('governanceScore', np.nan),
                            'ESGScore': item.get('ESGScore', np.nan)
                        }
                        esg_time_series.append(esg_entry)

                    self.cache[url] = esg_time_series
                    return esg_time_series
                return []
            else:
                print(f"Error fetching ESG data for {ticker}: {response.status_code}")
                return []
        except Exception as e:
            print(f"Exception while fetching ESG data for {ticker}: {e}")
            return []

    def get_esg_risk_rating(self, ticker):
        url = f"https://financialmodelingprep.com/api/v4/esg-environmental-social-governance-data-ratings?symbol={ticker}&apikey={self.api_key}"

        if url in self.cache:
            return self.cache[url]

        with self.lock:
            self.esg_request_counter += 1
            if self.esg_request_counter % 5 == 0:
                time.sleep(1)

        wait_time = self._check_rate_limit()
        if wait_time > 0:
            time.sleep(wait_time)

        try:
            response = requests.get(url)

            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    filtered_data = {
                        'ESGRiskRating': data[0].get('ESGRiskRating', np.nan)
                    }
                    self.cache[url] = filtered_data
                    return filtered_data
                return {}
            else:
                print(f"Error fetching ESG risk rating for {ticker}: {response.status_code}")
                return {}
        except Exception as e:
            print(f"Exception while fetching ESG risk rating for {ticker}: {e}")
            return {}

    def get_company_profile(self, ticker):
        url = f"https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey={self.api_key}"

        if url in self.cache:
            return self.cache[url]

        with self.lock:
            self.profile_request_counter += 1
            if self.profile_request_counter % 5 == 0:
                time.sleep(1)

        wait_time = self._check_rate_limit()
        if wait_time > 0:
            time.sleep(wait_time)

        try:
            response = requests.get(url)

            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    filtered_data = {
                        'sector': data[0].get('sector', None),
                        'ipoDate': data[0].get('ipoDate', None)
                    }
                    self.cache[url] = filtered_data
                    return filtered_data
                return {}
            else:
                print(f"Error fetching company profile for {ticker}: {response.status_code}")
                return {}
        except Exception as e:
            print(f"Exception while fetching company profile for {ticker}: {e}")
            return {}

api_manager = None

def create_panel_dataset(etf_ticker, api_key, start_year=2015, max_workers=10):
    global api_manager
    if api_manager is None:
        api_manager = ApiManager(api_key)

    print(f"Starting data collection for ETF: {etf_ticker}")

    holdings = get_etf_holdings(etf_ticker, api_key)
    if holdings is None or holdings.empty:
        print(f"Could not retrieve holdings for ETF: {etf_ticker}")
        return None

    print(f"Retrieved {len(holdings)} holdings from {etf_ticker}")

    all_tickers = []
    for i, row in holdings.iterrows():
        ticker = row['symbol'] if 'symbol' in row else row.get('asset', None)
        if ticker:
            all_tickers.append(ticker)

    print(f"Processing {len(all_tickers)} tickers")

    print("Fetching ESG data for each ticker...")
    esg_data = {}
    esg_risk_data = {}

    for ticker in tqdm(all_tickers, desc="Fetching ESG data"):
        ticker_esg_data = api_manager.get_esg_data(ticker)
        if ticker_esg_data:
            esg_data[ticker] = ticker_esg_data

        ticker_risk_data = api_manager.get_esg_risk_rating(ticker)
        if ticker_risk_data:
            esg_risk_data[ticker] = ticker_risk_data

    print(f"Retrieved ESG data for {len(esg_data)} tickers")
    print(f"Retrieved ESG risk data for {len(esg_risk_data)} tickers")

    esg_dfs = {}
    for ticker, esg_entries in esg_data.items():
        if esg_entries:
            df = pd.DataFrame(esg_entries)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                esg_dfs[ticker] = df

    print(f"Converted ESG data to DataFrames for {len(esg_dfs)} tickers")

    if esg_dfs:
        sample_keys = list(esg_dfs.keys())[:2]
        print(f"Sample ESG dataframes keys: {sample_keys}")
        for key in sample_keys:
            print(f"  {key} shape: {esg_dfs[key].shape}")
            print(f"  {key} columns: {esg_dfs[key].columns.tolist()}")
            if not esg_dfs[key].empty:
                print(f"  {key} sample data:")
                print(esg_dfs[key].head(2))

    print("Fetching company profile data for each ticker...")
    profile_data = {}

    for ticker in tqdm(all_tickers, desc="Fetching company profiles"):
        ticker_profile = api_manager.get_company_profile(ticker)
        if ticker_profile:
            profile_data[ticker] = ticker_profile

    print(f"Retrieved company profile data for {len(profile_data)} tickers")

    if profile_data:
        sample_keys = list(profile_data.keys())[:2]
        print(f"Sample profile data keys: {sample_keys}")
        for key in sample_keys:
            print(f"  {key}: {profile_data[key]}")

    print("Fetching financial data in parallel...")
    financial_data = get_financial_data_parallel(all_tickers, api_key, start_year, max_workers)

    print("Building panel dataset...")
    all_data = []

    for ticker in tqdm(all_tickers, desc="Building panel dataset"):
        if ticker not in financial_data or financial_data[ticker] is None:
            continue

        holding_rows = holdings[holdings['symbol'] == ticker]
        if holding_rows.empty:
            holding_rows = holdings[holdings['asset'] == ticker]

        if holding_rows.empty:
            continue

        holding_row = holding_rows.iloc[0]

        merged_df = financial_data[ticker]

        for idx, fin_row in merged_df.iterrows():
            quarter_year = fin_row.get('quarter_year', '')
            date = fin_row['date']

            record = {
                'ticker': ticker,
                'etf': etf_ticker,
                'quarter_year': quarter_year,
                'date': date,
                'weight': holding_row.get('weightPercentage', np.nan)
            }

            record.update({
                'environmentalScore': np.nan,
                'socialScore': np.nan,
                'governanceScore': np.nan,
                'ESGScore': np.nan,
                'ESGRiskRating': np.nan
            })

            if ticker in esg_dfs:
                esg_df = esg_dfs[ticker]

                matching_esg = esg_df[esg_df['date'] == date]

                if not matching_esg.empty:
                    exact_esg = matching_esg.iloc[0]

                    record.update({
                        'environmentalScore': exact_esg.get('environmentalScore', np.nan),
                        'socialScore': exact_esg.get('socialScore', np.nan),
                        'governanceScore': exact_esg.get('governanceScore', np.nan),
                        'ESGScore': exact_esg.get('ESGScore', np.nan)
                    })

            if ticker in esg_risk_data:
                record.update({
                    'ESGRiskRating': esg_risk_data[ticker].get('ESGRiskRating', np.nan)
                })

            record['sector'] = None
            record['ipoDate'] = None
            record['companyAge'] = np.nan

            if ticker in profile_data:
                record['sector'] = profile_data[ticker].get('sector', None)
                record['ipoDate'] = profile_data[ticker].get('ipoDate', None)

                if record['ipoDate']:
                    try:
                        ipo_date = pd.to_datetime(record['ipoDate'])
                        record_date = pd.to_datetime(date)
                        age_days = (record_date - ipo_date).days
                        record['companyAge'] = age_days / 365.25
                    except Exception as e:
                        print(f"Error calculating age for {ticker} at date {date}: {e}")

            for col in merged_df.columns:
                if col not in ['date', 'quarter', 'quarter_year']:
                    record[col] = fin_row[col]

            all_data.append(record)

    panel_df = pd.DataFrame(all_data)

    print(f"Final panel dataset contains {len(panel_df)} records across {panel_df['ticker'].nunique()} companies")
    print(f"Columns in final dataset: {panel_df.columns.tolist()}")

    total_rows = len(panel_df)

    sector_avail = panel_df['sector'].notna().sum()
    print(f"Sector data available for {sector_avail} out of {total_rows} records ({sector_avail/total_rows*100:.2f}%)")

    age_avail = panel_df['companyAge'].notna().sum()
    print(f"Company age data available for {age_avail} out of {total_rows} records ({age_avail/total_rows*100:.2f}%)")

    return panel_df

def get_etf_holdings(etf_ticker, api_key):
    url = f"https://financialmodelingprep.com/api/v3/etf-holder/{etf_ticker}?apikey={api_key}"

    data = api_manager.get(url)
    if data:
        df = pd.DataFrame(data)
        print(f"ETF Holdings columns: {df.columns.tolist()}")

        if 'asset' in df.columns and 'symbol' not in df.columns:
            df['symbol'] = df['asset']

        return df

    print(f"Error fetching ETF holdings")
    return None

def get_financial_data_parallel(tickers, api_key, start_year, max_workers=10):
    results = {}

    def process_ticker(ticker):
        try:
            income_url = f"https://financialmodelingprep.com/api/v3/income-statement/{ticker}?period=quarter&apikey={api_key}"
            income_data = api_manager.get(income_url)

            if not income_data:
                print(f"No income data for {ticker}")
                return ticker, None

            balance_url = f"https://financialmodelingprep.com/api/v3/balance-sheet-statement/{ticker}?period=quarter&apikey={api_key}"
            balance_data = api_manager.get(balance_url)

            if not balance_data:
                print(f"No balance sheet data for {ticker}")
                return ticker, None

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
                return ticker, None

            income_df = filter_by_year(income_df, start_year)
            balance_df = filter_by_year(balance_df, start_year)

            if income_df is None or balance_df is None:
                return ticker, None

            merged_df = merge_financial_data(income_df, balance_df)

            return ticker, merged_df
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            return ticker, None

    batch_size = 50

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

def filter_by_year(df, start_year):
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None

    df['year'] = pd.to_datetime(df['date']).dt.year
    filtered_df = df[df['year'] >= start_year].copy()
    filtered_df.drop('year', axis=1, inplace=True)

    if filtered_df.empty:
        return None

    return filtered_df

def merge_financial_data(income_df, balance_df):
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

def run_etf_analysis(etf_ticker, api_key, start_year=2015, max_workers=5, output_path=None):
    global api_manager
    if api_manager is None:
        api_manager = ApiManager(api_key)

    try:
        panel_data = create_panel_dataset(etf_ticker, api_key, start_year, max_workers)

        if panel_data is not None and not panel_data.empty:
            if output_path is None:
                current_date = datetime.now().strftime("%Y%m%d")
                output_path = "panel_data.csv"

            panel_data.to_csv(output_path, index=False)
            print(f"Panel dataset saved to {output_path}")
            return panel_data
        else:
            print("Failed to create panel dataset")
            return None
    except Exception as e:
        print(f"Error in run_etf_analysis: {e}")
        return None
