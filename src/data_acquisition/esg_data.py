import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from typing import Dict, List, Optional

from .api_client import ApiManager
from ..utils import ESG_COLUMNS, ESG_RISK_COLUMNS, format_number


def collect_esg_data(tickers: List[str], api_manager: ApiManager) -> Dict[str, List]:
    print("Fetching ESG data for each ticker...")
    esg_data = {}
    
    for ticker in tqdm(tickers, desc="Fetching ESG data"):
        ticker_esg_data = api_manager.get_esg_data(ticker)
        if ticker_esg_data:
            esg_data[ticker] = ticker_esg_data
    
    print(f"Retrieved ESG data for {len(esg_data)} tickers")
    return esg_data


def collect_esg_risk_data(tickers: List[str], api_manager: ApiManager) -> Dict[str, Dict]:
    esg_risk_data = {}
    
    for ticker in tqdm(tickers, desc="Fetching ESG risk data"):
        ticker_risk_data = api_manager.get_esg_risk_rating(ticker)
        if ticker_risk_data:
            esg_risk_data[ticker] = ticker_risk_data
    
    print(f"Retrieved ESG risk data for {len(esg_risk_data)} tickers")
    return esg_risk_data


def convert_esg_to_dataframes(esg_data: Dict[str, List]) -> Dict[str, pd.DataFrame]:
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
    
    return esg_dfs


def validate_esg_data(esg_dfs: Dict[str, pd.DataFrame]) -> Dict[str, any]:
    validation_stats = {
        'total_tickers': len(esg_dfs),
        'column_coverage': {},
        'date_range': {},
        'missing_patterns': {},
        'warnings': []
    }
    
    if not esg_dfs:
        validation_stats['warnings'].append("No ESG data available for validation")
        return validation_stats
    
    all_observations = 0
    date_ranges = []
    
    for column in ESG_COLUMNS:
        column_stats = {
            'tickers_with_data': 0,
            'total_observations': 0,
            'total_missing': 0,
            'avg_observations_per_ticker': 0
        }
        
        for ticker, df in esg_dfs.items():
            if column in df.columns:
                non_missing = df[column].notna().sum()
                if non_missing > 0:
                    column_stats['tickers_with_data'] += 1
                    column_stats['total_observations'] += non_missing
                    column_stats['total_missing'] += df[column].isna().sum()
        
        if column_stats['tickers_with_data'] > 0:
            column_stats['avg_observations_per_ticker'] = column_stats['total_observations'] / column_stats['tickers_with_data']
        
        validation_stats['column_coverage'][column] = column_stats
    
    for ticker, df in esg_dfs.items():
        all_observations += len(df)
        if 'date' in df.columns and not df.empty:
            date_ranges.append({
                'ticker': ticker,
                'start_date': df['date'].min(),
                'end_date': df['date'].max(),
                'observations': len(df)
            })
    
    if date_ranges:
        validation_stats['date_range'] = {
            'earliest_date': min(dr['start_date'] for dr in date_ranges),
            'latest_date': max(dr['end_date'] for dr in date_ranges),
            'total_observations': all_observations
        }
    
    missing_all_esg = []
    for ticker, df in esg_dfs.items():
        esg_cols_present = [col for col in ESG_COLUMNS if col in df.columns]
        if esg_cols_present:
            all_missing = df[esg_cols_present].isna().all().all()
            if all_missing:
                missing_all_esg.append(ticker)
    
    validation_stats['missing_patterns'] = {
        'tickers_missing_all_esg': len(missing_all_esg),
        'sample_missing_tickers': missing_all_esg[:10] if missing_all_esg else []
    }
    
    low_coverage_columns = []
    for column, stats in validation_stats['column_coverage'].items():
        coverage_rate = stats['tickers_with_data'] / validation_stats['total_tickers']
        if coverage_rate < 0.5:
            low_coverage_columns.append(f"{column} ({coverage_rate:.1%})")
    
    if low_coverage_columns:
        validation_stats['warnings'].append(f"Low coverage columns: {', '.join(low_coverage_columns)}")
    
    return validation_stats


def print_esg_data_summary(esg_data: Dict[str, List], esg_risk_data: Dict[str, Dict], esg_dfs: Dict[str, pd.DataFrame]):
    print(f"\nESG Data Collection Summary:")
    print(f"ESG scores collected for: {format_number(len(esg_data))} tickers")
    print(f"ESG risk ratings collected for: {format_number(len(esg_risk_data))} tickers")
    print(f"Valid ESG DataFrames: {format_number(len(esg_dfs))} tickers")
    
    if esg_dfs:
        total_esg_observations = sum(len(df) for df in esg_dfs.values())
        print(f"Total ESG observations: {format_number(total_esg_observations)}")
        
        column_coverage = {}
        for column in ESG_COLUMNS:
            tickers_with_column = sum(1 for df in esg_dfs.values() if column in df.columns and df[column].notna().any())
            coverage_pct = (tickers_with_column / len(esg_dfs)) * 100
            column_coverage[column] = coverage_pct
            print(f"  {column}: {tickers_with_column}/{len(esg_dfs)} tickers ({coverage_pct:.1f}%)")
        
        return {
            'esg_scores_tickers': len(esg_data),
            'esg_risk_tickers': len(esg_risk_data),
            'valid_dataframes': len(esg_dfs),
            'total_observations': total_esg_observations,
            'column_coverage': column_coverage
        }
    
    return {
        'esg_scores_tickers': len(esg_data),
        'esg_risk_tickers': len(esg_risk_data),
        'valid_dataframes': 0,
        'total_observations': 0
    }


def get_esg_data_for_ticker(ticker: str, esg_dfs: Dict[str, pd.DataFrame], esg_risk_data: Dict[str, Dict], date: pd.Timestamp) -> Dict[str, any]:
    esg_record = {
        'environmentalScore': np.nan,
        'socialScore': np.nan,
        'governanceScore': np.nan,
        'ESGScore': np.nan,
        'ESGRiskRating': np.nan
    }
    
    if ticker in esg_dfs:
        esg_df = esg_dfs[ticker]
        matching_esg = esg_df[esg_df['date'] == date]
        
        if not matching_esg.empty:
            exact_esg = matching_esg.iloc[0]
            esg_record.update({
                'environmentalScore': exact_esg.get('environmentalScore', np.nan),
                'socialScore': exact_esg.get('socialScore', np.nan),
                'governanceScore': exact_esg.get('governanceScore', np.nan),
                'ESGScore': exact_esg.get('ESGScore', np.nan)
            })
    
    if ticker in esg_risk_data:
        esg_record.update({
            'ESGRiskRating': esg_risk_data[ticker].get('ESGRiskRating', np.nan)
        })
    
    return esg_record


def analyze_esg_missing_patterns(esg_dfs: Dict[str, pd.DataFrame]) -> Dict[str, any]:
    if not esg_dfs:
        return {'error': 'No ESG data available for analysis'}
    
    patterns = {
        'consecutive_missing': {},
        'total_missing_by_ticker': {},
        'quarterly_coverage': {},
        'recommendations': []
    }
    
    for ticker, df in esg_dfs.items():
        if 'environmentalScore' in df.columns:
            missing_mask = df['environmentalScore'].isna()
            
            consecutive_counts = []
            current_count = 0
            for is_missing in missing_mask:
                if is_missing:
                    current_count += 1
                else:
                    if current_count > 0:
                        consecutive_counts.append(current_count)
                    current_count = 0
            if current_count > 0:
                consecutive_counts.append(current_count)
            
            patterns['consecutive_missing'][ticker] = {
                'max_consecutive': max(consecutive_counts) if consecutive_counts else 0,
                'missing_streaks': len(consecutive_counts),
                'total_missing': missing_mask.sum(),
                'total_observations': len(df)
            }
    
    high_missing_tickers = []
    for ticker, stats in patterns['consecutive_missing'].items():
        missing_rate = stats['total_missing'] / stats['total_observations']
        patterns['total_missing_by_ticker'][ticker] = missing_rate
        
        if missing_rate > 0.7:
            high_missing_tickers.append(ticker)
        
        if stats['max_consecutive'] >= 8:
            patterns['recommendations'].append(f"Consider removing {ticker}: {stats['max_consecutive']} consecutive missing periods")
    
    if high_missing_tickers:
        patterns['recommendations'].append(f"High missing rate tickers (>70%): {len(high_missing_tickers)} tickers")
    
    overall_missing_rate = sum(patterns['total_missing_by_ticker'].values()) / len(patterns['total_missing_by_ticker']) if patterns['total_missing_by_ticker'] else 0
    patterns['overall_stats'] = {
        'avg_missing_rate': overall_missing_rate,
        'high_missing_tickers': len(high_missing_tickers),
        'tickers_analyzed': len(patterns['consecutive_missing'])
    }
    
    return patterns
