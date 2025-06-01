import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from typing import Dict, List, Optional
from datetime import datetime

from .api_client import ApiManager
from ..utils import PROFILE_COLUMNS, format_number


def collect_company_profiles(tickers: List[str], api_manager: ApiManager) -> Dict[str, Dict]:
    print("Fetching company profile data for each ticker...")
    profile_data = {}

    for ticker in tqdm(tickers, desc="Fetching company profiles"):
        ticker_profile = api_manager.get_company_profile(ticker)
        if ticker_profile:
            profile_data[ticker] = ticker_profile

    print(f"Retrieved company profile data for {len(profile_data)} tickers")

    if profile_data:
        sample_keys = list(profile_data.keys())[:2]
        print(f"Sample profile data keys: {sample_keys}")
        for key in sample_keys:
            print(f"  {key}: {profile_data[key]}")

    return profile_data


def calculate_company_age(ipo_date_str: Optional[str], reference_date: pd.Timestamp) -> Optional[float]:
    if not ipo_date_str:
        return None
    
    try:
        ipo_date = pd.to_datetime(ipo_date_str)
        age_days = (reference_date - ipo_date).days
        return age_days / 365.25
    except Exception:
        return None


def get_profile_data_for_ticker(ticker: str, profile_data: Dict[str, Dict], reference_date: pd.Timestamp) -> Dict[str, any]:
    profile_record = {
        'sector': None,
        'ipoDate': None,
        'companyAge': np.nan
    }
    
    if ticker in profile_data:
        ticker_profile = profile_data[ticker]
        profile_record['sector'] = ticker_profile.get('sector', None)
        profile_record['ipoDate'] = ticker_profile.get('ipoDate', None)
        
        if profile_record['ipoDate']:
            profile_record['companyAge'] = calculate_company_age(
                profile_record['ipoDate'], 
                reference_date
            )
    
    return profile_record


def validate_profile_data(profile_data: Dict[str, Dict]) -> Dict[str, any]:
    validation_stats = {
        'total_tickers': len(profile_data),
        'field_coverage': {},
        'sector_distribution': {},
        'ipo_date_analysis': {},
        'warnings': []
    }
    
    if not profile_data:
        validation_stats['warnings'].append("No profile data available for validation")
        return validation_stats
    
    field_stats = {}
    for field in ['sector', 'ipoDate']:
        non_null_count = sum(1 for profile in profile_data.values() if profile.get(field) is not None)
        coverage_pct = (non_null_count / len(profile_data)) * 100
        field_stats[field] = {
            'available_count': non_null_count,
            'coverage_percentage': coverage_pct
        }
    
    validation_stats['field_coverage'] = field_stats
    
    sectors = {}
    for profile in profile_data.values():
        sector = profile.get('sector')
        if sector:
            sectors[sector] = sectors.get(sector, 0) + 1
    
    validation_stats['sector_distribution'] = {
        'unique_sectors': len(sectors),
        'top_sectors': dict(sorted(sectors.items(), key=lambda x: x[1], reverse=True)[:10])
    }
    
    ipo_dates = []
    invalid_dates = 0
    for profile in profile_data.values():
        ipo_date_str = profile.get('ipoDate')
        if ipo_date_str:
            try:
                ipo_date = pd.to_datetime(ipo_date_str)
                ipo_dates.append(ipo_date)
            except:
                invalid_dates += 1
    
    if ipo_dates:
        validation_stats['ipo_date_analysis'] = {
            'valid_dates': len(ipo_dates),
            'invalid_dates': invalid_dates,
            'earliest_ipo': min(ipo_dates),
            'latest_ipo': max(ipo_dates),
            'median_ipo': pd.Series(ipo_dates).median()
        }
    
    if field_stats['sector']['coverage_percentage'] < 80:
        validation_stats['warnings'].append(
            f"Low sector coverage: {field_stats['sector']['coverage_percentage']:.1f}%"
        )
    
    if field_stats['ipoDate']['coverage_percentage'] < 70:
        validation_stats['warnings'].append(
            f"Low IPO date coverage: {field_stats['ipoDate']['coverage_percentage']:.1f}%"
        )
    
    if invalid_dates > 0:
        validation_stats['warnings'].append(f"Invalid IPO date formats: {invalid_dates} records")
    
    return validation_stats


def analyze_sector_distribution(profile_data: Dict[str, Dict]) -> Dict[str, any]:
    if not profile_data:
        return {'error': 'No profile data available for sector analysis'}
    
    sector_counts = {}
    missing_sector_count = 0
    
    for ticker, profile in profile_data.items():
        sector = profile.get('sector')
        if sector:
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
        else:
            missing_sector_count += 1
    
    total_tickers = len(profile_data)
    
    analysis = {
        'total_tickers': total_tickers,
        'tickers_with_sector': len(profile_data) - missing_sector_count,
        'missing_sector_count': missing_sector_count,
        'coverage_rate': ((total_tickers - missing_sector_count) / total_tickers) * 100,
        'unique_sectors': len(sector_counts),
        'sector_distribution': dict(sorted(sector_counts.items(), key=lambda x: x[1], reverse=True))
    }
    
    if len(sector_counts) > 0:
        analysis['largest_sector'] = max(sector_counts.items(), key=lambda x: x[1])
        analysis['smallest_sector'] = min(sector_counts.items(), key=lambda x: x[1])
        analysis['avg_tickers_per_sector'] = sum(sector_counts.values()) / len(sector_counts)
    
    return analysis


def analyze_company_ages(profile_data: Dict[str, Dict], reference_date: Optional[pd.Timestamp] = None) -> Dict[str, any]:
    if reference_date is None:
        reference_date = pd.Timestamp.now()
    
    if not profile_data:
        return {'error': 'No profile data available for age analysis'}
    
    ages = []
    invalid_dates = []
    missing_dates = []
    
    for ticker, profile in profile_data.items():
        ipo_date_str = profile.get('ipoDate')
        if not ipo_date_str:
            missing_dates.append(ticker)
            continue
        
        age = calculate_company_age(ipo_date_str, reference_date)
        if age is not None:
            ages.append({
                'ticker': ticker,
                'age_years': age,
                'ipo_date': ipo_date_str
            })
        else:
            invalid_dates.append(ticker)
    
    analysis = {
        'total_tickers': len(profile_data),
        'valid_ages': len(ages),
        'missing_ipo_dates': len(missing_dates),
        'invalid_ipo_dates': len(invalid_dates),
        'coverage_rate': (len(ages) / len(profile_data)) * 100 if len(profile_data) > 0 else 0
    }
    
    if ages:
        age_values = [item['age_years'] for item in ages]
        analysis.update({
            'min_age': min(age_values),
            'max_age': max(age_values),
            'mean_age': sum(age_values) / len(age_values),
            'median_age': sorted(age_values)[len(age_values) // 2],
            'companies_under_5_years': sum(1 for age in age_values if age < 5),
            'companies_over_20_years': sum(1 for age in age_values if age > 20)
        })
        
        analysis['age_distribution'] = {
            'under_5_years': sum(1 for age in age_values if age < 5),
            '5_to_10_years': sum(1 for age in age_values if 5 <= age < 10),
            '10_to_20_years': sum(1 for age in age_values if 10 <= age < 20),
            'over_20_years': sum(1 for age in age_values if age >= 20)
        }
    
    return analysis


def print_profile_data_summary(profile_data: Dict[str, Dict]):
    print(f"\nCompany Profile Data Summary:")
    print(f"Total profiles collected: {format_number(len(profile_data))}")
    
    if not profile_data:
        return {'total_profiles': 0}
    
    sector_count = sum(1 for profile in profile_data.values() if profile.get('sector'))
    ipo_date_count = sum(1 for profile in profile_data.values() if profile.get('ipoDate'))
    
    sector_pct = (sector_count / len(profile_data)) * 100
    ipo_pct = (ipo_date_count / len(profile_data)) * 100
    
    print(f"Sector data available: {format_number(sector_count)} ({sector_pct:.1f}%)")
    print(f"IPO date data available: {format_number(ipo_date_count)} ({ipo_pct:.1f}%)")
    
    sector_analysis = analyze_sector_distribution(profile_data)
    if 'error' not in sector_analysis:
        print(f"Unique sectors: {sector_analysis['unique_sectors']}")
        if sector_analysis['sector_distribution']:
            top_3_sectors = list(sector_analysis['sector_distribution'].items())[:3]
            print("Top 3 sectors:")
            for sector, count in top_3_sectors:
                pct = (count / len(profile_data)) * 100
                print(f"  {sector}: {count} companies ({pct:.1f}%)")
    
    age_analysis = analyze_company_ages(profile_data)
    if 'error' not in age_analysis and age_analysis['valid_ages'] > 0:
        print(f"Company age statistics (years):")
        print(f"  Mean: {age_analysis['mean_age']:.1f}")
        print(f"  Range: {age_analysis['min_age']:.1f} - {age_analysis['max_age']:.1f}")
        print(f"  Companies under 5 years: {age_analysis['companies_under_5_years']}")
        print(f"  Companies over 20 years: {age_analysis['companies_over_20_years']}")
    
    return {
        'total_profiles': len(profile_data),
        'sector_coverage': sector_pct,
        'ipo_coverage': ipo_pct,
        'unique_sectors': sector_analysis.get('unique_sectors', 0),
        'age_stats': age_analysis if 'error' not in age_analysis else {}
    }
