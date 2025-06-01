import pandas as pd
from typing import Optional, List

from .api_client import ApiManager
from ..utils import format_number


def get_etf_holdings(etf_ticker: str, api_manager: ApiManager) -> Optional[pd.DataFrame]:
    url = f"https://financialmodelingprep.com/api/v3/etf-holder/{etf_ticker}?apikey={api_manager.api_key}"

    data = api_manager.get(url)
    if data:
        df = pd.DataFrame(data)
        print(f"ETF Holdings columns: {df.columns.tolist()}")

        if 'asset' in df.columns and 'symbol' not in df.columns:
            df['symbol'] = df['asset']

        return df

    print(f"Error fetching ETF holdings")
    return None


def validate_etf_holdings(holdings_df: pd.DataFrame, etf_ticker: str) -> dict:
    validation_stats = {
        'etf_ticker': etf_ticker,
        'total_holdings': len(holdings_df),
        'validation_passed': True,
        'warnings': [],
        'errors': []
    }
    
    if holdings_df.empty:
        validation_stats['validation_passed'] = False
        validation_stats['errors'].append("Holdings dataframe is empty")
        return validation_stats
    
    required_columns = ['symbol']
    missing_columns = [col for col in required_columns if col not in holdings_df.columns]
    if missing_columns:
        validation_stats['validation_passed'] = False
        validation_stats['errors'].append(f"Missing required columns: {missing_columns}")
    
    if 'symbol' in holdings_df.columns:
        null_symbols = holdings_df['symbol'].isna().sum()
        if null_symbols > 0:
            validation_stats['warnings'].append(f"Found {null_symbols} holdings with null symbols")
        
        duplicate_symbols = holdings_df['symbol'].duplicated().sum()
        if duplicate_symbols > 0:
            validation_stats['warnings'].append(f"Found {duplicate_symbols} duplicate symbols")
    
    if 'weightPercentage' in holdings_df.columns:
        weight_stats = {
            'has_weights': True,
            'null_weights': holdings_df['weightPercentage'].isna().sum(),
            'total_weight': holdings_df['weightPercentage'].sum(),
            'max_weight': holdings_df['weightPercentage'].max(),
            'min_weight': holdings_df['weightPercentage'].min()
        }
        
        if weight_stats['total_weight'] > 105 or weight_stats['total_weight'] < 95:
            validation_stats['warnings'].append(
                f"Total weight percentage unusual: {weight_stats['total_weight']:.1f}%"
            )
        
        validation_stats['weight_stats'] = weight_stats
    else:
        validation_stats['weight_stats'] = {'has_weights': False}
        validation_stats['warnings'].append("No weight percentage information available")
    
    return validation_stats


def extract_tickers_from_holdings(holdings_df: pd.DataFrame) -> List[str]:
    if holdings_df.empty:
        return []
    
    all_tickers = []
    
    for i, row in holdings_df.iterrows():
        ticker = row['symbol'] if 'symbol' in row else row.get('asset', None)
        if ticker and pd.notna(ticker):
            all_tickers.append(str(ticker).strip())
    
    unique_tickers = list(set(all_tickers))
    return unique_tickers


def analyze_holdings_composition(holdings_df: pd.DataFrame) -> dict:
    if holdings_df.empty:
        return {'error': 'Empty holdings dataframe'}
    
    analysis = {
        'total_holdings': len(holdings_df),
        'unique_tickers': len(holdings_df['symbol'].unique()) if 'symbol' in holdings_df.columns else 0
    }
    
    if 'weightPercentage' in holdings_df.columns:
        weights = holdings_df['weightPercentage'].dropna()
        if len(weights) > 0:
            analysis.update({
                'weight_analysis': {
                    'total_weight': weights.sum(),
                    'mean_weight': weights.mean(),
                    'median_weight': weights.median(),
                    'std_weight': weights.std(),
                    'top_10_weight': weights.nlargest(10).sum(),
                    'concentration_ratio': weights.nlargest(5).sum()  # Top 5 concentration
                }
            })
            
            # Weight distribution
            analysis['weight_distribution'] = {
                'large_holdings_5_plus': (weights >= 5.0).sum(),
                'medium_holdings_1_to_5': ((weights >= 1.0) & (weights < 5.0)).sum(),
                'small_holdings_under_1': (weights < 1.0).sum()
            }
    
    if 'sector' in holdings_df.columns:
        sector_weights = holdings_df.groupby('sector')['weightPercentage'].sum().sort_values(ascending=False)
        analysis['sector_allocation'] = sector_weights.to_dict()
    
    return analysis


def print_etf_holdings_summary(holdings_df: pd.DataFrame, etf_ticker: str, validation_stats: dict):
    print(f"\nETF Holdings Summary for {etf_ticker}:")
    print(f"Total holdings: {format_number(len(holdings_df))}")
    
    if validation_stats['validation_passed']:
        print("✓ Holdings validation passed")
    else:
        print("✗ Holdings validation failed")
        for error in validation_stats['errors']:
            print(f"  Error: {error}")
    
    for warning in validation_stats['warnings']:
        print(f"  Warning: {warning}")
    
    if 'weight_stats' in validation_stats and validation_stats['weight_stats']['has_weights']:
        weight_stats = validation_stats['weight_stats']
        print(f"Weight information:")
        print(f"  Total weight: {weight_stats['total_weight']:.1f}%")
        print(f"  Largest holding: {weight_stats['max_weight']:.2f}%")
        print(f"  Smallest holding: {weight_stats['min_weight']:.2f}%")
        print(f"  Holdings with null weights: {weight_stats['null_weights']}")
    
    composition = analyze_holdings_composition(holdings_df)
    if 'error' not in composition and 'weight_analysis' in composition:
        weight_analysis = composition['weight_analysis']
        print(f"Concentration analysis:")
        print(f"  Top 5 holdings weight: {weight_analysis['concentration_ratio']:.1f}%")
        print(f"  Top 10 holdings weight: {weight_analysis['top_10_weight']:.1f}%")
        
        if 'weight_distribution' in composition:
            dist = composition['weight_distribution']
            print(f"Holdings distribution:")
            print(f"  Large holdings (≥5%): {dist['large_holdings_5_plus']}")
            print(f"  Medium holdings (1-5%): {dist['medium_holdings_1_to_5']}")
            print(f"  Small holdings (<1%): {dist['small_holdings_under_1']}")
    
    return {
        'etf_ticker': etf_ticker,
        'total_holdings': len(holdings_df),
        'validation_passed': validation_stats['validation_passed'],
        'composition_analysis': composition
    }


def filter_holdings_by_weight(holdings_df: pd.DataFrame, min_weight: float = 0.0, max_weight: float = 100.0) -> pd.DataFrame:
    if 'weightPercentage' not in holdings_df.columns:
        print("Warning: No weight percentage column found, returning all holdings")
        return holdings_df.copy()
    
    filtered_df = holdings_df[
        (holdings_df['weightPercentage'] >= min_weight) & 
        (holdings_df['weightPercentage'] <= max_weight)
    ].copy()
    
    removed_count = len(holdings_df) - len(filtered_df)
    if removed_count > 0:
        print(f"Filtered out {removed_count} holdings outside weight range {min_weight}%-{max_weight}%")
    
    return filtered_df


def get_top_holdings(holdings_df: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    if 'weightPercentage' not in holdings_df.columns:
        print("Warning: No weight percentage column found, returning first N holdings")
        return holdings_df.head(top_n)
    
    top_holdings = holdings_df.nlargest(top_n, 'weightPercentage')
    print(f"Selected top {len(top_holdings)} holdings by weight")
    
    if len(top_holdings) > 0:
        total_weight = top_holdings['weightPercentage'].sum()
        print(f"Top {len(top_holdings)} holdings represent {total_weight:.1f}% of the ETF")
    
    return top_holdings
