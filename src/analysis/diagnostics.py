import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import random
import warnings
from typing import List, Optional, Dict, Any, Union, Tuple

from ...utils import (
    DEFAULT_TICKER_COLUMN, print_section_header, print_subsection_header,
    format_number, print_dataset_info, print_processing_stats
)

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100


class PanelDiagnostics:
    def __init__(self, file_path: str = None, df: pd.DataFrame = None, 
                 categorical_columns: Optional[List[str]] = None, auto_detect: bool = True,
                 ticker_column: str = DEFAULT_TICKER_COLUMN, date_column: str = 'date'):
        
        if file_path and df is None:
            self.df = self._load_data(file_path)
        elif df is not None:
            self.df = df.copy()
        else:
            raise ValueError("Either file_path or df must be provided")
        
        self.ticker_column = ticker_column
        self.date_column = date_column
        self.categorical_columns = self._detect_categorical_columns(categorical_columns, auto_detect)
        self.has_categorical = len(self.categorical_columns) > 0
        
    def _detect_categorical_columns(self, specified_cols: Optional[List[str]], auto_detect: bool) -> List[str]:
        if specified_cols:
            return [col for col in specified_cols if col in self.df.columns]
        
        if not auto_detect:
            return []
        
        categorical_indicators = ['sector', 'industry', 'category', '_num', '_code', 'type', 'group', 'class']
        detected_cols = []
        
        for col in self.df.columns:
            if any(indicator in col.lower() for indicator in categorical_indicators):
                if pd.api.types.is_numeric_dtype(self.df[col]) or self.df[col].dtype == 'object':
                    detected_cols.append(col)
        
        return detected_cols

    def _load_data(self, file_path: str) -> pd.DataFrame:
        df = pd.read_csv(file_path)
        
        if self.date_column in df.columns:
            df[self.date_column] = pd.to_datetime(df[self.date_column])
        
        if self.ticker_column in df.columns:
            df = df.sort_values([self.ticker_column, self.date_column])
        
        print_dataset_info(df, "Loaded Dataset")
        
        if self.has_categorical:
            for col in self.categorical_columns:
                if col in df.columns:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        print(f"Categorical column '{col}': {format_number(df[col].nunique())} unique values")
                        print(f"Values: {sorted(df[col].unique())}")
                    else:
                        print(f"Warning: {col} column is not numeric. Assuming categorical labels.")
        
        return df

    def sample_companies(self, n: int = 5, strategy: str = 'auto') -> List[str]:
        if strategy == 'auto':
            strategy = 'balanced' if self.has_categorical else 'random'
        
        if strategy == 'balanced' and self.has_categorical:
            return self._balanced_sampling(n)
        else:
            return self._random_sampling(n)

    def _balanced_sampling(self, n: int) -> List[str]:
        primary_categorical = self.categorical_columns[0]
        categories = self.df[primary_categorical].unique()
        sample = []
        
        for category in categories[:min(n, len(categories))]:
            companies_in_category = self.df[self.df[primary_categorical] == category][self.ticker_column].unique()
            if len(companies_in_category) > 0:
                sample.append(random.choice(companies_in_category))
        
        if len(sample) < n:
            remaining_companies = set(self.df[self.ticker_column].unique()) - set(sample)
            additional_needed = min(n - len(sample), len(remaining_companies))
            sample.extend(random.sample(list(remaining_companies), additional_needed))
        
        return sample[:n]

    def _random_sampling(self, n: int) -> List[str]:
        unique_companies = self.df[self.ticker_column].unique()
        return random.sample(list(unique_companies), min(n, len(unique_companies)))

    def _get_category_label(self, ticker: str) -> str:
        if not self.has_categorical:
            return ""
        
        primary_categorical = self.categorical_columns[0]
        ticker_data = self.df[self.df[self.ticker_column] == ticker]
        
        if ticker_data.empty:
            return ""
        
        category_value = ticker_data[primary_categorical].iloc[0]
        
        if pd.api.types.is_numeric_dtype(self.df[primary_categorical]):
            return f"{primary_categorical.replace('_', ' ').title()} {int(category_value)}"
        else:
            return str(category_value)

    def plot_company_time_series(self, companies: List[str], column: str, title: str, ylabel: str, 
                                normalize: bool = False, plot_type: str = 'line', verbose: bool = True) -> None:
        if verbose:
            print_subsection_header(f"Company Time Series: {column}")
        
        plt.figure(figsize=(14, 8))
        
        for company in companies:
            company_data = self.df[self.df[self.ticker_column] == company].sort_values(self.date_column)
            
            if len(company_data) == 0:
                if verbose:
                    print(f"No data for company {company}")
                continue
            
            values = company_data[column].values
            
            if normalize and len(values) > 0:
                values = values / values[0] if values[0] != 0 else values
            
            if self.has_categorical:
                category_label = self._get_category_label(company)
                label = f"{company} ({category_label})" if category_label else company
            else:
                label = company
            
            if plot_type == 'line':
                plt.plot(company_data[self.date_column], values, marker='o', label=label)
            else:
                plt.bar(company_data[self.date_column], values, alpha=0.7, label=label)
        
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.xticks(rotation=45)
        plt.show()

    def plot_category_time_series(self, column: str, title: str, ylabel: str, 
                                 aggregation: str = 'sum', verbose: bool = True) -> None:
        if not self.has_categorical:
            if verbose:
                print("No categorical variables detected. Skipping category-level analysis.")
            return
        
        if verbose:
            print_subsection_header(f"Category Time Series: {column}")
        
        primary_categorical = self.categorical_columns[0]
        
        if aggregation == 'sum':
            category_data = self.df.groupby([primary_categorical, self.date_column])[column].sum().reset_index()
        elif aggregation == 'mean':
            category_data = self.df.groupby([primary_categorical, self.date_column])[column].mean().reset_index()
        elif aggregation == 'median':
            category_data = self.df.groupby([primary_categorical, self.date_column])[column].median().reset_index()
        else:
            raise ValueError("Aggregation must be 'sum', 'mean', or 'median'")
        
        plt.figure(figsize=(14, 10))
        
        for category in sorted(category_data[primary_categorical].unique()):
            category_subset = category_data[category_data[primary_categorical] == category]
            
            if pd.api.types.is_numeric_dtype(self.df[primary_categorical]):
                label = f"{primary_categorical.replace('_', ' ').title()} {int(category)}"
            else:
                label = str(category)
            
            plt.plot(category_subset[self.date_column], category_subset[column], marker='o', label=label)
        
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel(ylabel)
        plt.legend(loc='best', bbox_to_anchor=(1, 1))
        plt.grid(True)
        plt.tight_layout()
        plt.xticks(rotation=45)
        plt.show()

    def plot_overall_time_series(self, column: str, title: str, ylabel: str, 
                                aggregation: str = 'sum', verbose: bool = True) -> None:
        if verbose:
            print_subsection_header(f"Overall Time Series: {column}")
        
        if aggregation == 'sum':
            agg_data = self.df.groupby(self.date_column)[column].sum().reset_index()
            agg_label = 'Sum'
        elif aggregation == 'mean':
            agg_data = self.df.groupby(self.date_column)[column].mean().reset_index()
            agg_label = 'Mean'
        elif aggregation == 'median':
            agg_data = self.df.groupby(self.date_column)[column].median().reset_index()
            agg_label = 'Median'
        else:
            raise ValueError("Aggregation must be 'sum', 'mean', or 'median'")
        
        plt.figure(figsize=(14, 8))
        plt.plot(agg_data[self.date_column], agg_data[column], marker='o', linewidth=2, label=agg_label)
        
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.xticks(rotation=45)
        plt.show()

    def plot_mean_time_series(self, column: str, title: str, ylabel: str, verbose: bool = True) -> None:
        if verbose:
            print_subsection_header(f"Mean Time Series: {column}")
        
        mean_data = self.df.groupby(self.date_column)[column].mean().reset_index()
        
        plt.figure(figsize=(14, 8))
        plt.plot(mean_data[self.date_column], mean_data[column], 
                linestyle='-', marker='o', linewidth=2, label='Mean')
        
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.xticks(rotation=45)
        plt.show()

    def plot_histograms(self, columns: List[str], title: str, bins: int = 30, verbose: bool = True) -> None:
        if verbose:
            print_subsection_header("Distribution Analysis")
        
        plt.figure(figsize=(14, 8))
        
        for column in columns:
            if column in self.df.columns:
                sns.histplot(self.df[column].dropna(), bins=bins, alpha=0.5, kde=True, label=column)
        
        plt.title(title)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_categorical_histogram(self, column: str, title: str, verbose: bool = True) -> None:
        if column not in self.df.columns:
            if verbose:
                print(f"Column {column} not found in the dataframe")
            return
        
        if verbose:
            print_subsection_header(f"Categorical Distribution: {column}")
        
        category_counts = self.df[column].value_counts().sort_index()
        
        plt.figure(figsize=(14, 8))
        plt.bar(category_counts.index, category_counts.values)
        plt.title(title)
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.grid(True, axis='y')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def calculate_category_summary_statistics(self, columns: List[str], 
                                            verbose: bool = True) -> Optional[pd.DataFrame]:
        if not self.has_categorical:
            if verbose:
                print("No categorical variables detected. Skipping category-level statistics.")
            return None
        
        if verbose:
            print_subsection_header("Category Summary Statistics")
        
        primary_categorical = self.categorical_columns[0]
        result = []
        
        for category in sorted(self.df[primary_categorical].unique()):
            category_data = self.df[self.df[primary_categorical] == category]
            
            if pd.api.types.is_numeric_dtype(self.df[primary_categorical]):
                category_stats = {
                    'Category': int(category), 
                    'Companies': category_data[self.ticker_column].nunique()
                }
            else:
                category_stats = {
                    'Category': category, 
                    'Companies': category_data[self.ticker_column].nunique()
                }
            
            for column in columns:
                if column in self.df.columns:
                    values = category_data[column].dropna()
                    
                    if len(values) > 0:
                        category_stats[f'{column}_mean'] = values.mean()
                        category_stats[f'{column}_median'] = values.median()
                        category_stats[f'{column}_min'] = values.min()
                        category_stats[f'{column}_max'] = values.max()
                        category_stats[f'{column}_std'] = values.std()
                        
                        if len(values) > 2:
                            category_stats[f'{column}_skewness'] = stats.skew(values)
                            category_stats[f'{column}_kurtosis'] = stats.kurtosis(values)
                        else:
                            category_stats[f'{column}_skewness'] = np.nan
                            category_stats[f'{column}_kurtosis'] = np.nan
                    else:
                        for suffix in ['mean', 'median', 'min', 'max', 'std', 'skewness', 'kurtosis']:
                            category_stats[f'{column}_{suffix}'] = np.nan
            
            result.append(category_stats)
        
        return pd.DataFrame(result)

    def calculate_xtsum(self, columns: List[str], verbose: bool = True) -> pd.DataFrame:
        if verbose:
            print_subsection_header("Panel Summary Statistics (xtsum)")
        
        result = []
        
        for column in columns:
            if column not in self.df.columns:
                if verbose:
                    print(f"Column {column} not found in the dataframe")
                continue
            
            overall_mean = self.df[column].mean()
            overall_std = self.df[column].std()
            overall_min = self.df[column].min()
            overall_max = self.df[column].max()
            n_obs = self.df[column].count()
            overall_skewness = self.df[column].skew()
            overall_kurtosis = self.df[column].kurt()
            overall_cv = (overall_std / overall_mean) * 100 if overall_mean != 0 else np.nan
            
            company_means = self.df.groupby(self.ticker_column)[column].mean()
            between_std = company_means.std()
            between_min = company_means.min()
            between_max = company_means.max()
            n_companies = company_means.count()
            between_skewness = company_means.skew()
            between_kurtosis = company_means.kurt()
            between_cv = (between_std / company_means.mean()) * 100 if company_means.mean() != 0 else np.nan
            
            company_means_expanded = self.df.merge(
                pd.DataFrame({
                    self.ticker_column: company_means.index, 
                    f'{column}_mean': company_means.values
                }),
                on=self.ticker_column, how='left'
            )
            
            within_values = self.df[column] - company_means_expanded[f'{column}_mean'] + overall_mean
            within_std = within_values.std()
            within_min = within_values.min()
            within_max = within_values.max()
            within_skewness = within_values.skew()
            within_kurtosis = within_values.kurt()
            within_cv = (within_std / within_values.mean()) * 100 if within_values.mean() != 0 else np.nan
            
            result.append({
                'Variable': column,
                'Mean': overall_mean,
                'Std Dev': overall_std,
                'CV (%)': overall_cv,
                'Min': overall_min,
                'Max': overall_max,
                'Skewness': overall_skewness,
                'Kurtosis': overall_kurtosis,
                'Observations': n_obs,
                'Between CV (%)': between_cv,
                'Between Min': between_min,
                'Between Max': between_max,
                'Between Skewness': between_skewness,
                'Between Kurtosis': between_kurtosis,
                'Within CV (%)': within_cv,
                'Within Min': within_min,
                'Within Max': within_max,
                'Within Skewness': within_skewness,
                'Within Kurtosis': within_kurtosis,
            })
        
        return pd.DataFrame(result)

    def run_full_diagnostics(self, sample_companies: Optional[List[str]] = None, 
                           sample_size: int = 5, target_columns: Optional[List[str]] = None,
                           verbose: bool = True) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        
        if verbose:
            print_section_header("Panel Data Diagnostics")
            print(f"Dataset contains {format_number(self.df.shape[0])} observations of {format_number(self.df[self.ticker_column].nunique())} companies")
            
            if self.has_categorical:
                print(f"Categorical variables detected: {self.categorical_columns}")
            else:
                print("No categorical variables detected - using simplified analysis")
        
        if sample_companies is None:
            sample_companies = self.sample_companies(sample_size)
        
        if verbose:
            print(f"Sampled companies: {sample_companies}")
        
        if target_columns is None:
            numeric_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
            target_columns = [col for col in numeric_columns 
                            if col not in [self.ticker_column] + self.categorical_columns]
        
        if target_columns and verbose:
            print(f"Analyzing columns: {target_columns}")
            
            if len(target_columns) >= 1:
                self.plot_company_time_series(
                    sample_companies, target_columns[0], 
                    f'{target_columns[0]} by Company', target_columns[0], verbose=False
                )
                
                self.plot_mean_time_series(
                    target_columns[0], f'{target_columns[0]} Over Time', 
                    target_columns[0], verbose=False
                )
                
                if self.has_categorical:
                    self.plot_category_time_series(
                        target_columns[0], f'{target_columns[0]} by Category', 
                        target_columns[0], 'mean', verbose=False
                    )
            
            if len(target_columns) >= 3:
                self.plot_histograms(target_columns[:3], 'Distribution of Key Variables', 
                                   bins=50, verbose=False)
        
        xtsum_results = self.calculate_xtsum(target_columns, verbose=verbose)
        
        category_stats = None
        if self.has_categorical:
            category_stats = self.calculate_category_summary_statistics(target_columns, verbose=verbose)
        
        return xtsum_results, category_stats

    def get_dataset_info(self) -> Dict[str, Any]:
        info = {
            'shape': self.df.shape,
            'companies': self.df[self.ticker_column].nunique(),
            'time_periods': self.df[self.date_column].nunique(),
            'date_range': (self.df[self.date_column].min(), self.df[self.date_column].max()),
            'has_categorical': self.has_categorical,
            'categorical_columns': self.categorical_columns,
            'missing_data': self.df.isnull().sum().to_dict()
        }
        
        if self.has_categorical:
            for col in self.categorical_columns:
                if col in self.df.columns:
                    info[f'{col}_categories'] = self.df[col].nunique()
                    info[f'{col}_values'] = sorted(self.df[col].unique().tolist())
        
        return info


def quick_diagnostics(file_path: str = None, df: pd.DataFrame = None, sample_size: int = 5, 
                     target_columns: Optional[List[str]] = None, 
                     ticker_column: str = DEFAULT_TICKER_COLUMN) -> PanelDiagnostics:
    diagnostics = PanelDiagnostics(file_path=file_path, df=df, ticker_column=ticker_column)
    diagnostics.run_full_diagnostics(sample_size=sample_size, target_columns=target_columns)
    return diagnostics
