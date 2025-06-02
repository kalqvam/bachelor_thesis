import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional, Tuple

from ..utils import (
    DEFAULT_TICKER_COLUMN, DEFAULT_TIME_COLUMN, print_subsection_header,
    format_number, print_processing_stats
)


class DataProcessor:
    def __init__(self, file_path: str):
        self.df = pd.read_csv(file_path)
        self.original_columns = self.df.columns.tolist()
        self.transformations = []

        if DEFAULT_TICKER_COLUMN not in self.df.columns:
            raise ValueError(f"The dataset must contain a '{DEFAULT_TICKER_COLUMN}' column")

        self._detect_and_prepare_time_column()

        print(f"Dataset loaded with {format_number(self.df.shape[0])} rows and {self.df.shape[1]} columns")
        print(f"Available columns: {', '.join(self.original_columns)}")
        print(f"Number of unique tickers: {format_number(self.df[DEFAULT_TICKER_COLUMN].nunique())}")

    def _detect_and_prepare_time_column(self):
        time_column_candidates = ['date', 'quarter_year', 'quarter']
        self.time_column = None

        for candidate in time_column_candidates:
            if candidate in self.df.columns:
                self.time_column = candidate
                break

        if self.time_column == 'date' and pd.api.types.is_datetime64_any_dtype(self.df['date']):
            pass
        elif self.time_column in self.df.columns:
            try:
                self.df[self.time_column] = pd.to_datetime(self.df[self.time_column])
            except:
                pass

    def categorical_to_numerical(self, column: str) -> Dict[str, int]:
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in the dataset")

        if self.df[column].dtype == 'object' or self.df[column].dtype.name == 'category':
            categories = self.df[column].unique()
            mapping = {cat: i for i, cat in enumerate(categories)}
            new_column = f"{column}_num"
            self.df[new_column] = self.df[column].map(mapping)

            self.transformations.append({
                'type': 'categorical_to_numerical',
                'source_column': column,
                'new_column': new_column,
                'mapping': mapping
            })

            print(f"Created new column '{new_column}' with the following mapping:")
            for cat, num in mapping.items():
                print(f"  {cat} -> {num}")

            return mapping
        else:
            print(f"Column '{column}' is already numeric. No conversion needed.")
            return {}

    def categorical_to_numerical_ordered(self, column: str, order: List[str]) -> Dict[str, int]:
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in the dataset")

        categories = self.df[column].unique()
        missing_categories = [cat for cat in categories if cat not in order]

        if missing_categories:
            raise ValueError(f"The following categories are not in the order list: {missing_categories}")

        mapping = {cat: i for i, cat in enumerate(order)}
        new_column = f"{column}_ordered"
        self.df[new_column] = self.df[column].map(mapping)

        self.transformations.append({
            'type': 'categorical_to_numerical_ordered',
            'source_column': column,
            'new_column': new_column,
            'mapping': mapping,
            'order': order
        })

        print(f"Created new column '{new_column}' with the following ordered mapping:")
        for cat, num in mapping.items():
            if cat in categories:
                print(f"  {cat} -> {num}")

        return mapping

    def calculate_ratio(self, numerator: str, denominator: str, handle_zeros: str = 'replace') -> str:
        if numerator not in self.df.columns:
            raise ValueError(f"Column '{numerator}' not found in the dataset")
        if denominator not in self.df.columns:
            raise ValueError(f"Column '{denominator}' not found in the dataset")

        new_column = f"{numerator}_to_{denominator}_ratio"

        if handle_zeros == 'drop':
            self.df = self.df[self.df[denominator] != 0]
            print(f"Dropped rows where {denominator} = 0")

        self.df[new_column] = self.df[numerator] / self.df[denominator]

        if handle_zeros == 'replace':
            self.df = self.df.copy()
            self.df[new_column] = self.df[new_column].replace([np.inf, -np.inf], np.nan)
            print(f"Replaced infinite values with NaN in '{new_column}'")

        self.transformations.append({
            'type': 'ratio',
            'numerator': numerator,
            'denominator': denominator,
            'new_column': new_column,
            'handle_zeros': handle_zeros
        })

        print(f"Created new column '{new_column}'")
        return new_column

    def calculate_percentage_change(self, column: str, lag: int = 1) -> str:
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in the dataset")

        if not self.time_column:
            raise ValueError("No valid time column found for time-based calculations")

        self.df = self.df.sort_values([DEFAULT_TICKER_COLUMN, self.time_column])

        new_column = f"{column}_pct_change_lag{lag}"

        self.df[new_column] = self.df.groupby(DEFAULT_TICKER_COLUMN)[column].pct_change(periods=lag) * 100

        self.transformations.append({
            'type': 'percentage_change',
            'source_column': column,
            'new_column': new_column,
            'lag': lag
        })

        print(f"Created new column '{new_column}' with {lag}-period lag")
        return new_column

    def calculate_lag_differences(self, columns: List[str], lags: List[int]) -> List[str]:
        for column in columns:
            if column not in self.df.columns:
                raise ValueError(f"Column '{column}' not found in the dataset")

        if not self.time_column:
            raise ValueError("No valid time column found for time-based calculations")

        self.df = self.df.sort_values([DEFAULT_TICKER_COLUMN, self.time_column])

        new_columns = []

        for column in columns:
            for lag in lags:
                new_column = f"{column}_diff_lag{lag}"
                lagged_column = f"{column}_lag{lag}"
                self.df[lagged_column] = self.df.groupby(DEFAULT_TICKER_COLUMN)[column].shift(lag)
                self.df[new_column] = self.df[column] - self.df[lagged_column]

                self.transformations.append({
                    'type': 'lag_difference',
                    'source_column': column,
                    'new_column': new_column,
                    'lag': lag,
                    'intermediate_column': lagged_column
                })

                new_columns.append(new_column)

                print(f"Created new column '{new_column}' with difference between current and {lag}-period lag")
                self.df = self.df.drop(lagged_column, axis=1)

        return new_columns

    def calculate_normalized_values(self, column: str) -> List[str]:
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in the dataset")

        ticker_stats = self.df.groupby(DEFAULT_TICKER_COLUMN)[column].agg(['mean', 'std']).reset_index()
        ticker_stats.columns = [DEFAULT_TICKER_COLUMN, f'{column}_mean', f'{column}_std']

        self.df = self.df.merge(ticker_stats, on=DEFAULT_TICKER_COLUMN, how='left')

        new_columns = []

        norm1_col = f"{column}_norm_by_mean"
        self.df[norm1_col] = self.df[column] / self.df[f'{column}_mean']
        new_columns.append(norm1_col)

        norm2_col = f"{column}_norm_by_std"
        self.df[norm2_col] = self.df[column] / self.df[f'{column}_std']
        new_columns.append(norm2_col)

        norm3_col = f"{column}_z_score"
        self.df[norm3_col] = (self.df[column] - self.df[f'{column}_mean']) / self.df[f'{column}_std']
        new_columns.append(norm3_col)

        norm4_col = f"{column}_rel_norm"
        self.df[norm4_col] = (self.df[column] / self.df[f'{column}_mean']) / self.df[f'{column}_std']
        new_columns.append(norm4_col)

        self.df = self.df.copy()
        for col in new_columns:
            self.df[col] = self.df[col].replace([np.inf, -np.inf], np.nan)

        self.transformations.append({
            'type': 'normalized_values',
            'source_column': column,
            'new_columns': new_columns
        })

        print(f"Created {len(new_columns)} normalized columns for '{column}':")
        for col in new_columns:
            print(f"  - {col}")

        self.df = self.df.drop([f'{column}_mean', f'{column}_std'], axis=1)

        return new_columns

    def calculate_natural_log(self, column: str, strict: bool = True) -> str:
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in the dataset")

        new_column = f"ln_{column}"

        zero_or_neg_count = (self.df[column] <= 0).sum()

        if zero_or_neg_count > 0:
            if strict:
                raise ValueError(f"Column '{column}' contains {zero_or_neg_count} zero or negative values, "
                                f"which cannot be log-transformed. Use strict=False to replace with NaN.")
            else:
                print(f"Warning: Column '{column}' contains {zero_or_neg_count} zero or negative values. "
                     f"These will be replaced with NaN in the log-transformed column.")

        self.df[new_column] = np.log(self.df[column])

        self.transformations.append({
            'type': 'natural_log',
            'source_column': column,
            'new_column': new_column,
            'strict_mode': strict
        })

        print(f"Created new column '{new_column}' with natural logarithm of '{column}'")
        if not strict and zero_or_neg_count > 0:
            print(f"  Note: {zero_or_neg_count} values were set to NaN due to zero or negative inputs")

        return new_column

    def calculate_cross_sectional_ratio(self, column: str) -> str:
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in the dataset")

        if not self.time_column:
            raise ValueError("No valid time column found for this calculation")

        new_column = f"{column}_rel_to_avg"

        period_avg = self.df.groupby(self.time_column)[column].mean().reset_index()
        period_avg.columns = [self.time_column, f'{column}_period_avg']

        self.df = self.df.merge(period_avg, on=self.time_column, how='left')

        self.df[new_column] = self.df[column] / self.df[f'{column}_period_avg']

        self.df = self.df.copy()
        self.df[new_column] = self.df[new_column].replace([np.inf, -np.inf], np.nan)

        self.transformations.append({
            'type': 'cross_sectional_ratio',
            'source_column': column,
            'new_column': new_column,
            'time_column': self.time_column
        })

        print(f"Created new column '{new_column}' with ratio of '{column}' to period average")

        self.df = self.df.drop(f'{column}_period_avg', axis=1)

        return new_column

    def calculate_cross_sectional_difference(self, column: str) -> str:
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in the dataset")

        if not self.time_column:
            raise ValueError("No valid time column found for this calculation")

        new_column = f"{column}_diff_from_avg"

        period_avg = self.df.groupby(self.time_column)[column].mean().reset_index()
        period_avg.columns = [self.time_column, f'{column}_period_avg']

        self.df = self.df.merge(period_avg, on=self.time_column, how='left')

        self.df[new_column] = self.df[column] - self.df[f'{column}_period_avg']

        self.transformations.append({
            'type': 'cross_sectional_difference',
            'source_column': column,
            'new_column': new_column,
            'time_column': self.time_column
        })

        print(f"Created new column '{new_column}' with difference of '{column}' from period average")

        self.df = self.df.drop(f'{column}_period_avg', axis=1)

        return new_column

    def calculate_min_max_normalization(self, column: str) -> str:
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in the dataset")

        if not self.time_column:
            raise ValueError("No valid time column found for this calculation")

        new_column = f"{column}_minmax_norm"

        period_min = self.df.groupby(self.time_column)[column].min().reset_index()
        period_min.columns = [self.time_column, f'{column}_period_min']

        period_max = self.df.groupby(self.time_column)[column].max().reset_index()
        period_max.columns = [self.time_column, f'{column}_period_max']

        self.df = self.df.merge(period_min, on=self.time_column, how='left')
        self.df = self.df.merge(period_max, on=self.time_column, how='left')

        self.df[new_column] = (self.df[column] - self.df[f'{column}_period_min']) / \
                             (self.df[f'{column}_period_max'] - self.df[f'{column}_period_min'])

        self.df = self.df.copy()
        self.df[new_column] = self.df[new_column].replace([np.inf, -np.inf], np.nan)

        self.transformations.append({
            'type': 'min_max_normalization',
            'source_column': column,
            'new_column': new_column,
            'time_column': self.time_column
        })

        print(f"Created new column '{new_column}' with min-max normalization of '{column}' per period")

        self.df = self.df.drop([f'{column}_period_min', f'{column}_period_max'], axis=1)

        return new_column

    def calculate_equal_weight_average(self, columns: List[str], name: str) -> str:
        for column in columns:
            if column not in self.df.columns:
                raise ValueError(f"Column '{column}' not found in the dataset")

        new_column = f"{name}_avg"

        self.df[new_column] = self.df[columns].mean(axis=1)

        self.transformations.append({
            'type': 'equal_weight_average',
            'source_columns': columns,
            'new_column': new_column
        })

        print(f"Created new column '{new_column}' with equal-weight average of: {', '.join(columns)}")

        return new_column

    def clean_dataset(self, keep_original: bool = False, drop_na: bool = True,
                     drop_na_subset: Optional[List[str]] = None) -> pd.DataFrame:
        if not keep_original:
            columns_to_remove = []

            for transform in self.transformations:
                if transform['type'] in ['categorical_to_numerical', 'categorical_to_numerical_ordered']:
                    columns_to_remove.append(transform['source_column'])
                elif transform['type'] == 'ratio':
                    columns_to_remove.extend([transform['numerator'], transform['denominator']])
                elif transform['type'] in ['percentage_change', 'normalized_values', 'natural_log', 'lag_difference',
                                          'cross_sectional_ratio', 'cross_sectional_difference', 'min_max_normalization']:
                    columns_to_remove.append(transform['source_column'])
                elif transform['type'] == 'equal_weight_average':
                    columns_to_remove.extend(transform['source_columns'])

            columns_to_remove = list(set(columns_to_remove))

            existing_columns = [col for col in columns_to_remove if col in self.df.columns]
            if existing_columns:
                self.df = self.df.drop(existing_columns, axis=1)
                print(f"Removed {len(existing_columns)} original columns that were used in transformations")

        if drop_na:
            row_count_before = len(self.df)

            if drop_na_subset:
                self.df = self.df.dropna(subset=drop_na_subset)
                print(f"Removed {format_number(row_count_before - len(self.df))} rows with NaN values in columns: {', '.join(drop_na_subset)}")
            else:
                self.df = self.df.dropna()
                print(f"Removed {format_number(row_count_before - len(self.df))} rows with NaN values")

        return self.df

    def save_dataset(self, file_path: str) -> None:
        self.df.to_csv(file_path, index=False)
        print(f"Dataset saved to {file_path}")

    def get_transformed_data(self) -> pd.DataFrame:
        return self.df.copy()

    def get_transformation_summary(self) -> Dict:
        summary = {
            'total_transformations': len(self.transformations),
            'transformation_types': {},
            'columns_created': [],
            'final_shape': self.df.shape,
            'original_columns': len(self.original_columns),
            'final_columns': len(self.df.columns)
        }

        for transform in self.transformations:
            transform_type = transform['type']
            summary['transformation_types'][transform_type] = summary['transformation_types'].get(transform_type, 0) + 1
            
            if 'new_column' in transform:
                summary['columns_created'].append(transform['new_column'])
            elif 'new_columns' in transform:
                summary['columns_created'].extend(transform['new_columns'])

        return summary

    def print_transformation_summary(self) -> None:
        summary = self.get_transformation_summary()
        
        print_subsection_header("Transformation Summary")
        print(f"Total transformations applied: {summary['total_transformations']}")
        print(f"Original columns: {summary['original_columns']}")
        print(f"Final columns: {summary['final_columns']}")
        print(f"Columns created: {len(summary['columns_created'])}")
        
        print("\nTransformations by type:")
        for transform_type, count in summary['transformation_types'].items():
            print(f"  {transform_type.replace('_', ' ').title()}: {count}")
        
        print(f"\nFinal dataset shape: {summary['final_shape']}")
