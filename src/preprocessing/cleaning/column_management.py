import pandas as pd
from typing import List, Dict, Optional, Union, Tuple

from ...utils import (
    DEFAULT_TICKER_COLUMN, print_subsection_header, 
    format_number, print_processing_stats
)


def remove_columns(df: pd.DataFrame,
                  columns_to_remove: Union[str, List[str]],
                  ignore_missing: bool = True,
                  verbose: bool = True) -> Tuple[pd.DataFrame, Dict]:
    
    if verbose:
        print_subsection_header("Removing Columns")
    
    if isinstance(columns_to_remove, str):
        columns_to_remove = [columns_to_remove]
    
    original_columns = df.columns.tolist()
    existing_cols = [col for col in columns_to_remove if col in df.columns]
    missing_cols = [col for col in columns_to_remove if col not in df.columns]
    
    if not existing_cols:
        if verbose:
            print("No columns to remove (all specified columns are missing)")
        
        stats = {
            'original_columns': len(original_columns),
            'columns_requested': len(columns_to_remove),
            'columns_removed': 0,
            'missing_columns': missing_cols,
            'final_columns': len(original_columns)
        }
        return df.copy(), stats
    
    df_result = df.drop(columns=existing_cols)
    
    stats = {
        'original_columns': len(original_columns),
        'columns_requested': len(columns_to_remove),
        'columns_removed': len(existing_cols),
        'columns_removed_list': existing_cols,
        'missing_columns': missing_cols,
        'final_columns': len(df_result.columns)
    }
    
    if verbose:
        print(f"Column removal results:")
        print(f"  Requested: {len(columns_to_remove)} columns")
        print(f"  Removed: {len(existing_cols)} columns")
        if existing_cols:
            print(f"  Removed columns: {existing_cols}")
        
        if missing_cols:
            if ignore_missing:
                print(f"  Missing columns (ignored): {missing_cols}")
            else:
                print(f"  Warning: Missing columns: {missing_cols}")
        
        print(f"  Final column count: {len(df_result.columns)}")
    
    return df_result, stats


def reorder_columns(df: pd.DataFrame,
                   column_order: Optional[List[str]] = None,
                   priority_columns: Optional[List[str]] = None,
                   ticker_column: str = DEFAULT_TICKER_COLUMN,
                   date_column: str = 'quarter',
                   verbose: bool = True) -> Tuple[pd.DataFrame, Dict]:
    
    if verbose:
        print_subsection_header("Reordering Columns")
    
    original_order = df.columns.tolist()
    
    if column_order:
        existing_order_cols = [col for col in column_order if col in df.columns]
        remaining_cols = [col for col in df.columns if col not in column_order]
        new_order = existing_order_cols + remaining_cols
        
        method_used = "explicit_order"
        
    elif priority_columns:
        existing_priority = [col for col in priority_columns if col in df.columns]
        remaining_cols = [col for col in df.columns if col not in priority_columns]
        new_order = existing_priority + remaining_cols
        
        method_used = "priority_columns"
        
    else:
        new_order = []
        
        if ticker_column in df.columns:
            new_order.append(ticker_column)
        
        if date_column in df.columns and date_column not in new_order:
            new_order.append(date_column)
        
        remaining_cols = [col for col in df.columns if col not in new_order]
        new_order.extend(remaining_cols)
        
        method_used = "default_ticker_date_first"
    
    df_reordered = df[new_order]
    
    stats = {
        'original_order': original_order,
        'new_order': new_order,
        'method_used': method_used,
        'columns_moved': original_order != new_order,
        'total_columns': len(new_order)
    }
    
    if verbose:
        print(f"Column reordering results:")
        print(f"  Method: {method_used}")
        print(f"  Columns moved: {'Yes' if stats['columns_moved'] else 'No'}")
        if stats['columns_moved']:
            if len(new_order) <= 10:
                print(f"  New order: {new_order}")
            else:
                print(f"  First 5 columns: {new_order[:5]}")
    
    return df_reordered, stats


def rename_columns(df: pd.DataFrame,
                  column_mapping: Dict[str, str],
                  ignore_missing: bool = True,
                  verbose: bool = True) -> Tuple[pd.DataFrame, Dict]:
    
    if verbose:
        print_subsection_header("Renaming Columns")
    
    existing_mappings = {old: new for old, new in column_mapping.items() if old in df.columns}
    missing_columns = [old for old in column_mapping.keys() if old not in df.columns]
    
    if not existing_mappings:
        if verbose:
            print("No columns to rename (all specified columns are missing)")
        
        stats = {
            'requested_renames': len(column_mapping),
            'successful_renames': 0,
            'missing_columns': missing_columns
        }
        return df.copy(), stats
    
    df_renamed = df.rename(columns=existing_mappings)
    
    stats = {
        'requested_renames': len(column_mapping),
        'successful_renames': len(existing_mappings),
        'renamed_columns': existing_mappings,
        'missing_columns': missing_columns
    }
    
    if verbose:
        print(f"Column renaming results:")
        print(f"  Requested: {len(column_mapping)} renames")
        print(f"  Successful: {len(existing_mappings)} renames")
        if existing_mappings:
            print(f"  Renamed: {existing_mappings}")
        
        if missing_columns:
            if ignore_missing:
                print(f"  Missing columns (ignored): {missing_columns}")
            else:
                print(f"  Warning: Missing columns: {missing_columns}")
    
    return df_renamed, stats


def add_columns(df: pd.DataFrame,
               new_columns: Dict[str, any],
               position: Optional[str] = None,
               verbose: bool = True) -> Tuple[pd.DataFrame, Dict]:
    
    if verbose:
        print_subsection_header("Adding New Columns")
    
    df_result = df.copy()
    columns_added = []
    columns_updated = []
    
    for col_name, col_value in new_columns.items():
        if col_name in df_result.columns:
            columns_updated.append(col_name)
        else:
            columns_added.append(col_name)
        
        df_result[col_name] = col_value
    
    if position and columns_added:
        current_cols = df_result.columns.tolist()
        
        for col_name in columns_added:
            current_cols.remove(col_name)
            
            if position == 'first':
                current_cols.insert(0, col_name)
            elif position == 'second':
                current_cols.insert(1, col_name)
            elif position == 'last':
                current_cols.append(col_name)
            elif position.startswith('after_'):
                target_col = position.replace('after_', '')
                if target_col in current_cols:
                    target_idx = current_cols.index(target_col)
                    current_cols.insert(target_idx + 1, col_name)
                else:
                    current_cols.append(col_name)
        
        df_result = df_result[current_cols]
    
    stats = {
        'columns_added': len(columns_added),
        'columns_updated': len(columns_updated),
        'new_column_names': columns_added,
        'updated_column_names': columns_updated,
        'position': position,
        'final_column_count': len(df_result.columns)
    }
    
    if verbose:
        print(f"Column addition results:")
        print(f"  New columns added: {len(columns_added)}")
        if columns_added:
            print(f"  Added: {columns_added}")
        
        print(f"  Existing columns updated: {len(columns_updated)}")
        if columns_updated:
            print(f"  Updated: {columns_updated}")
        
        if position:
            print(f"  Position: {position}")
    
    return df_result, stats


def standardize_column_names(df: pd.DataFrame,
                           naming_convention: str = 'snake_case',
                           verbose: bool = True) -> Tuple[pd.DataFrame, Dict]:
    
    if verbose:
        print_subsection_header("Standardizing Column Names")
    
    import re
    
    original_names = df.columns.tolist()
    new_names = []
    
    for col in original_names:
        if naming_convention == 'snake_case':
            new_name = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', col).lower()
            new_name = re.sub(r'[^a-z0-9_]', '_', new_name)
            new_name = re.sub(r'_+', '_', new_name).strip('_')
        
        elif naming_convention == 'camelCase':
            new_name = re.sub(r'[^a-zA-Z0-9]', ' ', col).title().replace(' ', '')
            new_name = new_name[0].lower() + new_name[1:] if new_name else new_name
        
        elif naming_convention == 'PascalCase':
            new_name = re.sub(r'[^a-zA-Z0-9]', ' ', col).title().replace(' ', '')
        
        elif naming_convention == 'lower':
            new_name = col.lower()
        
        elif naming_convention == 'upper':
            new_name = col.upper()
        
        else:
            new_name = col
        
        new_names.append(new_name)
    
    mapping = dict(zip(original_names, new_names))
    changed_names = {old: new for old, new in mapping.items() if old != new}
    
    df_standardized = df.rename(columns=changed_names)
    
    stats = {
        'naming_convention': naming_convention,
        'total_columns': len(original_names),
        'columns_changed': len(changed_names),
        'name_mapping': changed_names
    }
    
    if verbose:
        print(f"Column name standardization results:")
        print(f"  Convention: {naming_convention}")
        print(f"  Columns changed: {len(changed_names)}")
        if changed_names and len(changed_names) <= 10:
            print(f"  Changes: {changed_names}")
        elif len(changed_names) > 10:
            sample_changes = dict(list(changed_names.items())[:5])
            print(f"  Sample changes: {sample_changes}...")
    
    return df_standardized, stats


def get_column_info(df: pd.DataFrame, verbose: bool = True) -> Dict:
    
    column_info = {
        'total_columns': len(df.columns),
        'column_names': df.columns.tolist(),
        'column_types': df.dtypes.to_dict(),
        'memory_usage': df.memory_usage(deep=True).to_dict()
    }
    
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_columns = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    column_info.update({
        'numeric_columns': numeric_columns,
        'categorical_columns': categorical_columns,
        'datetime_columns': datetime_columns,
        'column_counts_by_type': {
            'numeric': len(numeric_columns),
            'categorical': len(categorical_columns),
            'datetime': len(datetime_columns)
        }
    })
    
    if verbose:
        print_subsection_header("Column Information Summary")
        print(f"Total columns: {column_info['total_columns']}")
        print(f"Numeric columns: {len(numeric_columns)}")
        print(f"Categorical columns: {len(categorical_columns)}")
        print(f"Datetime columns: {len(datetime_columns)}")
        
        total_memory = sum(column_info['memory_usage'].values())
        print(f"Total memory usage: {total_memory / 1024 / 1024:.2f} MB")
    
    return column_info


def validate_required_columns_present(df: pd.DataFrame,
                                     required_columns: List[str],
                                     verbose: bool = True) -> Dict:
    
    present_columns = [col for col in required_columns if col in df.columns]
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    validation_result = {
        'all_present': len(missing_columns) == 0,
        'required_columns': required_columns,
        'present_columns': present_columns,
        'missing_columns': missing_columns,
        'completion_rate': len(present_columns) / len(required_columns) if required_columns else 1.0
    }
    
    if verbose:
        print_subsection_header("Required Columns Validation")
        if validation_result['all_present']:
            print(f"✓ All {len(required_columns)} required columns are present")
        else:
            print(f"✗ Missing {len(missing_columns)} required columns:")
            for col in missing_columns:
                print(f"  - {col}")
            print(f"Completion rate: {validation_result['completion_rate']:.1%}")
    
    return validation_result
