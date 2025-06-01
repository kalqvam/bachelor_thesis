import pandas as pd
import numpy as np
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Union

from .constants import (
    FILE_ENCODING, CSV_INDEX, OUTPUT_TIMESTAMP_FORMAT,
    DEFAULT_OUTPUT_DIR, CSV_DATETIME_FORMAT
)


def ensure_directory_exists(directory: Union[str, Path]) -> Path:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def generate_timestamp_filename(base_name: str, extension: str = 'csv') -> str:
    timestamp = datetime.now().strftime(OUTPUT_TIMESTAMP_FORMAT)
    return f"{base_name}_{timestamp}.{extension}"


def load_csv_with_validation(
    file_path: Union[str, Path],
    required_columns: Optional[list] = None,
    encoding: str = FILE_ENCODING
) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path, encoding=encoding)
        
        if required_columns:
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
        
        return df
    
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except pd.errors.EmptyDataError:
        raise ValueError(f"File is empty: {file_path}")
    except Exception as e:
        raise Exception(f"Error loading {file_path}: {str(e)}")


def save_csv_with_timestamp(
    df: pd.DataFrame,
    base_name: str,
    output_dir: Union[str, Path] = DEFAULT_OUTPUT_DIR,
    include_timestamp: bool = True,
    encoding: str = FILE_ENCODING,
    index: bool = CSV_INDEX
) -> Path:
    output_path = ensure_directory_exists(output_dir)
    
    if include_timestamp:
        filename = generate_timestamp_filename(base_name)
    else:
        filename = f"{base_name}.csv"
    
    full_path = output_path / filename
    
    df.to_csv(full_path, index=index, encoding=encoding)
    return full_path


def save_csv(
    df: pd.DataFrame,
    file_path: Union[str, Path],
    encoding: str = FILE_ENCODING,
    index: bool = CSV_INDEX,
    create_dirs: bool = True
) -> Path:
    path = Path(file_path)
    
    if create_dirs and path.parent != Path('.'):
        ensure_directory_exists(path.parent)
    
    df.to_csv(path, index=index, encoding=encoding)
    return path


def backup_file(file_path: Union[str, Path], backup_suffix: str = 'backup') -> Path:
    original_path = Path(file_path)
    if not original_path.exists():
        raise FileNotFoundError(f"Cannot backup non-existent file: {file_path}")
    
    timestamp = datetime.now().strftime(OUTPUT_TIMESTAMP_FORMAT)
    backup_name = f"{original_path.stem}_{backup_suffix}_{timestamp}{original_path.suffix}"
    backup_path = original_path.parent / backup_name
    
    import shutil
    shutil.copy2(original_path, backup_path)
    return backup_path


def get_latest_file(directory: Union[str, Path], pattern: str = "*.csv") -> Optional[Path]:
    dir_path = Path(directory)
    if not dir_path.exists():
        return None
    
    files = list(dir_path.glob(pattern))
    if not files:
        return None
    
    return max(files, key=lambda f: f.stat().st_mtime)


def clean_filename(filename: str) -> str:
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename


def get_file_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    stat = path.stat()
    return {
        'path': str(path.absolute()),
        'size_bytes': stat.st_size,
        'size_mb': round(stat.st_size / (1024 * 1024), 2),
        'created': datetime.fromtimestamp(stat.st_ctime),
        'modified': datetime.fromtimestamp(stat.st_mtime),
        'extension': path.suffix,
        'name': path.name,
        'stem': path.stem
    }


def convert_dates_for_csv(df: pd.DataFrame, date_columns: Optional[list] = None) -> pd.DataFrame:
    df_copy = df.copy()
    
    if date_columns is None:
        date_columns = df_copy.select_dtypes(include=['datetime64']).columns
    
    for col in date_columns:
        if col in df_copy.columns and pd.api.types.is_datetime64_any_dtype(df_copy[col]):
            df_copy[col] = df_copy[col].dt.strftime(CSV_DATETIME_FORMAT)
    
    return df_copy


def read_multiple_csvs(
    file_paths: list,
    combine: bool = False,
    sort_by: Optional[str] = None
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    dataframes = {}
    
    for file_path in file_paths:
        path = Path(file_path)
        df = load_csv_with_validation(file_path)
        dataframes[path.stem] = df
    
    if combine:
        combined_df = pd.concat(dataframes.values(), ignore_index=True)
        if sort_by and sort_by in combined_df.columns:
            combined_df = combined_df.sort_values(sort_by)
        return combined_df
    
    return dataframes
