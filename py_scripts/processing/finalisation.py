import pandas as pd

def clean_data(csv_file, columns_to_remove=None, columns_to_rename=None):
    df = pd.read_csv(csv_file)

    if columns_to_remove and isinstance(columns_to_remove, str):
        columns_to_remove = [columns_to_remove]

    if columns_to_remove:
        existing_cols = [col for col in columns_to_remove if col in df.columns]
        missing_cols = [col for col in columns_to_remove if col not in df.columns]

        if existing_cols:
            df = df.drop(columns=existing_cols)
            print(f"Removed columns: {existing_cols}")

        if missing_cols:
            print(f"Warning: Columns {missing_cols} not found in the DataFrame")

    if columns_to_rename and isinstance(columns_to_rename, dict):
        existing_cols = {old: new for old, new in columns_to_rename.items() if old in df.columns}
        missing_cols = [old for old in columns_to_rename.keys() if old not in df.columns]

        if existing_cols:
            df = df.rename(columns=existing_cols)
            print(f"Renamed columns: {existing_cols}")

        if missing_cols:
            print(f"Warning: Columns to rename {missing_cols} not found in the DataFrame")

    if 'quarter' in df.columns:
        cols = [col for col in df.columns if col != 'quarter']
        cols.insert(1, 'quarter')
        df = df[cols]
        print("Moved 'quarter' to the second position")
    else:
        print("Warning: 'quarter' column not found in the DataFrame")

    return df

def add_shock_dummy(df, shock_name, year=None, quarter=None):
    df = df.copy()

    if 'date' not in df.columns:
        print("Error: 'date' column not found in the DataFrame")
        return df

    df[shock_name] = 0

    if pd.api.types.is_string_dtype(df['date']):
        if df['date'].iloc[0].find('Q') > 0:
            df['temp_year'] = df['date'].str.split('-').str[0].astype(int)
            df['temp_quarter'] = df['date'].str.split('-Q').str[1].astype(int)
        else:
            df['date'] = pd.to_datetime(df['date'])
            df['temp_year'] = df['date'].dt.year
            df['temp_quarter'] = df['date'].dt.quarter
    elif pd.api.types.is_datetime64_dtype(df['date']):
        df['temp_year'] = df['date'].dt.year
        df['temp_quarter'] = df['date'].dt.quarter

    if year is not None and not isinstance(year, list):
        year = [year]
    if quarter is not None and not isinstance(quarter, list):
        quarter = [quarter]

    if year is not None and quarter is not None:
        for y in year:
            for q in quarter:
                mask = (df['temp_year'] == y) & (df['temp_quarter'] == q)
                df.loc[mask, shock_name] = 1
        print(f"Added shock dummy '{shock_name}' for year(s) {year}, quarter(s) {quarter}")
    elif year is not None:
        for y in year:
            mask = df['temp_year'] == y
            df.loc[mask, shock_name] = 1
        print(f"Added shock dummy '{shock_name}' for year(s) {year}")
    else:
        print("Error: At least 'year' must be specified")

    df = df.drop(columns=['temp_year', 'temp_quarter'], errors='ignore')

    return df

def add_time_dummy(df, company_col='company', date_col='date'):
    df = df.copy()

    if company_col not in df.columns:
        print(f"Error: '{company_col}' column not found in the DataFrame")
        return df

    if date_col not in df.columns:
        print(f"Error: '{date_col}' column not found in the DataFrame")
        return df
    if not pd.api.types.is_datetime64_dtype(df[date_col]):
        if pd.api.types.is_string_dtype(df[date_col]):
            if df[date_col].iloc[0].find('Q') > 0:
                df['temp_year'] = df[date_col].str.split('-').str[0].astype(int)
                df['temp_quarter'] = df[date_col].str.split('-Q').str[1].astype(int)
                df['temp_date'] = df['temp_year'] * 10 + df['temp_quarter']
            else:
                df['temp_date'] = pd.to_datetime(df[date_col])
        else:
            df['temp_date'] = df[date_col]
    else:
        df['temp_date'] = df[date_col]

    df = df.sort_values([company_col, 'temp_date'])

    df['time_dummy'] = 0

    for company, group in df.groupby(company_col):
        df.loc[group.index, 'time_dummy'] = range(1, len(group) + 1)

    if 'temp_year' in df.columns:
        df = df.drop(columns=['temp_year', 'temp_quarter', 'temp_date'])
    elif 'temp_date' in df.columns:
        df = df.drop(columns=['temp_date'])

    print(f"Added 'time_dummy' column with sequential numbers for each {company_col}")

    return df
