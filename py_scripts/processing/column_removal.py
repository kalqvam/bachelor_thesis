import pandas as pd

def clean_data(csv_file, columns_to_remove=None):
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

    if 'date_quarter' in df.columns:
        cols = [col for col in df.columns if col != 'date_quarter']

        cols.insert(1, 'date_quarter')

        df = df[cols]
        print("Moved 'date_quarter' to the second position")
    else:
        print("Warning: 'date_quarter' column not found in the DataFrame")

    return df
