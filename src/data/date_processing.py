import pandas as pd

def process_date_column(df, date_column='Date'):
    """
    Processes the 'Date' column by converting it to a datetime object, extracting useful components,
    and handling errors. If the column doesn't exist, it will be ignored.

    Parameters:
    - df: Input DataFrame
    - date_column: The name of the date column (default is 'Date')

    Returns:
    - df: DataFrame with processed date features
    """
    if date_column in df.columns:
        # Convert 'Date' column to datetime type
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')

        # Extract year, month, and day into separate columns
        df['Year'] = df[date_column].dt.year
        df['Month'] = df[date_column].dt.month
        df['Day'] = df[date_column].dt.day

        # Drop the original 'Date' column if required (optional)
        df.drop(columns=[date_column], inplace=True)

        print(f"Processed '{date_column}' and extracted 'Year', 'Month', 'Day'.")
    else:
        print(f"'{date_column}' column not found in DataFrame.")

    return df
