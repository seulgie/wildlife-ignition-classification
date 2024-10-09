import pandas as pd

def load_data(file_path):
    """
    Load dataset from a CSV file.

    Parameters:
    - file_path: Path to the CSV file.

    Returns:
    - df: Loaded DataFrame.
    """
    return pd.read_csv(file_path, index_col=0)


def save_data(df, file_path):
    """
    Save the Processed DataFrame to a CSV file.

    Parameters:
    - df: DataFrame to save.
    - file_path: Path where the CSV will be saved.
    """
    df.to_csv(file_path, index=False)
    print("Processed Data saved to:", file_path)
