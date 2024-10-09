import numpy as np
import pandas as pd
from config import COLUMNS_WITH_FEW_MISSING, VEGETATION_COLS
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# Function to handle NaN values in numerical and categorical columns
def handle_missing_values(df):
    for column in df.columns:
        if df[column].dtype == 'object' or pd.api.types.is_categorical_dtype(X[column]):
            # For categorical columns, fill NaNs with the mode
            mode_value = df[column].mode()[0]
            df[column] = df[column].fillna(mode_value)
        else:
            # For numerical columns, fill NaNs with the mean
            mean_value = df[column].mean()
            df[column] = df[column].fillna(mean_value)
    return df


# Function to handle infinite values
def handle_infinite_values(df):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace inf/-inf with NaN
    return handle_missing_values(df)  # Re-impute NaN values after replacing infinities


# Function to cap extreme values
def cap_extreme_values(df, threshold=1e6):
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    df[numerical_columns] = df[numerical_columns].clip(upper=threshold)  # Cap values at threshold
    return df


def feature_scaling(df, method='standardize', columns=None, target_column='ignition'):
    """
    Function to scale numerical features in a dataframe, excluding the target column.
    
    Parameters:
    - df (pd.DataFrame): Input dataframe containing the data.
    - method (str): Scaling method to use. Options are 'standardize' or 'normalize'.
    - columns (list): List of columns to scale. If None, scales all numerical columns.
    - target_column (str): Name of the target column to exclude from scaling.
    
    Returns:
    - df_scaled (pd.DataFrame): Dataframe with scaled features.
    """
    # If no columns are specified, select all numerical columns except the target column
    if columns is None:
        columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # Remove the target column from the list of columns to scale
    if target_column in columns:
        columns.remove(target_column)
    
    # Make a copy of the dataframe to avoid modifying the original one
    df_scaled = df.copy()
    
    # Choose the scaler based on the method specified
    if method == 'normalize':
        scaler = MinMaxScaler()
    elif method == 'standardize':
        scaler = StandardScaler()
    else:
        raise ValueError("Invalid method. Choose 'normalize' or 'standardize'.")
    
    # Apply scaling to the specified columns
    df_scaled[columns] = scaler.fit_transform(df_scaled[columns])
    
    return df_scaled


# Main function to preprocess the data
def preprocess_data(df, target_column='ignition', threshold=1e6):
   
    # Drops rows where the specified columns have missing values.
    df.dropna(subset=COLUMNS_WITH_FEW_MISSING, inplace=True)

    # Clean the vegetation columns by ensuring valid ratios
    df = df[(df[VEGETATION_COLS].sum(axis=1) - 1).abs() <= 1e-6]

    # Clean 'vegetation_class' and drop duplicates
    df['vegetation_class'] = df['vegetation_class'].astype('category')

    # Drop unnecessary columns such as 'yearly_avg_temp'
    df.drop(columns=['yearly_avg_temp'], inplace=True)

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Handle NaN values
    X = handle_missing_values(X)

    # Handle infinite values
    X = handle_infinite_values(X)

    # Cap extreme values
    X = cap_extreme_values(X, threshold)

    # Scale features
    X_scaled = scale_features(X)

    # Combine scaled features with the target variable
    df_preprocessed = pd.concat([pd.DataFrame(X_scaled, columns=X.columns), y.reset_index(drop=True)], axis=1)
    
    return df_preprocessed
