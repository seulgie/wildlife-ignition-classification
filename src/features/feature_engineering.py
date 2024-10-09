import numpy as np
import pandas as pd
from config import PCA_COLS, KEY_COLUMNS
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from src.data.data_cleaning import feature_scaling


def feature_selection_rfe(df, target_column, num_features=10):
    """
    Perform Recursive Feature Elimination (RFE) to select important features.
    
    Parameters:
    - df (pd.DataFrame): Input dataframe.
    - target_column (str): The name of the target column.
    - num_features (int): Number of top features to select.
    
    Returns:
    - selected_features (list): List of selected feature names.
    """
    # Identify numerical columns only
    numerical_columns = df.select_dtypes(exclude=['object', 'category']).columns.tolist()

    # Remove the target column from the numerical columns list
    if target_column in numerical_columns:
        numerical_columns.remove(target_column)
    
    X = df[numerical_columns]  # Use only numerical features
    y = df[target_column]  # Target column
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Use a random forest classifier for RFE
    model = RandomForestClassifier()
    
    # Perform RFE
    rfe = RFE(model, n_features_to_select=num_features)
    rfe.fit(X_train, y_train)
    
    # Get the list of selected features
    selected_features = X_train.columns[rfe.support_].tolist()
    
    return selected_features


def clean_and_remove_outliers(df, target_column='ignition', threshold=0.01):
    # Step 1: Replace inf values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Step 2: Drop or impute rows with NaN values (choose one)
    df.dropna(inplace=True)  # Dropping rows with NaN values
    # Alternatively, you can use imputation like df.fillna(df.mean(), inplace=True)
    
    # Step 3: Select numerical columns, excluding the target column
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    numerical_columns = [col for col in numerical_columns if col != target_column]
    
    # Step 4: For each numerical column, calculate the 2.5th and 97.5th percentiles
    for col in numerical_columns:
        lower_bound = df[col].quantile(0.005) 
        upper_bound = df[col].quantile(0.995)
        
        # Step 5: Remove rows outside the bounds
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    return df


def create_interactions(df):
    """
    Create interaction features and new columns.
    
    Parameters:
    - df: DataFrame to modify.
    
    Returns:
    - df: DataFrame with new interaction features.
    """
    # Create interaction features
    df['wind_forest_interaction'] = df['max_wind_vel'] * df['forest']
    df['urban_proximity'] = df['distance_roads'] * df['urban']

    # Heat index
    df['heat_index'] = df['avg_temp'] + (df['avg_rel_hum'] * df['avg_temp'])

    # Cumulative distance
    df['cumulative_distance'] = (1 / df['distance_roads']) + (1 / df['distance_fire_stations']) + (1 / df['distance_rivers'])

    return df


def apply_pca(df):
    """
    Apply PCA to distance features and add the components back to the DataFrame.
    
    Parameters:
    - df: DataFrame to transform.
    
    Returns:
    - df: DataFrame with PCA components.
    """
    # Select distance features
    distance_features = df[PCA_COLS]

    # Standardize the features
    scaler = StandardScaler()
    distance_features_scaled = scaler.fit_transform(distance_features)

    # Apply PCA
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(distance_features_scaled)

    # Add the PCA components back to the dataframe
    df['pca_dist_1'] = pca_components[:, 0]
    df['pca_dist_2'] = pca_components[:, 1]

    # Drop original PCA related features
    df.drop(columns=PCA_COLS, inplace=True)

    return df


def apply_smote(df, target_column):

    X = df.drop(columns=[target_column])
    y = df[target_column]

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled



def engineer_features(df, target_column):

    # create interactions
    df = create_interactions(df)

    # handling multicollinearity
    df = apply_pca(df)

    # feature scaling
    df = feature_scaling(df)

    # outlier removal
    df_cleaned = clean_and_remove_outliers(df[KEY_COLUMNS], target_column='ignition')

    # feature selection
    selected_features = feature_selection_rfe(df_cleaned, target_column='ignition', num_features=10)

    # balancing the dataset
    # X_resampled, y_resampled = apply_smote(df_selected, 'ignition')

    df_cleaned = df_cleaned[selected_features + [target_column]]

    return df_cleaned
