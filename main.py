import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Add src to the Python path to import modules
src_path = Path(__file__).parent / 'src'
sys.path.append(str(src_path))

# Import functions from your organized scripts
from data.load_data import load_data, save_data
from data.data_cleaning import preprocess_data
from data.date_processing import process_date_column
from features.feature_engineering import engineer_features
from models.model_training import separate_features_target, define_models, define_param_grids, train_models
from models.evaluate_model import evaluate_models
# from visualization.visualize import perform_eda

# Set the base directory
BASE_DIR = Path().resolve()
DATA_PATH = BASE_DIR / 'data'
RAW_DATA_PATH = DATA_PATH / 'raw'
PROCESSED_DATA_PATH = DATA_PATH / 'processed'

# Load the raw dataset
df_raw = load_data(RAW_DATA_PATH / "dataset.csv")

# Preprocess the 'Date' column (optional)
df_processed = process_date_column(df_raw, date_column='Date')

# Preprocess the data (clean, handle NaN/infinite values, cap extreme values, and scale)
df_preprocessed = preprocess_data(df_processed)

# Save the preprocessed data
cleaned_dataset_path = PROCESSED_DATA_PATH / "cleaned_dataset.csv"
save_data(df_preprocessed, cleaned_dataset_path)

# Load the preprocessed data
df_cleaned = load_data(PROCESSED_DATA_PATH / "cleaned_dataset.csv")

# Perform EDA (optional)
# perform_eda(df_cleaned)

# Feature Engineering
df_featured = engineer_features(df_cleaned)

# Save the feature engineered data
feature_engineered_path = PROCESSED_DATA_PATH / "cleaned_dataset2.csv"
save_data(df_featured, feature_engineered_path)

# Load the feature engineered data
df_engineered = load_data(PROCESSED_DATA_PATH / "cleaned_dataset2.csv")

# Train models and get the best ones
best_models, X_test, y_test = train_models(df_engineered, 'ignition')

# Evaluate models on the test set
evaluation_results = evaluate_models(best_models, X_test, y_test)
