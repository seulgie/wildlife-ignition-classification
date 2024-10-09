from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# Function to separate features and target variable
def separate_features_target(df, target_column):
    """
    Separate features and target variable from the DataFrame.

    Parameters:
    - df: DataFrame containing features and target.
    - target_column: The name of the target column.

    Returns:
    - X: Feature DataFrame (without the target column).
    - y: Target Series.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

# Function to define machine learning models
def define_models():
    """
    Define machine learning models.

    Returns:
        dict: A dictionary of models.
    """
    return {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

# Function to define hyperparameter grids for model tuning
def define_param_grids():
    """
    Define hyperparameter grids for model tuning.

    Returns:
        dict: A dictionary of hyperparameter grids.
    """
    return {
        'Logistic Regression': {
            'C': [0.1, 1],
            'solver': ['lbfgs']
        },
        'Random Forest': {
            'n_estimators': [50, 100],
            'max_depth': [None, 10]
        },
        'XGBoost': {
            'n_estimators': [50, 100],
            'learning_rate': [0.1],
            'max_depth': [3, 5]
        }
    }

# Function to tune and train models using GridSearchCV
def train_models(df, target_column):
    """
    Split the data into training and testing sets and train models using GridSearchCV.

    Args:
        df (pd.DataFrame): The full dataset containing features and target variable.
        target_column (str): The name of the target column.

    Returns:
        dict: Best models found during tuning.
        pd.DataFrame: The test features.
        pd.Series: The test target variable.
    """
    # Separate features and target variable
    X, y = separate_features_target(df, target_column)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define models and their hyperparameter grids
    models = define_models()
    param_grids = define_param_grids()
    best_models = {}
    
    # Loop through each model and perform Grid Search
    for name, model in models.items():
        print(f"Tuning {name}...")
        grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        # Store the best model
        best_models[name] = grid_search.best_estimator_
        print(f"Best parameters for {name}: {grid_search.best_params_}")

    return best_models, X_test, y_test
