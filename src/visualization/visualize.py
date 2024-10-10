import matplotlib.pyplot as plt
import seaborn as sns

# Histograms for Numerical Variables
def plot_histograms(df, numerical_columns):
    """
    Plots histograms for numerical variables in the DataFrame.

    Parameters:
    - df: DataFrame containing the data.
    - numerical_columns: List of numerical column names to plot.
    """
    df[numerical_columns].hist(figsize=(15, 15), bins=20, color='skyblue', edgecolor='black')
    plt.suptitle('Histograms of Numerical Features', fontsize=20)
    plt.show()

# Correlation Analysis
def correlation_analysis(df, numerical_columns):
    """
    Plots a heatmap of the correlation matrix for numerical variables.

    Parameters:
    - df: DataFrame containing the data.
    - numerical_columns: List of numerical column names to include in the correlation matrix.
    """
    corr_matrix = df[numerical_columns].corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix of Numerical Features', fontsize=18)
    plt.show()

# Check Class Balance
def check_class_balance(df, target='ignition'):
    """
    Plots a bar chart showing the class balance for the target variable.

    Parameters:
    - df: DataFrame containing the data.
    - target: The target variable to analyze (default is 'ignition').
    """
    class_counts = df[target].value_counts()
    sns.barplot(x=class_counts.index, y=class_counts.values, palette="Blues")
    plt.title(f'Class Balance of {target}', fontsize=18)
    plt.ylabel('Count')
    plt.xlabel(f'{target}')
    plt.xticks([0, 1], ['Non-Ignition', 'Ignition'])
    plt.show()
