import pandas as pd
from ucimlrepo import fetch_ucirepo

# Fetch the rice dataset
rice_cammeo_and_osmancik = fetch_ucirepo(id=545) 

# data (as pandas dataframes) 
X = rice_cammeo_and_osmancik.data.features 
y = rice_cammeo_and_osmancik.data.targets 

# Combining the features and target into a single DataFrame for convenience
df = X.copy()
df['Target'] = y
print(rice_cammeo_and_osmancik.variables)

# Convert all numeric columns to their appropriate types
# This will exclude the target (if categorical) from numeric conversion
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

# Define a function to find and remove outliers using IQR on numeric columns only
def remove_outliers_iqr(df, numeric_columns):
    Q1 = df[numeric_columns].quantile(0.25)  # First quartile (25%)
    Q3 = df[numeric_columns].quantile(0.75)  # Third quartile (75%)
    IQR = Q3 - Q1                            # Interquartile Range

    # Define lower and upper bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Removing outliers from the numeric columns
    df_cleaned = df[~((df[numeric_columns] < lower_bound) | (df[numeric_columns] > upper_bound)).any(axis=1)]

    return df_cleaned

# Remove outliers from the dataset
df_cleaned = remove_outliers_iqr(df, numeric_columns)

# Separate the cleaned features and target
X_cleaned = df_cleaned.drop(columns=['Target'])
y_cleaned = df_cleaned['Target']

# Display the cleaned dataset
print("Cleaned Features:")
print(X_cleaned.head())

print("\nCleaned Target:")
print(y_cleaned.head())

# Print the number of rows before and after removing outliers
print(f"\nOriginal dataset size: {df.shape[0]}")
print(f"Cleaned dataset size: {df_cleaned.shape[0]}")