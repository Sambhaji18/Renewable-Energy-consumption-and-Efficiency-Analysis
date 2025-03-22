import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the dataset
df = pd.read_csv('C:\\Users\\Sambhaji\\numpy2\\numpy\\Renewable_Energy_enhanced.csv')

# Step 2: Check the data types of all columns
print("Data types of each column:")
print(df.dtypes)

# Step 3: Identify columns with non-numeric values (object type)
non_numeric_columns = df.select_dtypes(include=['object']).columns
print("\nColumns with non-numeric values:")
print(non_numeric_columns)

# Step 4: Check the unique values in the non-numeric columns
for column in non_numeric_columns:
    print(f"\nUnique values in '{column}':")
    print(df[column].unique())

# Step 5: Check for missing values (NaN) in the dataset
print("\nMissing values in each column:")
print(df.isna().sum())

# Step 6: Clean the data
# Convert 'Date' to datetime format and extract useful features (year, month, day, hour)
# Ensure 'Date' is in the correct format
try:
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y %H:%M')  # Adjust format if needed
except Exception as e:
    print(f"Error in converting 'Date' column to datetime: {e}")
    # Handle potential issues here or try other formats

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['Hour'] = df['Date'].dt.hour

# Drop the original 'Date' column as it is no longer needed
df_cleaned = df.drop(columns=['Date'])

# Step 7: Convert non-numeric columns to numeric using one-hot encoding (for categorical features)
df_cleaned = pd.get_dummies(df_cleaned, drop_first=True)

# Step 8: Check if missing values are removed
print("\nMissing values after cleaning:")
print(df_cleaned.isna().sum())

# Step 9: Prepare data for model training
# Assuming 'Consumption_MWh' is your target variable
target_column = 'Consumption_MWh'  # Update this to your desired target column

# Ensure target column exists in the cleaned data
if target_column not in df_cleaned.columns:
    print(f"Error: The target column '{target_column}' does not exist in the dataset.")
else:
    X = df_cleaned.drop(target_column, axis=1)  # Drop target column from features
    y = df_cleaned[target_column]  # Target column

    # Step 10: Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 11: Train the Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Step 12: Make predictions and evaluate the model
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')

    # Step 13: Visualizations
    
    # 1. Actual vs Predicted plot
    plt.figure(figsize=(8,6))
    plt.scatter(y_test, y_pred, color='blue', edgecolors='black')
    plt.title('Actual vs Predicted Consumption_MWh')
    plt.xlabel('Actual Consumption_MWh')
    plt.ylabel('Predicted Consumption_MWh')
    plt.grid(True)
    plt.show()

    # 2. Residuals Plot
    residuals = y_test - y_pred
    plt.figure(figsize=(8,6))
    plt.scatter(y_pred, residuals, color='red', edgecolors='black')
    plt.axhline(y=0, color='black', linestyle='--')
    plt.title('Residuals Plot')
    plt.xlabel('Predicted Consumption_MWh')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.grid(True)
    plt.show()

    # 3. Feature Importance (for linear regression, it is simply the coefficients)
    feature_importance = model.coef_
    features = X.columns

    plt.figure(figsize=(10,6))
    sns.barplot(x=feature_importance, y=features, palette='viridis')
    plt.title('Feature Importance (Linear Regression Coefficients)')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Features')
    plt.show()
