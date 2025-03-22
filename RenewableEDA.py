import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset
df = pd.read_csv('C:\\Users\\Sambhaji\\numpy2\\numpy\\Renewable_Energy_enhanced.csv')

# Step 2: Inspect the dataset
print("First few rows of the dataset:")
print(df.head())
print("\nDataset Info:")
print(df.info())

# Step 3: Handle missing values
# Checking for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Dropping rows with missing values (if necessary, or you can fill with a method like df.fillna())
df = df.dropna()  # Dropping rows with missing values (you can also use df.fillna() if you prefer)

# Step 4: Convert 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y %H:%M')  # Parsing the date correctly
df['Year'] = df['Date'].dt.year  # Extracting the year from the 'Date' column
df['Month'] = df['Date'].dt.month  # Extracting the month for seasonal analysis

# Step 5: Distribution Analysis (histograms for consumption, efficiency, and CO2 reduction)
plt.figure(figsize=(12, 6))

# Energy Consumption Distribution
plt.subplot(1, 3, 1)
sns.histplot(df['Consumption_MWh'], kde=True, color='blue')
plt.title('Distribution of Energy Consumption (MWh)')
plt.xlabel('Energy Consumption (MWh)')
plt.ylabel('Frequency')

# Efficiency Distribution
plt.subplot(1, 3, 2)
sns.histplot(df['Efficiency_Percentage'], kde=True, color='green')
plt.title('Distribution of Efficiency Percentage')
plt.xlabel('Efficiency (%)')
plt.ylabel('Frequency')

# CO2 Reduction Distribution
plt.subplot(1, 3, 3)
sns.histplot(df['CO2_Reduction_kg'], kde=True, color='red')
plt.title('Distribution of CO2 Reduction (kg)')
plt.xlabel('CO2 Reduction (kg)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Step 6: Energy Source Distribution
# Check the unique energy sources
print("\nUnique Energy Sources:")
print(df['Energy_Source'].unique())

# Plot energy source distribution
plt.figure(figsize=(6, 6))
sns.countplot(data=df, x='Energy_Source', palette='Set2')
plt.title('Distribution of Energy Sources')
plt.xlabel('Energy Source')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Step 7: Energy Consumption Trends (Yearly and Monthly)
# Yearly Energy Consumption Trend
yearly_consumption = df.groupby('Year')['Consumption_MWh'].sum()

plt.figure(figsize=(10, 6))
yearly_consumption.plot(kind='line', color='purple', marker='o')
plt.title('Yearly Energy Consumption (MWh)')
plt.xlabel('Year')
plt.ylabel('Total Consumption (MWh)')
plt.tight_layout()
plt.show()

# Monthly Energy Consumption Trend
monthly_consumption = df.groupby('Month')['Consumption_MWh'].sum()

plt.figure(figsize=(10, 6))
monthly_consumption.plot(kind='line', color='orange', marker='o')
plt.title('Monthly Energy Consumption (MWh)')
plt.xlabel('Month')
plt.ylabel('Total Consumption (MWh)')
plt.tight_layout()
plt.show()

# Step 8: Efficiency vs. Energy Consumption (Scatter Plot)
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Consumption_MWh', y='Efficiency_Percentage', color='green')
plt.title('Efficiency vs. Energy Consumption')
plt.xlabel('Energy Consumption (MWh)')
plt.ylabel('Efficiency (%)')
plt.tight_layout()
plt.show()

# Step 9: CO2 Reduction vs. Energy Consumption (Scatter Plot)
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Consumption_MWh', y='CO2_Reduction_kg', color='red')
plt.title('CO2 Reduction vs. Energy Consumption')
plt.xlabel('Energy Consumption (MWh)')
plt.ylabel('CO2 Reduction (kg)')
plt.tight_layout()
plt.show()

# Step 10: Correlation Analysis (Correlation Matrix)
# Correlation matrix between 'Consumption_MWh', 'Efficiency_Percentage', and 'CO2_Reduction_kg'
correlation_matrix = df[['Consumption_MWh', 'Efficiency_Percentage', 'CO2_Reduction_kg']].corr()

# Plot the heatmap of the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix Between Consumption, Efficiency, and CO2 Reduction')
plt.tight_layout()
plt.show()
