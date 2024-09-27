# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# Load the dataset from your local drive
df = pd.read_csv(r'C:\Users\akram\Downloads\road_saftety_data.csv')

# Display the first few rows of the dataset to understand its structure
print(df.head())

# 1. Total number of rows and columns
total_rows, total_columns = df.shape
print(f"Total rows: {total_rows}, Total columns: {total_columns}")

# 2. Number of missing values
missing_values = df.isnull().sum().sum()
print(f"Total missing values: {missing_values}")

# 3. Number of unique values in each column
unique_values = df.nunique()
total_values = df.count()

# Combire unique and total values into a single DataFrame
unique_vs_total = pd.DataFrame({
    'Unique Values': unique_values,
    'Total Values': total_values
})

print("Unique vs Total values in each column:")
print(unique_vs_total)

# 4. Descriptive statistics for numerical columns (mean, median, std, etc.)
descriptive_stats = df.describe()
print("\nDescriptive statistics:")
print(descriptive_stats)

# 5. Number of distinct categories in categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns
distinct_categories = {col: df[col].nunique() for col in categorical_columns}
print("\nDistinct categories in categorical columns:")
print(distinct_categories)

# Get data types of each column
data_types = df.dtypes
print("\nData types of each column:")
print(data_types)





# Assuming your dataset is already loaded into the 'df' DataFrame

# Step 1: Selecting only numeric columns
numeric_df = df.select_dtypes(include=['int64', 'float64'])

# Step 2: Standardize the data (mean = 0, variance = 1)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_df)

# Step 3: Apply PCA
pca = PCA()
pca.fit(scaled_data)

# Step 4: Calculate the explained variance for each component
explained_variance_ratio = pca.explained_variance_ratio_

# Cumulative explained variance
cumulative_variance = pca.explained_variance_ratio_.cumsum()

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--', color='r')
plt.title('Cumulative Explained Variance by Principal Components')
plt.xlabel('Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.show()


# Get the loadings (contributions of each feature to each principal component)
loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(12)], index=numeric_df.columns)

# Display the loadings
print(loadings)








# 6. Correlation matrix (for numerical columns only)
numeric_df = df.select_dtypes(include=['int64', 'float64'])  # Select only numeric columns
correlation_matrix = numeric_df.corr()

# Create a heatmap of the correlation matrix
plt.figure(figsize=(10, 8))  # Adjust figure size if needed
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix Heatmap")
plt.show()

# 7. Minimum and Maximum values for numerical columns
min_values = df.min(numeric_only=True)
max_values = df.max(numeric_only=True)
print("\nMinimum values for numerical columns:")
print(min_values)
print("\nMaximum values for numerical columns:")
print(max_values)

# Functional dependency approximation using correlation and uniqueness
def check_dependency(df, col1, col2):
    if df.groupby(col1)[col2].nunique().max() == 1:
        return True
    return False

potential_fds = []
for col1 in df.columns:
    for col2 in df.columns:
        if col1 != col2 and check_dependency(df, col1, col2):
            potential_fds.append((col1, col2))

print("\nPotential functional dependencies (if one column determines another):")
print(potential_fds)
