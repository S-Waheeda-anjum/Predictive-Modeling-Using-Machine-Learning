import pandas as pd

# Load dataset
df = pd.read_csv("dataset/housing.csv")

print("Dataset Shape:", df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum().sort_values(ascending=False).head(20))

# Remove duplicate rows
df = df.drop_duplicates()

# Fill missing numeric values with median
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Fill missing categorical values with mode
categorical_cols = df.select_dtypes(include=["object"]).columns

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Convert categorical columns into numbers
df = pd.get_dummies(df, drop_first=True)

print("\nNew Shape After Encoding:")
print(df.shape)

# Save cleaned dataset
df.to_csv("dataset/cleaned_housing.csv", index=False)

print("\n✅ Preprocessing Completed Successfully!")