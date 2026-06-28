import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/house_price_model.pkl")

# Load cleaned dataset
df = pd.read_csv("dataset/cleaned_housing.csv")

# Remove target column
X = df.drop("SalePrice", axis=1)

# Take first house as a sample
sample_house = X.iloc[[0]]

# Predict price
predicted_price = model.predict(sample_house)

print("=" * 40)
print("HOUSE PRICE PREDICTION")
print("=" * 40)
print(f"Predicted House Price: ${predicted_price[0]:,.2f}")