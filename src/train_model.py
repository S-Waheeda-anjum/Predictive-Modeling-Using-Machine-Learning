import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load cleaned dataset
df = pd.read_csv("dataset/cleaned_housing.csv")

print("Dataset Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

# -----------------------------
# Select Target Variable
# -----------------------------

target = "SalePrice"

if target not in df.columns:
    raise ValueError(f"Target column '{target}' not found!")

# Features and Target
X = df.drop(target, axis=1)
y = df[target]

# -----------------------------
# Train-Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(f"\nTraining Samples: {len(X_train)}")
print(f"Testing Samples: {len(X_test)}")

# -----------------------------
# Train Random Forest Model
# -----------------------------

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

print("\n✅ Model Trained Successfully!")

# Save Model
joblib.dump(model, "models/house_price_model.pkl")

print("✅ Model Saved Successfully!")