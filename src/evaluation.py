import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# -----------------------------
# Load Cleaned Dataset
# -----------------------------

df = pd.read_csv("dataset/cleaned_housing.csv")

# -----------------------------
# Features and Target
# -----------------------------

X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

# -----------------------------
# Train Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Load Saved Model
# -----------------------------

model = joblib.load("models/house_price_model.pkl")

# -----------------------------
# Predictions
# -----------------------------

y_pred = model.predict(X_test)

# -----------------------------
# Evaluation Metrics
# -----------------------------

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("=" * 40)
print("MODEL PERFORMANCE")
print("=" * 40)

print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R² Score : {r2:.4f}")

# -----------------------------
# Actual vs Predicted
# -----------------------------

plt.figure(figsize=(8,6))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Price")

plt.ylabel("Predicted Price")

plt.title("Actual vs Predicted House Prices")

plt.savefig("images/actual_vs_predicted.png")

plt.show()

# -----------------------------
# Residual Plot
# -----------------------------

residual = y_test - y_pred

plt.figure(figsize=(8,6))

plt.scatter(y_pred, residual)

plt.axhline(y=0, linestyle="--")

plt.xlabel("Predicted Price")

plt.ylabel("Residual")

plt.title("Residual Plot")

plt.savefig("images/residual_plot.png")

plt.show()

# -----------------------------
# Feature Importance
# -----------------------------

importance = model.feature_importances_

feature_importance = (
    pd.DataFrame({
        "Feature": X.columns,
        "Importance": importance
    })
    .sort_values("Importance", ascending=False)
)

print("\nTop 10 Important Features\n")

print(feature_importance.head(10))

plt.figure(figsize=(10,6))

plt.barh(
    feature_importance["Feature"][:10],
    feature_importance["Importance"][:10]
)

plt.title("Top 10 Important Features")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig("images/feature_importance.png")

plt.show()

print("\n✅ Evaluation Completed Successfully!")