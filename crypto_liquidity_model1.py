import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- Setup ---
os.makedirs("reports", exist_ok=True)

# --- Load Data ---
df1 = pd.read_csv("coin_gecko_2022-03-16.csv")
df2 = pd.read_csv("coin_gecko_2022-03-17.csv")

# --- Merge and Clean ---
df = pd.concat([df1, df2], ignore_index=True)
df.dropna(inplace=True)
df['date'] = pd.to_datetime(df['date'])

# --- Normalize Numerical Features ---
scaler = StandardScaler()
num_cols = ['price', '1h', '24h', '7d', '24h_volume', 'mkt_cap']
df[num_cols] = scaler.fit_transform(df[num_cols])

# Save target scaler separately
volume_scaler = StandardScaler()
df['24h_volume_raw'] = df['24h_volume']
df['24h_volume'] = volume_scaler.fit_transform(df[['24h_volume']])
pickle.dump(volume_scaler, open("volume_scaler.pkl", "wb"))

# --- Feature Engineering ---
df['price_change_score'] = df['1h'] * 0.2 + df['24h'] * 0.3 + df['7d'] * 0.5
df['volume_to_marketcap'] = df['24h_volume'] / (df['mkt_cap'] + 1e-6)

# --- EDA Section ---
print("\n--- Dataset Summary Statistics ---")
print(df.describe())

# Correlation Heatmap (Fix for string columns like 'coin')
plt.figure(figsize=(10, 8))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap='rocket', fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("reports/eda_correlation_heatmap.png")
plt.show()

# Price Distribution
plt.figure(figsize=(12, 6))
sns.histplot(df['price'], bins=30, kde=True, color='skyblue')
plt.title("Price Distribution")
plt.tight_layout()
plt.savefig("reports/eda_price_distribution.png")
plt.show()

# Volume Distribution
plt.figure(figsize=(12, 6))
sns.histplot(df['24h_volume'], bins=30, kde=True, color='lightgreen')
plt.title("24h Volume Distribution")
plt.tight_layout()
plt.savefig("reports/eda_volume_distribution.png")
plt.show()

# Market Cap vs Volume
plt.figure(figsize=(10, 6))
sns.scatterplot(x='mkt_cap', y='24h_volume', data=df, alpha=0.5)
plt.title("Market Cap vs 24h Volume")
plt.tight_layout()
plt.savefig("reports/eda_volume_vs_mktcap.png")
plt.show()

# Price Change Score Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['price_change_score'], kde=True, color='orange')
plt.title("Price Change Score Distribution")
plt.tight_layout()
plt.savefig("reports/eda_price_change_score.png")
plt.show()

# --- Model Training ---
features = ['price', 'price_change_score', 'volume_to_marketcap', 'mkt_cap']
X = df[features]
y = df['24h_volume']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

params = {
    'max_depth': [3, 5],
    'n_estimators': [100, 200],
    'learning_rate': [0.1, 0.2]
}

grid = GridSearchCV(XGBRegressor(objective='reg:squarederror'), params, cv=3, scoring='r2')
grid.fit(X_train, y_train)
model = grid.best_estimator_

# --- Evaluation ---
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation ---")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# Save model
pickle.dump(model, open("xgb_model.pkl", "wb"))
