# ==================================================
# Regression Models: Car Price Prediction (ANN)
# ==================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib

# 1. Load Dataset
# Ensure 'Automobile_data.csv' is in the same directory
df = pd.read_csv('Automobile_data.csv', na_values=['?'])

# 2. Data Cleaning
df = df.dropna(subset=['price']) # Price is our target

# Impute missing numerical values
num_cols = df.select_dtypes(include=['int64','float64']).columns
num_imputer = SimpleImputer(strategy='median')
df[num_cols] = num_imputer.fit_transform(df[num_cols])

# Impute missing categorical values
cat_cols = df.select_dtypes(include=['object']).columns
cat_imputer = SimpleImputer(strategy='most_frequent')
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

# One-hot encoding
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# 3. Split Data
X = df.drop('price', axis=1)
y = df['price'].astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Build Neural Network Model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1) # Output layer for regression
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

# 6. Training
history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=150,
    batch_size=32,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# 7. Evaluation
y_pred = model.predict(X_test_scaled).flatten()

print("\n--- Model Evaluation ---")
print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

# 7. Evaluation
y_pred = model.predict(X_test_scaled).flatten()

print("\n--- Model Evaluation ---")
print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

# 8. Visualization: Model Performance (Loss Curve & Scatter Plot)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Grafik 1: Training vs Validation Loss (Loss Curve)
axes[0].plot(history.history['loss'], label='Train Loss', color='blue', linewidth=2)
axes[0].plot(history.history['val_loss'], label='Val Loss', color='orange', linewidth=2)
axes[0].set_title('Model Loss (Training vs Validation)', fontweight='bold')
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Loss (MSE)')
axes[0].legend()

# Grafik 2: Actual vs Predicted Prices (Scatter Plot)
axes[1].scatter(y_test, y_pred, alpha=0.6, color='teal')
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1].set_title("Actual vs Predicted Car Prices", fontweight='bold')
axes[1].set_xlabel("Actual Price")
axes[1].set_ylabel("Predicted Price")

plt.tight_layout()
plt.savefig("ann_performance_charts.png")
plt.show()

# 9. Save Model and Scaler
model.save("car_price_model.h5")
joblib.dump(scaler, "scaler_ann.pkl")
print("\nModel and Scaler saved!")

# 9. Save Model and Scaler
model.save("car_price_model.h5")
joblib.dump(scaler, "scaler_ann.pkl")
print("\nModel and Scaler saved!")
