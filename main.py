import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Dataset Creation (House Area vs. Price)
# This represents a typical Predictive Modeling task (Regression)
data = {
    'Area_sqft': [1100, 1500, 1800, 2200, 2700, 3000, 3500, 4000, 4400, 5000],
    'Price_USD': [250000, 320000, 380000, 450000, 520000, 600000, 680000, 750000, 820000, 900000]
}
df = pd.DataFrame(data)

# 2. Features (X) and Target (y)
X = df[['Area_sqft']] # Independent variable
y = df['Price_USD']    # Dependent variable (What we want to predict)

# 3. Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Model Initialization and Training
# Using Linear Regression for numerical forecasting
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Prediction on Test Set
y_pred = model.predict(X_test)

# 6. Evaluation Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score (Accuracy): {r2*100:.2f}%")

# 7. Visualizing the results (The Regression Line)
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Actual Data Points')
plt.plot(X, model.predict(X), color='red', linewidth=2, label='Predictive Line')
plt.title('House Price Prediction Model (Chapter 1)')
plt.xlabel('Area (sqft)')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.show()
