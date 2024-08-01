import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression

# Helper function to convert time strings to total minutes
def time_to_minutes(time_str):
    h, m, s = map(int, time_str.split(':'))
    return h * 60 + m + s / 60

# Read the CSV file
df = pd.read_csv('./高邮/6-7/poly/weekend_train_data.csv')

# Prepare the data
times_str = df['time'].values
times = df['time'].apply(time_to_minutes).values
numbers = df['ratio'].values

# Convert to numpy arrays
X = np.array(times).reshape(-1, 1)
y = np.array(numbers).reshape(-1, 1)

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Try different polynomial degrees
for degree in range(3, 9):
    # Transform the data for polynomial regression
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    # Create and train the polynomial regression model
    model_poly = LinearRegression()
    model_poly.fit(X_poly, y)

    # Predict values (for plotting)
    predicted_y_poly = model_poly.predict(X_poly)

    # Plot the data and the polynomial model
    plt.figure(figsize=(10, 6))
    plt.scatter(X, numbers, color='blue', label='Data points')
    sorted_indices = np.argsort(times)
    plt.plot(X[sorted_indices], predicted_y_poly[sorted_indices], color='red', label=f'Polynomial regression curve (degree={degree})')
    plt.xticks(ticks=X[sorted_indices][::int(len(X)/10)].ravel(), labels=times_str[sorted_indices][::int(len(X)/10)], rotation=45)  # Set x-ticks to show a subset of labels for clarity
    plt.xlabel('Time')
    plt.ylabel('Ratio')
    plt.legend()
    plt.title(f'Polynomial Regression (degree={degree})')
    plt.tight_layout()  # Adjust layout to prevent label overlap
    plt.show()
