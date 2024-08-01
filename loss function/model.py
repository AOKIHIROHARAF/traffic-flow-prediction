import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the data from CSV file
data = pd.read_csv('./高邮/6-7/svr/weekday_train_data.csv')

# Parse the time column to extract the hour
data['time'] = pd.to_datetime(data['time'], format='%H:%M:%S').dt.hour

# Prepare to store the coefficients and predictions
coefficients = []
predictions = []

# Define the custom loss function
def loss_function(coefs, X, y):
    a, b = coefs
    predictions = a * X[:, 0] + b * X[:, 1]
    return np.mean((y - predictions) ** 2)

# Loop over each hour
unique_times = data['time'].unique()
for hour in unique_times:
    # Select data for the current hour
    hourly_data = data[data['time'] == hour]
    
    # Prepare features and target
    X = hourly_data[['total_up', 'total_down']].values
    y_target = hourly_data['total_toll'].values
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initial guess for the coefficients
    initial_coefs = np.array([1, 1])
    
    # Define the constraints to ensure coefficients are positive
    constraints = ({'type': 'ineq', 'fun': lambda coefs: coefs})
    
    # Perform the constrained optimization
    result = minimize(loss_function, initial_coefs, args=(X_scaled, y_target), constraints=constraints)
    
    # Extract the optimized coefficients
    opt_coefs = result.x
    
    # Store the coefficients with the corresponding hour
    coefficients.append({'hour': hour, 'coef_total_up': opt_coefs[0], 'coef_total_down': opt_coefs[1]})
    
    # Calculate the predictions
    y_pred = opt_coefs[0] * X_scaled[:, 0] + opt_coefs[1] * X_scaled[:, 1]
    predictions.append({'hour': hour, 'predictions': y_pred, 'actual': y_target})

# Convert coefficients to DataFrame for analysis
coeff_df = pd.DataFrame(coefficients)
pred_df = pd.DataFrame(predictions)

# Plot the coefficients over time
plt.figure(figsize=(14, 7))
plt.plot(coeff_df['hour'], coeff_df['coef_total_up'], label='Coefficient of total_up', marker='o')
plt.plot(coeff_df['hour'], coeff_df['coef_total_down'], label='Coefficient of total_down', marker='o')
plt.xlabel('Hour of the Day')
plt.ylabel('Coefficient')
plt.title('Pattern of Coefficients over Different Times of the Day')
plt.legend()
plt.grid(True)
plt.show()

# Plot predictions vs actual
for hour_data in predictions:
    hour = hour_data['hour']
    plt.figure(figsize=(10, 5))
    plt.plot(hour_data['actual'], label='Actual')
    plt.plot(hour_data['predictions'], label='Predicted')
    plt.xlabel('Sample')
    plt.ylabel('Total Toll')
    plt.title(f'Actual vs Predicted Total Toll at Hour {hour}')
    plt.legend()
    plt.show()
