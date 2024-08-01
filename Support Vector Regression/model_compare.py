import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

# Load the data from CSV file
data = pd.read_csv('./高邮/6-7/svr/weekday_train_data.csv')

# Parse the time column to extract the hour
data['time'] = pd.to_datetime(data['time'], format='%H:%M:%S').dt.hour

# Prepare to store the coefficients and predictions
coefficients = []
predictions = []

# Hyperparameter tuning
param_grid = {
    'C': [0.1, 1, 10, 100],
    'epsilon': [0.01, 0.1, 1],
    'gamma': ['scale', 'auto']
}

# Loop over each hour
unique_times = data['time'].unique()
for hour in unique_times:
    # Select data for the current hour
    hourly_data = data[data['time'] == hour]
    
    # Prepare features and target
    X = hourly_data[['total_up', 'total_down']]
    y_target = hourly_data['total_toll']
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train SVR model with hyperparameter tuning
    svr = SVR(kernel='rbf')
    grid_search = GridSearchCV(svr, param_grid, cv=3)
    grid_search.fit(X_scaled, y_target)
    
    # Best model
    best_svr = grid_search.best_estimator_
    
    # Get the support vectors and their coefficients
    support_vectors = best_svr.support_vectors_
    dual_coef = best_svr.dual_coef_
    
    # Calculate the approximate coefficients for total_up and total_down
    approx_coef = np.dot(dual_coef, support_vectors).flatten()
    
    # Store the coefficients with the corresponding hour
    coefficients.append({'hour': hour, 'coef_total_up': approx_coef[0], 'coef_total_down': approx_coef[1]})
    
    # Make predictions for the current hour
    y_pred = best_svr.predict(X_scaled)
    
    # Store predictions and actual values for plotting
    predictions.append({'hour': hour, 'actual': y_target.values, 'predicted': y_pred})

# Convert coefficients to DataFrame for analysis
coeff_df = pd.DataFrame(coefficients)

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

# Plot predictions vs actual for each hour
for pred in predictions:
    hour = pred['hour']
    plt.figure(figsize=(10, 5))
    plt.plot(pred['actual'], label='Actual', marker='o')
    plt.plot(pred['predicted'], label='Predicted', marker='x')
    plt.xlabel('Sample')
    plt.ylabel('Total Toll')
    plt.title(f'Actual vs Predicted Total Toll at Hour {hour}')
    plt.legend()
    plt.show()
