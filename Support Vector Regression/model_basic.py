import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the data from CSV file
data = pd.read_csv('./高邮/6-7/svr/weekday_train_data.csv')

# Parse the time column to extract the hour
data['time'] = pd.to_datetime(data['time'], format='%H:%M:%S').dt.hour

# Prepare to store the coefficients
coefficients = []

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
    
    # Train SVR model
    svr = SVR(kernel='rbf')
    svr.fit(X_scaled, y_target)
    
    # Get the support vectors and their coefficients
    support_vectors = svr.support_vectors_
    dual_coef = svr.dual_coef_
    
    # Calculate the approximate coefficients for total_up and total_down
    approx_coef = np.dot(dual_coef, support_vectors).flatten()
    
    # Store the coefficients with the corresponding hour
    coefficients.append({'hour': hour, 'coef_total_up': approx_coef[0], 'coef_total_down': approx_coef[1]})

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
