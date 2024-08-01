import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score

# Load the data from the CSV file
file_path = './Graph Convolutional Network/total_actual_vs_predicted.csv'
data = pd.read_csv(file_path)

# Extract actual and predicted values
y_true = data['Actual']
y_pred = data['Predicted']

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

# Calculate MAE
mae = mean_absolute_error(y_true, y_pred)

# Calculate Accuracy
accuracy = 1 - (np.linalg.norm(y_true - y_pred) / np.linalg.norm(y_true))

# Calculate R2 Score
r2 = r2_score(y_true, y_pred)

# Calculate Explained Variance Score
explained_variance = explained_variance_score(y_true, y_pred)

# Print the results
print(f'RMSE: {rmse}')
print(f'MAE: {mae}')
print(f'Accuracy: {accuracy}')
print(f'R2: {r2}')
print(f'Explained Variance: {explained_variance}')
