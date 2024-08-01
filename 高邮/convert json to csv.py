import pandas as pd
import json

# Load the JSON file
json_file_path = './高邮/actual test/收费站.json' 

with open(json_file_path, 'r') as file:
    json_data = json.load(file)

# Extract the 'data' part
data = json_data['data']

# Normalize the nested JSON structure
data_normalized = pd.json_normalize(data, record_path=['out'])   #out,tf

"""
# Convert the DataFrame to a CSV file
csv_file_path = '高邮南out_total.csv'  
data_normalized.to_csv(csv_file_path, index=False)
"""

# Group by 'data_period' and sum the 'total'
grouped_data = data_normalized.groupby('data_period', as_index=False)['total'].sum()

# Save the result to a new CSV file
grouped_csv_file_path = './高邮/actual test/收费站data.csv' 
grouped_data.to_csv(grouped_csv_file_path, index=False)

print(f"Grouped data has been successfully saved to {grouped_csv_file_path}")

