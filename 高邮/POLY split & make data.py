import pandas as pd

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('./高邮/6-7/上下行data.csv')

# Convert the 'data_period' column to datetime format
df['data_period'] = pd.to_datetime(df['data_period'])

# Determine if each date is a weekday or weekend
df['is_weekend'] = df['data_period'].dt.weekday >= 5

# Split the data into two DataFrames
weekday_df = df[df['is_weekend'] == False]
weekend_df = df[df['is_weekend'] == True]

# Drop the 'is_weekend' column as it is no longer needed
weekday_df = weekday_df.drop(columns=['is_weekend'])
weekend_df = weekend_df.drop(columns=['is_weekend'])

# Extract the time part from the 'data_period' column and create a new column 'time'
weekday_df['time'] = pd.to_datetime(df['data_period']).dt.time
weekend_df['time'] = pd.to_datetime(df['data_period']).dt.time

# Create a new DataFrame with only the 'time' and 'ratio' columns
new_weekday_df = weekday_df[['time', 'total_sum', 'ratio']]
new_weekend_df = weekend_df[['time', 'total_sum', 'ratio']]

# Save the two DataFrames to separate CSV files
new_weekday_df.to_csv('./高邮/6-7/weekday_train_data.csv', index=False)
new_weekend_df.to_csv('./高邮/6-7/weekend_train_data.csv', index=False)



