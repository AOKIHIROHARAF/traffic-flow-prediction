import pandas as pd

df1 = pd.read_csv('./高邮/6-7/上行data.csv')
df2 = pd.read_csv('./高邮/6-7/下行data.csv')
df3 = pd.read_csv('./高邮/6-7/高邮南out_total.csv')

merged_df = pd.merge(df1, df2, on='data_period', suffixes=('_up', '_down'))
merged_df = pd.merge(merged_df, df3, on='data_period')
merged_df.rename(columns={'total': 'total_toll'}, inplace=True)

merged_df['data_period'] = pd.to_datetime(merged_df['data_period']).dt.time

"""
merged_df['data_period'] = pd.to_datetime(merged_df['data_period'])

# Determine if each date is a weekday or weekend
merged_df['is_weekend'] = merged_df['data_period'].dt.weekday >= 5

# Split the data into two DataFrames
weekday_df = merged_df[merged_df['is_weekend'] == False]
weekend_df = merged_df[merged_df['is_weekend'] == True]

# Drop the 'is_weekend' column as it is no longer needed
weekday_df = weekday_df.drop(columns=['is_weekend'])
weekend_df = weekend_df.drop(columns=['is_weekend'])

# Extract the time part from the 'data_period' column and create a new column 'time'
weekday_df['time'] = pd.to_datetime(merged_df['data_period']).dt.time
weekend_df['time'] = pd.to_datetime(merged_df['data_period']).dt.time

# Create a new DataFrame with only the 'time' and 'ratio' columns
new_weekday_df = weekday_df[['time', 'total_up', 'total_down', 'total_toll']]
new_weekend_df = weekend_df[['time', 'total_up', 'total_down', 'total_toll']]

# Save the two DataFrames to separate CSV files
new_weekday_df.to_csv('./高邮/actual test/train data/weekday_train_data.csv', index=False)
new_weekend_df.to_csv('./高邮/actual test/train data/weekend_train_data.csv', index=False)
"""

merged_df.to_csv('./高邮/actual test/test data/total_test_data.csv', index=False)

