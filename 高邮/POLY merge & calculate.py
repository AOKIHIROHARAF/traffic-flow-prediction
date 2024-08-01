import pandas as pd

# Read the CSV files
file1 = './高邮/6-7/高邮南out_total.csv'
file2 = './高邮/6-7/上行data.csv'
file3 = './高邮/6-7/下行data.csv'

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
df3 = pd.read_csv(file3)

merged_df = pd.merge(df1, df2, on='data_period', suffixes=('_file1', '_file2'))

merged_df = pd.merge(merged_df, df3, on='data_period')

# Rename the 'total' column from df3 to avoid confusion
merged_df.rename(columns={'total': 'total_file3'}, inplace=True)

# Sum the 'total' values from both files
merged_df['total_sum'] = merged_df['total_file2'] + merged_df['total_file3']
merged_df['ratio'] = merged_df['total_file1'] / merged_df['total_sum']

# Keep only the relevant columns
result_df = merged_df[['data_period', 'total_sum', 'ratio']]

# Save the result to a new CSV file
result_df.to_csv('./高邮/6-7/上下行data.csv', index=False)


