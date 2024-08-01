import pandas as pd

# Load the data from CSV
df = pd.read_csv('./高邮/actual test/上行data.csv', parse_dates=['data_period'])

# Set 'data_period' as the index
df.set_index('data_period', inplace=True)

# Drop duplicates in the index (keeping the first occurrence)
df = df[~df.index.duplicated(keep='first')]

# Generate a complete date range based on the existing data
full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='h')

# Reindex the DataFrame to fill in the missing periods
df = df.reindex(full_range)

# Fill missing 'total' values with 0
df['total'] = df['total'].fillna(0)

# Convert 'total' column to integers to avoid float representation
df['total'] = df['total'].astype(int)

# Reset the index
df.reset_index(inplace=True)

# Rename the index to 'data_period'
df.rename(columns={'index': 'data_period'}, inplace=True)

# Save the filled DataFrame back to a CSV file
df.to_csv('./高邮/actual test/上行data2.csv', index=False)

print("Missing lines have been filled and saved.")
