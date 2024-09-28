
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
df = pd.read_csv('C:/automatic_vehicular_control/datasets/Next_Generation_Simulation__NGSIM__Vehicle_Trajectories_and_Supporting_Data.csv')


# Extract relevant columns
df = df[['Vehicle_ID', 'Frame_ID', 'v_Vel', 'Local_X', 'Local_Y', 'Space_Headway', 'v_Acc']]

# Clean the data (handle missing values)
df_cleaned = df.dropna()

# Sort the data by Vehicle_ID and Frame_ID
df_sorted = df_cleaned.sort_values(by=['Vehicle_ID', 'Frame_ID'])

# Print the first few rows of the sorted data to verify the sorting and grouping
print(df_sorted.head(20))

# Calculate relative position to the vehicle ahead
df_sorted['Relative_Position_X'] = df_sorted.groupby('Vehicle_ID')['Local_X'].diff().fillna(0)
df_sorted['Relative_Position_Y'] = df_sorted.groupby('Vehicle_ID')['Local_Y'].diff().fillna(0)

# Initialize the scaler
scaler = MinMaxScaler()

# Normalize the data
df_normalized = pd.DataFrame(scaler.fit_transform(df_sorted[['v_Vel', 'Relative_Position_X', 'Relative_Position_Y', 'Space_Headway', 'v_Acc']]), 
                             columns=['v_Vel', 'Relative_Position_X', 'Relative_Position_Y', 'Space_Headway', 'v_Acc'])

# Add back the Vehicle_ID and Frame_ID columns
df_normalized['Vehicle_ID'] = df_sorted['Vehicle_ID'].values
df_normalized['Frame_ID'] = df_sorted['Frame_ID'].values

# Save the processed data to a new CSV file
df_normalized.to_csv('C:/automatic_vehicular_control/datasets/processed_ngsim.csv', index=False)

# Save the scaler for later de-normalization
import joblib
joblib.dump(scaler, 'scaler.save')

# Example of normalized data
print(df_normalized.head())













