import pandas as pd

# Load the updated weather data
weather_df = pd.read_csv('queensWEATHER10yrs.csv')
weather_df['datetime'] = pd.to_datetime(weather_df['datetime'])

# Filter for summer months (June, July, August) and for Manhattan only
weather_summer_df = weather_df[(weather_df['datetime'].dt.month.isin([6, 7, 8])) & (weather_df['name'] == 'queens')]

# Select only numeric columns needed for aggregation
numeric_columns = ['temp', 'visibility', 'winddir', 'windspeed', 'precip', 'solarradiation', 'cloudcover', 'humidity']
weather_summer_numeric = weather_summer_df[numeric_columns]

# Aggregate the numeric weather data by date, taking the average of each parameter
weather_aggregated = weather_summer_numeric.groupby(weather_summer_df['datetime'].dt.date).mean()
weather_aggregated.reset_index(inplace=True)
weather_aggregated.rename(columns={'index': 'datetime'}, inplace=True)
weather_aggregated['datetime'] = pd.to_datetime(weather_aggregated['datetime'])

# Initialize an empty DataFrame to store all merged data
all_years_merged_df = pd.DataFrame()

# Process and merge the data for each year from 2013 to 2023
for year in range(2013, 2024):
    pm25_file_name = f'REALQUEENS{year}.csv'
    
    try:
        # Load the PM2.5 data
        pm25_df = pd.read_csv(pm25_file_name)
        pm25_df['Date'] = pd.to_datetime(pm25_df['Date'], format='%m/%d/%y')

        # Select specific columns from PM2.5 data
        pm25_selected = pm25_df[['Date', 'Daily Mean PM2.5 Concentration']]

        # Merge with the aggregated weather data
        merged_df = pd.merge(pm25_selected, weather_aggregated, left_on='Date', right_on='datetime', how='inner')

        # Append to the all_years_merged_df
        all_years_merged_df = pd.concat([all_years_merged_df, merged_df], ignore_index=True)
    except FileNotFoundError:
        print(f"File not found: {pm25_file_name}")

# Select only the specified columns in the final DataFrame
selected_columns = ['Date', 'Daily Mean PM2.5 Concentration', 'temp', 'visibility', 'winddir', 'windspeed', 'precip', 'solarradiation', 'cloudcover', 'humidity']
all_years_merged_df = all_years_merged_df[selected_columns]

# Save the final merged DataFrame to a CSV file
merged_csv_file_path = 'QUEENSsummersmerged.csv'
all_years_merged_df.to_csv(merged_csv_file_path, index=False)

# Display the shape of the final merged DataFrame and confirmation message
print("Final DataFrame shape:", all_years_merged_df.shape)
print(f"Merged data saved to {merged_csv_file_path}")
