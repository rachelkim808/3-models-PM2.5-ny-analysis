import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor  # Import XGBoost's XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('BROOKLYNsummersmerged.csv')
data['Date'] = pd.to_datetime(data['Date'])
#data = data[data['PM25'] <= 35]
data = data.sort_values('Date')  # Sort the data by date

#filter only N,  NW directions
#R2 value is 0.4785
filtered_data = data[((data['winddir'] >=0)  & (data['winddir'] <67.5)) | ((data['winddir'] >=257.5)  & (data['winddir'] <360))  ]

#filter also summer months.
summer_months = [6, 7, 8]
filtered_data_summer = filtered_data[filtered_data['Date'].dt.month.isin(summer_months)]
print(filtered_data_summer)

filtered_data_summer.to_csv('filtered_data_dir_summer.csv', index=False)

# Select Features and Target
features = ['temp',	'visibility',	'winddir', 'windspeed','precip','humidity',	
            'solarradiation',	'cloudcover'] #independent varaibles
target = 'Daily Mean PM2.5 Concentration' #dependent variable


X = filtered_data[features]
y = filtered_data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=30)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = XGBRegressor(n_estimators=100, random_state=30)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

# Accuracy-like metric calculation
tolerance = 5  # Define a tolerance level (e.g., within ±5 units)
correct_predictions = np.sum(np.abs(y_test - y_pred) <= tolerance)
total_predictions = len(y_test)
accuracy_like_metric = correct_predictions / total_predictions
print(f'Accuracy-like metric (with tolerance ±{tolerance}): {accuracy_like_metric:.2f}')

feature_name_mapping = {
    'windspeed': 'WS',
    'winddir': 'WD',
    'temp': 'Temp',
    'visibility': 'Vis',
    'humidity': 'RH',
    'solarradiation': 'Solar',
    'cloudcover': 'Cloud',
    'precip': 'Rain',
    'Daily Mean PM2.5 Concentration': 'PM2.5'
}

feature_importance = model.feature_importances_
sorted_idx = feature_importance.argsort()

# Use mapped feature names for plots
mapped_features = [feature_name_mapping.get(feature, feature) for feature in features]

# Save and Show Feature Importance Plot
plt.barh(np.array(mapped_features)[sorted_idx], feature_importance[sorted_idx])
plt.xlabel('Importance', fontsize=20)
plt.ylabel('Feature', fontsize=20)
plt.title('Feature Importance from GXBoost', fontsize=20)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.savefig('feature_importance_GXBoost.tiff', format='tiff', dpi=300)
plt.show()

corr_matrix = data[features + [target]].corr()
plt.figure(figsize=(8,5))  # Adjust size to make sure text is not cluttered
sns.heatmap(corr_matrix.rename(columns=feature_name_mapping, index=feature_name_mapping),
            annot=True, cmap='coolwarm', fmt=".2f", annot_kws={'size': 12})  # Adjust annot_kws for bigger font size in annotations
plt.title('Correlation Heatmap from GXBoost', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xticks(rotation=90)  # Rotate x-axis labels to vertical
plt.yticks(rotation=0)   # Rotate y-axis labels to horizontal

plt.savefig('corr_heatmap_GXBoost.tiff', format='tiff', dpi=300)
plt.show()

plt.figure(figsize=(8,5))
ax = sns.kdeplot(x=y_test, y=y_pred, cmap="Blues", fill=True, cbar=True)
ax.set_xlabel('Actual PM2.5', fontsize=20)
ax.set_ylabel('Predicted PM2.5', fontsize=20)
ax.set_title('PM2.5 Density Heatmap from GXBoost', fontsize=20)
plt.xlim(-5, 25)  # Set x-axis limit
plt.ylim(-5, 25)  # Set y-axis limit
plt.savefig('density_heatmap_GXBoost.tiff', format='tiff', dpi=300)
plt.show()

plt.figure(figsize=(8,5))
plt.hist2d(y_test, y_pred, bins=(30, 30), cmap='Blues')
plt.colorbar(label='Frequency')
plt.xlabel('Actual PM2.5')
plt.ylabel('Predicted PM2.5')
plt.title('Actual vs Predicted PM2.5 2D Histogram from GXBoost', fontsize=20)
plt.savefig('2D_histogram_GXBoost.tiff', format='tiff', dpi=300)
plt.show()
