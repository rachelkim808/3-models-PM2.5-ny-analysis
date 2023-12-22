import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import Dense

# Load the data
data = pd.read_csv('BROOKLYNsummersmerged.csv')
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')
filtered_data = data[((data['winddir'] >=0) & (data['winddir'] <67.5)) | ((data['winddir'] >=257.5) & (data['winddir'] <360))]
summer_months = [6, 7, 8]
filtered_data_summer = filtered_data[filtered_data['Date'].dt.month.isin(summer_months)]

# Select Features and Target
features = ['temp', 'visibility', 'winddir', 'windspeed', 'precip', 'humidity', 'solarradiation', 'cloudcover']
target = 'Daily Mean PM2.5 Concentration'
X = filtered_data[features]
y = filtered_data[target]

# Split and standardize the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=30)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ANN model
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=X_train_scaled.shape[1]))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train_scaled, y_train, epochs=150, batch_size=10)

# Predict using the ANN model
y_pred = model.predict(X_test_scaled)

# Accuracy-like metric
tolerance = 5
correct_predictions = np.sum(np.abs(y_test - y_pred.ravel()) <= tolerance)
total_predictions = len(y_test)
accuracy_like_metric = correct_predictions / total_predictions
print(f'Accuracy-like metric (with tolerance ±{tolerance}): {accuracy_like_metric:.2f}')

# Permutation feature importance
def permutation_importance(model, X_valid, y_valid, metric, feature_indices):
    baseline_metric = metric(y_valid, model.predict(X_valid).ravel())
    importance_scores = []

    for idx in feature_indices:
        X_temp = X_valid.copy()
        shuffled_feature = shuffle(X_temp[:, idx])
        X_temp[:, idx] = shuffled_feature
        m = metric(y_valid, model.predict(X_temp).ravel())
        importance = baseline_metric - m
        importance_scores.append(importance)

    return importance_scores

# Calculate importance scores using indices of features
feature_indices = range(X_train_scaled.shape[1])  # Assuming X_train_scaled is a 2D array
importance_scores = permutation_importance(model, X_test_scaled, y_test, r2_score, feature_indices)

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

mapped_features = [feature_name_mapping.get(feature, feature) for feature in features]

# Save and Show Feature Importance Plot
plt.barh(mapped_features, importance_scores)
plt.xlabel('Importance', fontsize=20)
plt.ylabel('Feature', fontsize=20)
plt.title('Feature Importance from ANN', fontsize=20)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.savefig('feature_importance_ANN.tiff', format='tiff', dpi=300)
plt.show()
    
corr_matrix = data[features + [target]].corr()
plt.figure(figsize=(8,5))  # Adjust size to make sure text is not cluttered
sns.heatmap(corr_matrix.rename(columns=feature_name_mapping, index=feature_name_mapping),
            annot=True, cmap='coolwarm', fmt=".2f", annot_kws={'size': 12})  # Adjust annot_kws for bigger font size in annotations
plt.title('Correlation Heatmap from RF', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xticks(rotation=90)  # Rotate x-axis labels to vertical
plt.yticks(rotation=0)   # Rotate y-axis labels to horizontal

plt.savefig('corr_heatmap_RF.tiff', format='tiff', dpi=300)
plt.show()

plt.figure(figsize=(8,5))
ax = sns.kdeplot(x=y_test, y=y_pred.ravel(), cmap="Blues", fill=True, cbar=True)  # Flattening y_pred
ax.set_xlabel('Actual PM2.5', fontsize=20)
ax.set_ylabel('Predicted PM2.5', fontsize=20)
ax.set_title('PM2.5 Density Heatmap from ANN', fontsize=20)
plt.xlim(-5, 25)  # Set x-axis limit
plt.ylim(-5, 25)  # Set y-axis limit
plt.savefig('density_heatmap_ANN.tiff', format='tiff', dpi=300)
plt.show()


plt.figure(figsize=(8,5))
plt.hist2d(y_test, y_pred.ravel(), bins=(30, 30), cmap='Blues')  # Flattening y_pred with .ravel()
plt.colorbar(label='Frequency')
plt.xlabel('Actual PM2.5')
plt.ylabel('Predicted PM2.5')
plt.title('Actual vs Predicted PM2.5 2D Histogram from ANN', fontsize=20)
plt.savefig('2D_histogram_ANN.tiff', format='tiff', dpi=300)
plt.show()