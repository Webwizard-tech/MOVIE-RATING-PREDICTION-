import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Download the latest version of the dataset
path = kagglehub.dataset_download("adrianmcmahon/imdb-india-movies")
print("Path to dataset files:", path)

# Load the data from the downloaded CSV file
data = pd.read_csv(f'{path}/IMDb Movies India.csv')  # Adjust filename if necessary

# Data preprocessing
data.dropna(inplace=True)  # Handle missing values
data['Movie_Age'] = 2023 - data['Year']  # Convert Year to Movie Age
data.drop(['Year', 'Name'], axis=1, inplace=True)  # Drop unnecessary columns

# Feature encoding
data = pd.get_dummies(data, columns=['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3'], drop_first=True)

# Define features and target
X = data.drop('Rating', axis=1)  # Features
y = data['Rating']  # Target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')

# Feature importance visualization
feature_importances = model.feature_importances_
features = X.columns

plt.barh(features, feature_importances)
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.show()
