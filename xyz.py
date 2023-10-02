# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Reading data from a CSV file (replace 'data.csv' with your file)
data = pd.read_csv('data.csv')

# Exploratory Data Analysis (EDA)
print(data.head())  # Display the first few rows of the dataset
print(data.describe())  # Summary statistics of numerical columns
print(data.info())  # Information about the dataset

# Data Preprocessing
# Handling missing values
data.dropna(inplace=True)  # Drop rows with missing values

# Feature selection
features = data[['feature1', 'feature2', 'feature3']]
target = data['target']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Machine Learning Model (Linear Regression in this example)
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions
predictions = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, predictions)
print('Mean Squared Error:', mse)

# Data Visualization
plt.scatter(y_test, predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True vs. Predicted Values')
plt.show()
