# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset from scikit-learn
from sklearn.datasets import load_iris
iris = load_iris()

# Creating a DataFrame from the dataset
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['target'] = iris.target

# Splitting the data into features (X) and target variable (y)
X = data.drop('target', axis=1)
y = data['target']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating a Random Forest Classifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Training the model
model.fit(X_train, y_train)

# Making predictions on the test set
predictions = model.predict(X_test)

# Calculating accuracy
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy:.2f}')

# Example: Predicting flower species for a new data point
new_data_point = [[5.1, 3.5, 1.4, 0.2]]  # Sepal length, Sepal width, Petal length, Petal width
predicted_species = iris.target_names[model.predict(new_data_point)[0]]
print(f'Predicted Flower Species: {predicted_species}')
