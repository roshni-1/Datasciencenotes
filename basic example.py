# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Creating a sample dataset (heights and weights)
data = {'Height': [150, 160, 165, 170, 175, 180, 185],
        'Weight': [50, 55, 62, 68, 72, 80, 85]}

# Creating a Pandas DataFrame from the data
df = pd.DataFrame(data)

# Visualizing the data
plt.figure(figsize=(8, 6))
plt.scatter(df['Height'], df['Weight'], color='blue')
plt.title('Height vs Weight')
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.grid(True)
plt.show()

# Splitting the data into features (X) and target variable (y)
X = df[['Height']]
y = df['Weight']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating a linear regression model
model = LinearRegression()

# Training the model
model.fit(X_train, y_train)

# Making predictions on the test set
y_pred = model.predict(X_test)

# Visualizing the linear regression line
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.title('Height vs Weight (Linear Regression)')
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.grid(True)
plt.show()

# Evaluating the model using mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')
