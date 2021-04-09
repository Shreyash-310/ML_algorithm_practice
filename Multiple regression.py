# Multiple Linear Regression
# Importing the libraries

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, 4]

# Convert the column into categorical columns

states = pd.get_dummies(X['State'], drop_first=True)

# Drop the state coulmn
X = X.drop('State', axis=1)

# concat the dummy variables
X = pd.concat([X, states], axis=1)

# Splitting the dataset into the Training set and Test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fitting Multiple Linear Regression to the Training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
plt.plot(X_test, y_pred)
plt.show()
print(y_pred)

score = r2_score(y_test, y_pred)
print(score)
