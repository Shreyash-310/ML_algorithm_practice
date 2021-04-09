# Step 1 - Load Data
import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer

dataset = pd.read_csv("50_Startups.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Step 2 - Encode Categorical Data

labelEncoder_X = LabelEncoder()
X[:, 3] = labelEncoder_X.fit_transform(X[:, 3])
s = X[:, 3]
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [3])],   # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough')                                        # Leave the rest of the columns untouched

X = ct.fit_transform(X)

# Step 3 - Dummy Trap
X = X[:, 1:]

# Step 4 - Split Data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Step 5 - Fit Regressor

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Step 6 - Predict
y_pred = regressor.predict(X_test)

# Add ones

ones = np.ones(shape=(50, 1), dtype=int)
X = np.append(arr=ones, values=X, axis=1)

# Backward Elimination

X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
