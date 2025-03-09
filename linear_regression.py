import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
path = "data/data.csv"
try: 
    df = pd.read_csv(path)
except FileNotFoundError:
    print(f"Error: Unable to find file path: {path}")
    exit(1)
    
# Remove missing or impute values
print("original shape: ", df.shape)
print("number of missing values: \n", df.isnull().sum())
df = df.dropna(how = 'any')
print("shape after removing missing values: ", df.shape)

# Preview data
print("\nPreview of data: ")
print(df.head())

# Feature matrix and target vector
X = df.loc[:, ['x']].values
if X.ndim != 2:
    raise ValueError(f"Feature matrix X must be a 2D array, found shape: {X.shape}")
y = df.loc[:, 'y'].values

print("Feature matrix X shape: ", X.shape)
print("Target vector y shape: ", y.shape)

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model and fine tune hyper parameters
model = LinearRegression(fit_intercept=True)
model.fit(X_train, y_train)

# predictions 
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# performance metrics
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
mse = mean_squared_error(y_test, test_predictions)
r2 = r2_score(y_test, test_predictions)

# print performance metrics
print("\nPerformance Metrics:")
print("Train Score: ", train_score)
print("Test Score: ", test_score)
print("MSE: ", mse)
print("R2: ", r2)

# equation of the line
m = model.coef_[0]
b = model.intercept_
print("Equation of the line: y = {:0.2f}x + {:0.2f}".format(m, b))