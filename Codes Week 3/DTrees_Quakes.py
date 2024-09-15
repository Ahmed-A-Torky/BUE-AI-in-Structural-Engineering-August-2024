# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 14:48:06 2020
https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/
@author: Ahmed_A_Torky
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
# Import the DecisionTreeRegressor and evaluation metrics
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Assign colum names to the dataset
M_Range = ['4', '5', '6', '7', '8', '9']

# =============================================================================
# REGRESSION
# =============================================================================
# Read dataset to pandas dataframe
dataset = pd.read_csv('quakes_values.csv')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
# print(dataset.head(None))
dataset_values = dataset.values
# split into input (X) and output (y) variables
X = dataset[['NumberOfSites','PeakAcceleration(gal)','Intensity']].values
y = dataset_values[:,3] 

try:
    if y.shape[1]:  
        ny = y.shape[1]
except IndexError:
    ny = 1
    y = np.reshape(y, [y.shape[0],1])
try:
    if X.shape[1]:  
        nX = X.shape[1]
except IndexError:
    nX = 1
    X = np.reshape(X, [X.shape[0],1])

# Scale X
scaler_X  = StandardScaler() 
scaler_X.fit(X)
X_scaled = scaler_X.transform(X)
# Scale y
scaler_y  = StandardScaler() 
scaler_y.fit(y)
y_scaled = scaler_y.transform(y)
# split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.10) 

y_test=scaler_y.inverse_transform(y_test)

# Instantiate the model
regressor = DecisionTreeRegressor(random_state=0)

# Fit the model on the training data
regressor.fit(X_train, y_train)

# Make predictions on the test data
y_pred_scaled = regressor.predict(X_test)

# Since y was scaled, we need to inverse_transform the predictions to get them back to the original scale
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Optional: Compare actual vs predicted values
results = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
print(results)


# Optimum from Curve
Name = 'Decision Tree'
# compare y_test (scatter plot)
nx = 20
q_x = np.linspace(1, nx, nx)
fig = plt.figure(figsize=(10,7))
fsS=12
ax = fig.add_subplot(1, 1, 1)
ax.set_title(Name,fontsize=fsS)
ax.scatter(q_x,y_test[:nx], marker= 'o', s=70, facecolors='none', edgecolors='tab:blue', label = 'actual')
ax.scatter(q_x,y_pred[:nx], marker= 'x', s=70, color='tab:orange', label = 'predicted')
ax.grid()
plt.ylabel('M')
plt.xlabel('quake')
plt.legend(loc='upper left')
plt.show()

# =============================================================================
# REGRESSION ALL DATASET (no train no test)
# =============================================================================
# Read dataset to pandas dataframe
dataset = pd.read_csv('quakes_values.csv')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
# print(dataset.head(None))
dataset_values = dataset.values
# split into input (X) and output (y) variables
X = dataset[['NumberOfSites','PeakAcceleration(gal)','Intensity']].values
y = dataset_values[:,3] 

try:
    if y.shape[1]:  
        ny = y.shape[1]
except IndexError:
    ny = 1
    y = np.reshape(y, [y.shape[0],1])
try:
    if X.shape[1]:  
        nX = X.shape[1]
except IndexError:
    nX = 1
    X = np.reshape(X, [X.shape[0],1])

# Scale X
scaler_X  = StandardScaler() 
scaler_X.fit(X)
X_scaled = scaler_X.transform(X)
# Scale y
scaler_y  = StandardScaler() 
scaler_y.fit(y)
y_scaled = scaler_y.transform(y)
# split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.01) 

y_test=scaler_y.inverse_transform(y_test)

# Instantiate the model
regressor = DecisionTreeRegressor(random_state=0)

# Fit the model on the training data
regressor.fit(X_train, y_train)

# Make predictions on the test data
y_pred_scaled = regressor.predict(X_test)

# Since y was scaled, we need to inverse_transform the predictions to get them back to the original scale
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Optional: Compare actual vs predicted values
results = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
print(results)

# Optimum from Curve
Name = 'Decision Tree - Almost Zero Test'
# compare y_test (scatter plot)
nx = 20
q_x = np.linspace(1, nx, nx)
fig = plt.figure(figsize=(10,7))
fsS=12
ax = fig.add_subplot(1, 1, 1)
ax.set_title(Name,fontsize=fsS)
ax.scatter(q_x,y_test[:nx], marker= 'o', s=70, facecolors='none', edgecolors='tab:blue', label = 'actual')
ax.scatter(q_x,y_pred[:nx], marker= 'x', s=70, color='tab:orange', label = 'predicted')
ax.grid()
plt.ylabel('M')
plt.xlabel('quake')
plt.legend(loc='upper left')
plt.show()