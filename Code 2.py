#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Importing libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv('food_waste.csv')

# Replace '-' with np.nan and convert specific columns to float
df.replace('-', np.nan, inplace=True)
cols_to_convert = ['Food consumption', 'Primary production waste', 'Processing and manufacturing waste', 'Distribution waste', 'Consumption waste', 'Total waste']
for col in cols_to_convert:
    df[col] = df[col].astype(float)

# Fill NaNs
df.fillna(0, inplace=True)

# Feature selection
X = df[cols_to_convert[:-1]]
y = df['Total waste']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Cross-validation
model_rf = RandomForestRegressor(random_state=1)
cv_score_rf = cross_val_score(model_rf, X_train, y_train, cv=5)
print(f"Random Forest CV Scores: {cv_score_rf}")

model_svr = SVR()
cv_score_svr = cross_val_score(model_svr, X_train, y_train, cv=5)
print(f"SVM CV Scores: {cv_score_svr}")

# Hyperparameter tuning with GridSearchCV
param_grid_rf = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20, 30],
}

grid_search_rf = GridSearchCV(model_rf, param_grid_rf, cv=5)
grid_search_rf.fit(X_train, y_train)
best_params_rf = grid_search_rf.best_params_
print(f"Best Params for Random Forest: {best_params_rf}")

param_grid_svr = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
}

grid_search_svr = GridSearchCV(model_svr, param_grid_svr, cv=5)
grid_search_svr.fit(X_train, y_train)
best_params_svr = grid_search_svr.best_params_
print(f"Best Params for SVM: {best_params_svr}")

# Retrain with best parameters and evaluate
best_rf = RandomForestRegressor(**best_params_rf, random_state=1)
best_rf.fit(X_train, y_train)
y_pred_rf = best_rf.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
print(f"Random Forest Mean Squared Error: {mse_rf}")

best_svr = SVR(**best_params_svr)
best_svr.fit(X_train, y_train)
y_pred_svr = best_svr.predict(X_test)
mse_svr = mean_squared_error(y_test, y_pred_svr)
print(f"SVM Mean Squared Error: {mse_svr}")

