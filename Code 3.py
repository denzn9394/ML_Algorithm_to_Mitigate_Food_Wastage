#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error


# In[3]:


# Initialize models
models = {
    'GradientBoosting': GradientBoostingRegressor(),
    'RandomForest': RandomForestRegressor(),
    'NeuralNetwork': MLPRegressor(),
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'KNN': KNeighborsRegressor(),
    'SVM': SVR(),
    'DecisionTree': DecisionTreeRegressor(),
    'AdaBoost': AdaBoostRegressor()
}


# In[4]:


# Evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)  # Train the model
    y_pred = model.predict(X_test)  # Make predictions
    mse = mean_squared_error(y_test, y_pred)  # Calculate MSE
    print(f'{name} Mean Squared Error: {mse}')


# In[5]:


#Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

# Defining parameter grid (as an example)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Initialize RandomForestRegressor
rf = RandomForestRegressor()

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                           cv=5, n_jobs=-1, verbose=2)

# Fit the model
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print(f"Best Params for Random Forest: {best_params}")


# In[6]:


#Feature Engineering
from sklearn.preprocessing import PolynomialFeatures

# Create polynomial features (degree=2 as an example)
poly = PolynomialFeatures(2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Then train the model using these polynomial features


# In[7]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

# Generate some example data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create polynomial features
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Initialize and train Random Forest model with best parameters
# Use best_params from the previous GridSearch result
best_params = {'max_depth': None, 'n_estimators': 100, 'min_samples_split': 2}

rf_poly = RandomForestRegressor(**best_params)
rf_poly.fit(X_train_poly, y_train)

# Make predictions
y_pred_train = rf_poly.predict(X_train_poly)
y_pred_test = rf_poly.predict(X_test_poly)

# Evaluate the model
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)

print(f"Training MSE with Polynomial Features: {mse_train}")
print(f"Test MSE with Polynomial Features: {mse_test}")

