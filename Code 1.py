#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Assuming your data is stored as 'food_waste.csv'
data = pd.read_csv('food_waste.csv')
data.head()


# In[2]:


# Replace missing values with column means
df = pd.DataFrame(data)
df.fillna(df.mean(), inplace=True)


# In[3]:


# Step 5: Data Sampling
# The random_state parameter ensures reproducibility of your results
# frac=0.2 means that you will randomly select 20% of the data for the sample
sample_df = df.sample(frac=0.2, random_state=1)

# Display the sampled data
print("Sampled DataFrame:")
print(sample_df)

# To save the sampled data as a new CSV file
# sample_df.to_csv('sampled_data.csv', index=False)

# If you wish to analyze the sampled data for any statistical properties or summary statistics, you can use describe()
print("\nSampled DataFrame Statistics:")
print(sample_df.describe())

# If you want to check if the sampled data is a good representation of the original data, you could compare means or other statistics
print("\nComparison of Mean values between Original and Sampled DataFrame:")
print("Original DataFrame:")
print(df.mean())
print("Sampled DataFrame:")
print(sample_df.mean())


# In[4]:


df.to_csv('dataframe.csv', index=False)

