#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 17:15:46 2020

@author: Jake
"""

# https://www.ers.usda.gov/data-products/food-access-research-atlas/
# download-the-data/

# TODO X_train.T.sum() == 0
# get rid of tracts that have no impact

# Imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Load data
variables = pd.read_csv('food_desert_data_variables.csv')
df = pd.read_csv('food_desert_data.csv', na_values = 0)

# Features, remove tracts with no info
features = [i for i in df.columns if '1share' in i]
features.append('LILATracts_1And10')
df = df[features]

# Check nulls
print('Missing values: {}'.format(df.isnull().sum().sum()))

# Remove any tracts with no information
print('Tracts with no info: {}'.format(df.T.isnull().all().sum()))
df = df.dropna().reset_index(drop = True)

# TODO need to reassign X and y now that you changed features - do they still line up??

# Regressors: columns which represent 1 mile urban, 10 mile rural
X = pd.DataFrame(df[df.columns[:-1]])

# Response: "Low income and low access tract measured at 1 mile for urban
# areas and 10 miles for rural areas"
y = pd.DataFrame(df[df.columns[-1]])

# Split train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.2,
                                                    random_state = 42)

# Create model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Check coefficients
coeff = pd.DataFrame(lr.coef_[0], X.columns, columns = ['Coefficient'])
print('\n--- Coefficients ---')
print(coeff)