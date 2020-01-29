#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 17:15:46 2020

@author: Jake
"""

# https://www.ers.usda.gov/data-products/food-access-research-atlas/
# download-the-data/

# Imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sns
from matplotlib import pyplot as plt

# Load data
variables = pd.read_csv('food_desert_data_variables.csv')
df = pd.read_csv('food_desert_data.csv')

# Features, remove tracts with no info
features = [i for i in df.columns if '1share' in i]
features.append('LILATracts_1And10')
df = df[features]

# Check nulls
print('Missing values: {}'.format(df.isnull().sum().sum()))

# Remove any tracts with no information
index_to_drop = df[(df[df.columns] == 0).T.all()].index
print('Tracts with no info: {}'.format(len(index_to_drop)))
df = df.drop(index_to_drop).reset_index(drop = True)

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
logr = LogisticRegression(max_iter = 500)
logr.fit(X_train, y_train.to_numpy().ravel())

# Check coefficients
coeff = pd.DataFrame(logr.coef_[0], X.columns, columns = ['Coefficient'])
print('\n--- Coefficients ---')
print(coeff)
print('\n--- Intercept ---')
print(logr.intercept_[0])

# Heatmap
sns.set(font_scale = 1.4)
corrmat = df.corr()
f_corr, ax_corr = plt.subplots(figsize = (15, 15))
sns.heatmap(corrmat, ax = ax_corr, annot = True,
            cmap = 'YlOrRd',
            linewidths = 2.0,
            square = True)
ax_corr.set_title('Correlation Between Features',
                  fontsize = 25,
                  fontweight = 'bold',
                  pad = 45)
plt.show()

# Highest correlations with food desert
correlations = [i for i
                in corrmat[corrmat['LILATracts_1And10'] > .25].index[:-1]]
for i in correlations:
    print()
    print(variables[variables['Field'] == i]['Description'].values[0] + ':\n' +
          str(corrmat[(i)]['LILATracts_1And10']))