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
import seaborn as sns
from matplotlib import pyplot as plt

def title_print(text):
    '''
    Used throughout to print section titles
    '''
    text_len = len(text)
    print()
    print('#' * (text_len + 4))
    print('#', text, '#')
    print('#' * (text_len + 4))

# Load data
variables = pd.read_csv('food_desert_data_variables.csv')
df = pd.read_csv('food_desert_data.csv')

# Features, remove tracts with no info
features = [i for i in df.columns if '1share' in i]
features.extend(['LA1and10', 'LILATracts_1And10'])
df = df[features]

# Check nulls
print('Missing values: {}'.format(df.isnull().sum().sum()))

# Remove any tracts with no information
index_to_drop = df[(df[df.columns] == 0).T.all()].index
print('Tracts with no info: {}'.format(len(index_to_drop)))
df = df.drop(index_to_drop).reset_index(drop = True)

# Heatmap
sns.set(font_scale = 1.4)
corrmat = df[df.columns[:-2]].corr()
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

# Regressors: columns which represent 1 mile urban, 10 mile rural
X = pd.DataFrame(df[df.columns[:-2]])

# Responses:
# 1. "Low access tract at 1 mile for urban areas and 10 miles for rural areas"
# 2. "Low income and low access tract measured at 1 mile for urban
#     areas and 10 miles for rural areas"
la_y = pd.DataFrame(df[df.columns[-2]])
lila_y = pd.DataFrame(df[df.columns[-1]])

def logistic_reg(X, y):
    '''
    Performs logistic regression on X, y
    '''
    title_print(y.columns[0])
    # Split train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size = 0.2,
                                                        random_state = 42)
    
    # Create model
    logr = LogisticRegression(max_iter = 500)
    logr.fit(X_train, y_train.to_numpy().ravel())
    
    # Check score for sanity
    print('Logistic regression score: {}%'.
          format(100 * round(logr.score(X_test, y_test), 4)))
    
    # Check coefficients
    coeff = pd.DataFrame(logr.coef_[0], X.columns, columns = ['Coefficient'])
    title_print('Coefficients')
    print(coeff)
    title_print('Intercept')
    print(logr.intercept_[0])
    
    # Top 3 high impact regressor variables
    title_print('High regressors')
    high_reg = abs(coeff).nlargest(n = 3, columns = 'Coefficient').index
    for i in high_reg:
        print()
        print(variables[variables['Field'] == i]['Description'].values[0] +
              ':\n' + str(round(coeff.loc[i][0], 4)))

logistic_reg(X, la_y)
logistic_reg(X, lila_y)