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

def logistic_reg(X, y, test_size, max_iters, n_top):
    '''
    Performs logistic regression on X, y
        X: regressor (independent) variables
        y: response variable
        test_size (float): 0-1, ratio of test set
        max_iters (int): max iterations to run until convergence
        n_top (int): number of top coefficients to print out after running regression
    '''
    title_print(y.columns[0])
    
    # Split train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size = test_size,
                                                        random_state = 42)
    
    # Create model
    logr = LogisticRegression(max_iter = max_iters)
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
    
    # Top n high impact regressor variables
    title_print('Top regressors')
    high_reg = abs(coeff).nlargest(n = n_top, columns = 'Coefficient').index
    for i in high_reg:
        print(variables[variables['Field'] == i]['Description'].values[0] +
              ':\n' + str(round(coeff.loc[i][0], 4)))
        print()

logistic_reg(X, la_y, test_size = 0.2, max_iters = 500, n_top = 5)
logistic_reg(X, lila_y, test_size = 0.2, max_iters = 500, n_top = 5)

# Run logistic regression with only races/nationalities
nat_features = ['lawhite1share', 'lablack1share', 'laasian1share',
                'lanhopi1share', 'laaian1share', 'laomultir1share',
                'lahisp1share', 'LILATracts_1And10']
nat_df = df[nat_features]
nat_X = pd.DataFrame(nat_df[nat_df.columns[:-1]])
nat_y = pd.DataFrame(nat_df[nat_df.columns[-1]])
logistic_reg(nat_X, nat_y, test_size = 0.2, max_iters = 500,
             n_top = len(nat_features) - 1)