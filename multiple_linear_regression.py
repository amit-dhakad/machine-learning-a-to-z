# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 11:19:03 2019

@author: Amit.Dhakad
"""

# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing dataset

dataset = pd.read_csv('datasets/50_Startups.csv')
X = dataset.iloc[:, :-1].values  # get all independent variable except dependent varible
y = dataset.iloc[:,4].values # get dependent variable

# encoding categorical data

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)

# Avoiding dummy variable trap 
X = X[:, 1:]

# Splitting data in to Training set and Test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=0)

# Fitting Multiple Linear Regression to Training set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
 
# predicting the test result

y_pred = regressor.predict(X_test)

# Building a optimal model using backward elimination

# step one : select significance level to stay in model (e.g = SL = 0.05)
# step two : Fit the model with all possible predictors
# step three: consider your predictor with highest p- value. if P > SL, go to step 4. otherwie your model is ready.
# step four: Remove the predictor
# step five: Fit model without this model

import statsmodels.api as sm

X = np.append(arr = np.ones((50,1)).astype(int), values =X , axis=1)  # add columns of 1 in X. x0 = 1

# OLS: Ordinary Least Squares  
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]


def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
 
X_Modeled = backwardElimination(X_opt, SL)


#X_opt = X[:, [0,1,2,3,4,5]]
#regressor_OLS = sm.OLS(endog= y, exog=X_opt).fit() # step two
#regressor_OLS.summary()
#
#X_opt = X[:, [0,1,3,4,5]]
#regressor_OLS = sm.OLS(endog= y, exog=X_opt).fit() # step two
#regressor_OLS.summary()
#
#X_opt = X[:, [0,3,4,5]]
#regressor_OLS = sm.OLS(endog= y, exog=X_opt).fit() # step two
#regressor_OLS.summary()
#
#X_opt = X[:, [0,3,5]]
#regressor_OLS = sm.OLS(endog= y, exog=X_opt).fit() # step two
#regressor_OLS.summary()
#
#X_opt = X[:, [0,3]]
#regressor_OLS = sm.OLS(endog= y, exog=X_opt).fit() # step two
#regressor_OLS.summary()