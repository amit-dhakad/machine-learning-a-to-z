# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 20:34:07 2019

@author: Amit.Dhakad
"""

# Polynomial regression 

# importing libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# impoting dataset 

dataset = pd.read_csv('datasets/Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y= dataset.iloc[:,2].values


# daaset is not splitting in Training and test set because data is less

# Linear Regression create to compare with polynomial Regression

from sklearn.linear_model import LinearRegression

linear_regressor = LinearRegression()
linear_regressor.fit(X,y)


# Polynomial Regression

from sklearn.preprocessing import PolynomialFeatures

polynomial_regressor = PolynomialFeatures(degree= 4)
X_poly = polynomial_regressor.fit_transform(X)

linear_regressor2 = LinearRegression()
linear_regressor2.fit(X_poly, y)


# Visualising Linear Regression results
plt.scatter(X,y, color="red")
plt.plot(X, linear_regressor.predict(X), color="blue")
plt.title('True or  Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')


# Visualising Polynomial Regression resuts

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid),1))

plt.scatter(X,y, color="red")
plt.plot(X_grid, linear_regressor2.predict(polynomial_regressor.fit_transform(X_grid)), color="blue")
plt.title('True or  Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')

# Predicting new result with Linear Regression 
linear_regressor.predict(np.array([6.5]).reshape(1, 1))
# Predicting new result with Polynomial Regression
linear_regressor2.predict(polynomial_regressor.fit_transform(np.array([6.5]).reshape(1, 1)))