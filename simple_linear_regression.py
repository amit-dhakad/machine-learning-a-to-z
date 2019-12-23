# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 15:06:31 2019

@author: Amit.Dhakad
"""

# simple linear Regresion

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Importing dataset 
dataset = pd.read_csv('datasets/Salary_Data.csv')
X= dataset.iloc[:, :-1].values
y= dataset.iloc[:, 1].values

# Splitting the data set into the training set and test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 1/3, random_state = 0)

# Fitting simple linear Regression model to training set

from sklearn.linear_model import LinearRegression  # import linear regression model

regressor = LinearRegression() # create linearRegression object
regressor.fit(X_train, y_train) # fit x train and y train to regression 

# Predicting the test set results

y_pred = regressor.predict(X_test)

# Visualising the training set results
plt.scatter(X_train, y_train, color="red")  # scatter plot for x train and y train
plt.plot(X_train, regressor.predict(X_train), color="blue") # plot line x train and predict result
plt.xlabel('Years of Experience') # x label
plt.title('Salary vs Experience (Training set)') # title 
plt.ylabel('Salary') # y label
plt.show() # display the plot

# Visualising the test set results
plt.scatter(X_test, y_test, color="red")
plt.plot(X_train, regressor.predict(X_train), color="blue")
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

