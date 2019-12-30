# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 21:38:13 2019

@author: Amit.Dhakad
"""

# Support Vector Regression

# 1. Support Vector machines support linear and nonlinear regression that we can refer to as SVR
# 2. Instead of trying to fit largest possible street between two classes while limiting margin violations,
# SVR tries to fit as many instances as possible on the street while limiting margin violations.
# 3. The width of the strret is controlled by a hyper parameter epsilon.
# 4. SVR perform  linear regression, in a higher (Dimensinal space).
# 5. We can think SVR as if each data point in the training represents it's own dimension. 
#    When you evalute your kernal between a test point and a point in the training set the resulting value gives you 
#    the coordinate of your test point  in the dimension.
# 6. The vector we get when wvalute the test point for all points in the training set, k(vector) is representation of the test point
# the higher dimensional space.
# 7. Once  you have that vector you the use it to perform a linear regression.


# importing libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing dataset

dataset = pd.read_csv('datasets/Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:,2].values

# Feature scaling

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()

X = sc_X.fit_transform(X)
y = sc_y.fit_transform(np.array(y).reshape(10,1))

# Fitting SVR to dataset

from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X,y)

y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([6.5]).reshape(1,1)))) 


# Visualising the SVR result 

plt.scatter(X,y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show() 


# Visualising SVR resuts (for higher resolution and smoother curve)

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid),1))

plt.scatter(X,y, color="red")
plt.plot(X_grid, regressor.predict(X_grid), color="blue")
plt.title('True or  Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')