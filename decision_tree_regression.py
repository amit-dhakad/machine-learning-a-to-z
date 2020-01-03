# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 20:31:35 2019

@author: Amit.Dhakad
"""

# Decision Tree Regression

# importing libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing dataset 

dataset = pd.read_csv('datasets/Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:,2].values

# Fitting decision Tree Regression to dataset

from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X,y)


y_pred = regressor.predict(np.array([6.5]).reshape(1,1))
#Visualising the Decision Tree regression results

X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Truth or bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()