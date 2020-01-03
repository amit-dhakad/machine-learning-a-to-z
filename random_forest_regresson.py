# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 21:04:41 2020

@author: Amit.Dhakad
"""

# Random forest regression

# importing libraries 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing dataset

dataset = pd.read_csv('datasets/Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

# fitting random forest regression to dataset

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(X,y)

y_pred = regressor.predict(np.array([6.5]).reshape(1,1))
# Visualising the regressor results (for higher resolution and smooth curve)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X, y, color="red")
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Truth or Bluff (Random Forest Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()