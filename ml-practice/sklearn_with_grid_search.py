# -*- coding: utf-8 -*-
"""SKLearn with Grid Search

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1RBVjd3jHc6476Y0Cq17K8NsM2t10b6sA
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
 
data = pd.read_csv('Salary_Data.csv')

import numpy as np
 
X = data['YearsExperience']
y = data['Salary']
X = X[:,np.newaxis]

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
model = SVR()
parameters = {
    'kernel': ['rbf'],
    'C':     [1000, 10000, 100000],
    'gamma': [0.5, 0.05,0.005]
}
grid_search = GridSearchCV(model, parameters)
grid_search.fit(X,y)

print(grid_search.best_params_)

model_baru  = SVR(C=100000, gamma=0.005, kernel='rbf')
model_baru.fit(X,y)

import matplotlib.pyplot as plt
plt.scatter(X, y)
plt.plot(X, model_baru.predict(X))
