import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression

data = pd.read_csv("diabetes.csv")
print(data)

#print(data.head())

#print(data.describe())

#data = pd.read_csv("diabetes.csv",header=None)
#print((data[[1,2,3,4,5]] == 0).sum())

# Mark zero values as missing or NaN
#data[[1,2,3,4,5]] = data[[1,2,3,4,5]].replace(0, np.NaN)
# Count the number of NaN values in each column
#print(data.isnull().sum())

# Fill missing values with mean column values
#data.fillna(data.mean(), inplace=True)
# Count the number of NaN values in each column
#print(data.isnull().sum())

values = data.values
X = values[:,0:8]
Y = values[:,8]

lr = LogisticRegression(penalty= 'l2' ,dual=False,max_iter=110)

lr.fit(X,Y)

print(lr.score(X,Y))

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score



kfold = KFold(n_splits=3, random_state=7)

result = cross_val_score(lr, X, Y, cv=kfold, scoring='accuracy')
print(result.mean())

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import GridSearchCV
dual=[True,False]
max_iter=[100,110,120,130,140]
param_grid = dict(dual=dual,max_iter=max_iter)

import warnings
warnings.filterwarnings('ignore')

import time

lr = LogisticRegression(penalty='l2')
grid = GridSearchCV(estimator=lr, param_grid=param_grid, cv = 3, n_jobs=-1)

start_time = time.time()
grid_result = grid.fit(X, Y)
# Summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
print("Execution time: " + str((time.time() - start_time)) + 'seconds')

dual=[True,False]
max_iter=[100,110,120,130,140]
C = [1.0,1.5,2.0,2.5]
param_grid = dict(dual=dual,max_iter=max_iter,C=C)
lr = LogisticRegression(penalty='l2')
grid = GridSearchCV(estimator=lr, param_grid=param_grid, cv = 3, n_jobs=-1)

start_time = time.time()
grid_result = grid.fit(X, Y)
# Summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
print("Execution time: " + str((time.time() - start_time)) + 'seconds')

import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import RandomizedSearchCV
random = RandomizedSearchCV(estimator=lr, param_distributions=param_grid, cv = 3, n_jobs=-1)

start_time = time.time()
random_result = random.fit(X, Y)
# Summarize results
print("Best: %f using %s" % (random_result.best_score_, random_result.best_params_))
print("Execution time: " + str((time.time() - start_time)) + 'seconds')









