# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
#from numpy.core.umath_tests import inner1d

# Importing the dataset
dataset = pd.read_csv('train.csv',header = None)
X = dataset.iloc[:, 0:].values
dataset = pd.read_csv('trainLabels.csv',header = None)
y = dataset.iloc[:, 0:].values
z = []
for i in range(1000):
    if(y[i]==1):
        z.append(1)
    else:
        z.append(0)
y = z
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)

#Fitting the model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100,max_depth = 22)
classifier.fit(X,y)
#predicting
y_pred = classifier.predict(X_train)
from sklearn.metrics import accuracy_score
temp = accuracy_score(y_train, y_pred)

dataset = pd.read_csv('test.csv',header = None)
X = dataset.iloc[:, 0:].values
y_pred = classifier.predict(X)
ans = ["Solution"]
for i in range(len(y_pred)):
    ans.append(int(y_pred[i]))
y_pred = pd.DataFrame(ans)
y_pred.to_csv("abc.csv")
