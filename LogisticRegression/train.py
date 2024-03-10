import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from Logistic import LogisticRegression


bc = datasets.load_breast_cancer()
X,Y = bc.data,bc.target

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=1234)

clf = LogisticRegression(lr=0.01)
clf.fit(X_train,Y_train)
y_pred = clf.predict(X_test)

def accuracy(y_pred , y_test):
    return np.sum(y_pred==y_test)/len(y_test)

acc = accuracy(y_pred,Y_test)
print(acc)
print('---------------------------')