""" 
import numpy as np


def sigmoid(x):
        return 1/1+np.exp(-x)

class LogisticRegression:

    def __init__(self,lr=0.01,n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        
    def fit(self,X,y):
        #We should initialize the weights and bias to zero
        n_samples,n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        #Repeat the same things for n number of times
        for _ in  range(self.n_iters):

            #Prediction of the results by sigmoid function
            # y = w.x + b
            linear_pred = np.dot(X,self.weights)+self.bias
            predictions = sigmoid(linear_pred)

            #Calculting the error
            dw = (1/n_samples)*np.dot(X.T,(predictions-y))
            db = (1/n_samples)*np.sum(predictions-y)

            #The below code is to calculate the gradient descent
            self.weights = self.weights - self.lr*dw
            self.bias = self.bias - self.bias*db

    #The code below is for Testing
    def predict(self,X):
         linear_pred = np.dot(X,self.weights)+self.bias
         y_pred = sigmoid(linear_pred)
         #To choose the class_label 
         class_label = [ 0 if y<=0.5 else 1 for y in y_pred]
         return class_label
 """

import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

class LogisticRegression:

    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        #We should initialize the weights and bias to zer
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        #Repeat the same things for n number of times
        for _ in range(self.n_iters):
            #Prediction of the results by sigmoid function
            # y = w.x + b
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(linear_pred)
            
            #Calculting the error
            dw = (1/n_samples) * np.dot(X.T, (predictions - y))
            db = (1/n_samples) * np.sum(predictions-y)

            #The below code is to calculate the gradient descent
            self.weights = self.weights - self.lr*dw
            self.bias = self.bias - self.lr*db

    
    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_pred)
        class_pred = [0 if y<=0.5 else 1 for y in y_pred]
        return class_pred