import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LinealRegresion():
    
    def __init__(self)->None:
        pass
    
    def fit(self, X, Y, learning_rate = 0.0001, epochs=1000, bias=True):
        n = int(len(X))
        y = np.resize(Y,(n,1))
        
        if bias:
            m = X.shape[1] + 1
            aux = np.ones((n,1))
            X = np.concatenate((X, aux), axis=1)
        else:
            m = X.shape[1]
        thetas = np.zeros((m,1))
        
        errores = []
        iter_ = []
        #perfoming gradient
        #nos permite calcular el valor minimo
        for i in range(epochs):
            Y_pred = X.dot(thetas)  #The current predicted value of Y
            error = y - Y_pred
            error = X.T.dot(error)
            thetas = thetas - learning_rate * (-2/m) * error #update thetas
            iter_.append(i)
            errores.append(self.mean_error(y,Y_pred))
        print (thetas)
        return (iter_, errores)
        
    def mean_error(self, actual, predicted):
            
            n = len(actual)
            mse = 0
            
            for i in range(n):
                mse += (predicted[i] - actual[i]) **2
            mse /= n
            return mse
