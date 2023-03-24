import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12.0, 9.0)

# Preprocessing Input data
data = pd.read_csv('data.csv')
X = data.iloc[:, 0]
Y = data.iloc[:, 1]
plt.scatter(X, Y)
plt.show()

# Building the model
A = 0
B = 0
C = 0

L = 0.0001  # The learning Rate
epochs = 1000  # The number of iterations to perform gradient descent

n = float(len(X)) # Number of elements in X

# Performing Gradient Descent 
for i in range(epochs): 
    Y_pred = (-A*X - C) / B  # The current predicted value of Y
    D_A = (-2/n) * sum(X * (Y - Y_pred))  # Derivative wrt A
    D_B = (-2/n) * sum(Y - Y_pred)  # Derivative wrt B
    D_C = (-2/n) * sum(-Y_pred)  # Derivative wrt C
    A = A - L * D_A  # Update A
    B = B - L * D_B  # Update B
    C = C - L * D_C  # Update C
    
print (A, B, C)

# Making predictions
Y_pred = (-A*X - C) / B

plt.scatter(X, Y) 
plt.plot(X, Y_pred, color='red')  # regression line
plt.show()
