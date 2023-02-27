# Making the imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12.0, 9.0)

# Preprocessing Input data
data = pd.read_csv('data12.csv')
X = data.iloc[:, :-1] # Select all columns except the last one
Y = data.iloc[:, -1] # Select only the last column as labels
n_features = X.shape[1] # Number of features
plt.scatter(X.iloc[:,0], Y)
plt.show()

# Building the model
m = np.zeros(n_features) # Initialize coefficients to 0

L = 0.0001  # The learning Rate
epochs = 1000  # The number of iterations to perform gradient descent

n = float(len(X)) # Number of elements in X

# Performing Gradient Descent 
for i in range(epochs): 
    Y_pred = np.dot(X, m) # The current predicted value of Y
    D_m = (-2/n) * np.dot(X.T, Y - Y_pred) # Derivative wrt m
    m = m - L * D_m  # Update m
    
print (m)


# Making predictions
Y_pred = np.dot(X, m)

plt.scatter(X.iloc[:,0], Y) 
plt.plot(X.iloc[:,0], Y_pred, color='red')  # regression line
plt.show()
