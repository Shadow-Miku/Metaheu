import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Cargamos los datos de ejemplo
data = pd.read_csv('log.csv')

# Dividimos los datos en entrenamiento y pruebas
X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,:-1], data.iloc[:,-1], test_size=0.2)

# Entrenamos el modelo de regresión logística
reg = LogisticRegression().fit(X_train, y_train)

# Realizamos las predicciones en el conjunto de pruebas
y_pred = reg.predict(X_test)

# Graficamos los resultados
plt.scatter(X_test[:,0], y_test, color='black')
plt.plot(X_test[:,0], y_pred, color='blue', linewidth=3)
plt.xlabel("Variable predictora")
plt.ylabel("Variable objetivo (Clase)")
plt.title("Regresión logística")
plt.show()