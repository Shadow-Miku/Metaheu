import numpy as np

class LinealRegresion():

    def __init__(self) -> None:
        self.thetas = np.asarray([])

    def _target_function(self, x, theta_1): #funcion definida
        return theta_1*x

    def fit(self, X, target, learning_rate=0.1, max_iters=1000):#aprendizaje del algoritmo X es una matriz de n*m
        self.thetas = np.random.rand(1) #self.thetas = np.random.rand(X.shape[1])
        aux = int(X.shape[0]*0.8)
        X_training = X[:aux]
        X_validate= X[:-aux]
        y_training = target[:aux]
        y_validate = target[:-aux]
        print(" Initial Tetha = "+ str(self.thetas[0]))
        for i in range(max_iters):
            derivate_0, derivate_1 = 0
            for i,x in enumerate(X_training):
                derivate_0 = derivate_0 + (self._target_function(x,self.thetas[0])) - y_training[i]
                derivate_1 = derivate_1 + (self._target_function(x,self.thetas[1])) - y_training[i]*x
            derivate_0 = (1/(2*X.shape[0])) * (derivate_0)  #derivate_0 = (1/X.shape[0]) * (derivate_0**2)
            temp_0 = self.thetas[0] - (learning_rate*derivate_0)
            temp_1 = self.thetas[1] - (learning_rate*derivate_1)
            self.thetas[0] = temp_0
            self.thetas[1] = temp_1
            print("New Theta ="+str(self.thetas[0]))

    ## TO TEST
        result = []
        for x in X_validate:
            result.append(self._target_function(x,self.thetas[0])) 
        return (result, X_validate, y_validate)

    def _mean_error(self):
        pass

    def predict(self):
        pass

