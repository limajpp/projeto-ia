import numpy as np

class MLPRegressor:
    def __init__(self, hidden_layers=(10,), activation='relu', learning_rate=0.01, epochs=1000):
        self.hidden_layers = hidden_layers
        self.activation_name = activation.lower()
        self.lr = learning_rate
        self.epochs = epochs
        
        self.weights = []
        self.biases = []

    def _activate(self, Z, name):
        if name == 'relu':
            return np.maximum(0, Z)
        elif name == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(Z, -250, 250)))
        elif name == 'linear':
            return Z

    def _derivative(self, Z, A, name):
        if name == 'relu':
            return (Z > 0).astype(float)
        elif name == 'sigmoid':
            return A * (1 - A)
        elif name == 'linear':
            return np.ones_like(Z)

    def fit(self, X, y):
        n_samples = X.shape[0]
        n_features = X.shape[1]
        
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
            
        layer_sizes = [n_features] + list(self.hidden_layers) + [1]
        
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            limit = np.sqrt(2 / layer_sizes[i]) if self.activation_name == 'relu' else np.sqrt(1 / layer_sizes[i])
            W = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * limit
            b = np.zeros((1, layer_sizes[i+1]))
            
            self.weights.append(W)
            self.biases.append(b)

        for _ in range(self.epochs):
            A_list = [X]  
            Z_list = []   
            
            for i in range(len(self.weights) - 1):
                Z = np.dot(A_list[-1], self.weights[i]) + self.biases[i]
                A = self._activate(Z, self.activation_name)
                Z_list.append(Z)
                A_list.append(A)
                
            Z_out = np.dot(A_list[-1], self.weights[-1]) + self.biases[-1]
            A_out = self._activate(Z_out, 'linear')
            Z_list.append(Z_out)
            A_list.append(A_out)
            
            dZ = (2 / n_samples) * (A_out - y)
            
            dW_list = []
            db_list = []
            
            dW = np.dot(A_list[-2].T, dZ)
            db = np.sum(dZ, axis=0, keepdims=True)
            dW_list.append(dW)
            db_list.append(db)
            
            for i in range(len(self.weights) - 2, -1, -1):
                dA = np.dot(dZ, self.weights[i+1].T)
                dZ = dA * self._derivative(Z_list[i], A_list[i+1], self.activation_name)
                
                dW = np.dot(A_list[i].T, dZ)
                db = np.sum(dZ, axis=0, keepdims=True)
                
                dW_list.insert(0, dW)
                db_list.insert(0, db)
                
            for i in range(len(self.weights)):
                self.weights[i] -= self.lr * dW_list[i]
                self.biases[i] -= self.lr * db_list[i]

    def predict(self, X):
        A = X
        for i in range(len(self.weights) - 1):
            Z = np.dot(A, self.weights[i]) + self.biases[i]
            A = self._activate(Z, self.activation_name)
            
        Z_out = np.dot(A, self.weights[-1]) + self.biases[-1]
        A_out = self._activate(Z_out, 'linear')
        
        return A_out.flatten()
    
