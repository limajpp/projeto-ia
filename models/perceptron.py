import numpy as np

class BinaryPerceptron:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_features = X.shape[1]
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.epochs):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                
                y_predicted = 1 if linear_output >= 0 else 0
                
                update = self.lr * (y[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def net_input(self, X):
        return np.dot(X, self.weights) + self.bias

    def predict(self, X):
        linear_output = self.net_input(X)
        return np.where(linear_output >= 0, 1, 0)


class OvAPerceptron:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.models = {}
        self.classes = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        
        for c in self.classes:
            y_binary = np.where(y == c, 1, 0)
            
            model = BinaryPerceptron(learning_rate=self.lr, epochs=self.epochs)
            model.fit(X, y_binary)
            
            self.models[c] = model

    def predict(self, X):
        confidences = np.zeros((X.shape[0], len(self.classes)))
        
        for idx, c in enumerate(self.classes):
            confidences[:, idx] = self.models[c].net_input(X)
            
        best_class_indices = np.argmax(confidences, axis=1)
        
        return self.classes[best_class_indices]
    
