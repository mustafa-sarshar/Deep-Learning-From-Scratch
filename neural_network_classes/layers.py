import numpy as np

class Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons) # Random numbers normally distributed
        self.weights = self.weights / np.max(self.weights) # scaling all numbers between -1 and 1
        self.biases = np.zeros(shape=(1, n_neurons))
        self.y = None
    
    def forward(self, x):
        self.y = np.dot(x, self.weights) + self.biases






if __name__ == "__main__":
    dense_1 = Dense(n_inputs=2, n_neurons=5)
