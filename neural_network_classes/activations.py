import numpy as np

class ReLU:
    def forward(self, x):
        self.y = np.maximum(0, x)

class Softmax:
    def forward(self, x):
        self.y = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.y = self.y / np.sum(self.y, axis=1, keepdims=True)