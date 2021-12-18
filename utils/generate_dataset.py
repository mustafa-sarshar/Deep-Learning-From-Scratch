"""
These code are inspired by:
    - https://cs231n.github.io/neural-networks-case-study/
"""
import numpy as np

def generate_spiral_data(n_points_per_class=100, n_classes=2, visualization=False):
    N_DIMENSIONS = 2
    X = np.zeros((n_points_per_class*n_classes, N_DIMENSIONS)) # data matrix (each row = single example)
    y = np.zeros(n_points_per_class*n_classes, dtype="uint8") # class labels
    for class_index in range(n_classes):
        ix = range(n_points_per_class*class_index, n_points_per_class*(class_index+1))
        r = np.linspace(0.0, 1, n_points_per_class) # radius
        t = np.linspace(class_index*4, (class_index+1)*4, n_points_per_class) + np.random.randn(n_points_per_class)*0.2 # theta
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = class_index

    if visualization:
        import matplotlib.pyplot as plt
        plt.clf()
        plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
        plt.show()
    return X, y

def generate_vertical_data(n_points_per_class=100, n_classes=2, visualization=False):
    N_DIMENSIONS = 2
    X = np.zeros((n_points_per_class*n_classes, N_DIMENSIONS)) # data matrix (each row = single example)
    y = np.zeros(n_points_per_class*n_classes, dtype="uint8") # class labels
    for class_index in range(n_classes):
        ix = range(n_points_per_class*class_index, n_points_per_class*(class_index+1))
        X[ix] = np.c_[np.random.randn(n_points_per_class)*.1 + (class_index)/3, np.random.randn(n_points_per_class)*.1 + 0.5]
        y[ix] = class_index

    if visualization:
        import matplotlib.pyplot as plt
        plt.clf()
        plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
        plt.show()
    return X, y




if __name__ == "__main__":
    X, y = generate_spiral_data(n_points_per_class=100, n_classes=2, visualization=False)