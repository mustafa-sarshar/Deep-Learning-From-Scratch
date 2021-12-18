# In[] Libs
import numpy as np
import matplotlib.pyplot as plt
from neural_network_classes.layers import Dense
from neural_network_classes.activations import ReLU, Softmax
from neural_network_classes.error_estimation import CategoricalCrossentropy, Accuracy
from utils.generate_dataset import generate_spiral_data

# In[] Run the App
if __name__ == "__main__":
    X, y = generate_spiral_data(n_points_per_class=200, n_classes=5, visualization=False)

    layer_1_dense = Dense(n_inputs=X.ndim, n_neurons=20)
    layer_1_dense.forward(X)
    print("*" * 20)
    print(f"Layer_1 y:\n{layer_1_dense.y}")

    activation_ReLU = ReLU()
    activation_ReLU.forward(layer_1_dense.y)
    print("*" * 20)
    print(f"Layer_1 y after activation:\n{activation_ReLU.y}")
    
    plt.clf()
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.show()

    plt.scatter(activation_ReLU.y[:, 0], activation_ReLU.y[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.show()

    activation_Softmax = Softmax()
    activation_Softmax.forward(layer_1_dense.y)
    print("*" * 20)
    print(f"Layer_2 y after activation:\n{activation_Softmax.y}")

    plt.scatter(activation_Softmax.y[:, 0], activation_Softmax.y[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.show()

    loss_func = CategoricalCrossentropy()
    loss = loss_func.calculate(y_pred=activation_Softmax.y, y_true=y)
    print(f"Loss: {loss:.4f}")

    acc = Accuracy().calculate(y_pred=activation_Softmax.y, y_true=y)
    print(f"Accuracy: {acc:.4f}")
