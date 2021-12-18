# In[] Libs
import numpy as np
import matplotlib.pyplot as plt
from neural_network_classes.layers import Dense
from neural_network_classes.activations import ReLU, Softmax
from neural_network_classes.error_estimation import CategoricalCrossentropy, Accuracy
from utils.generate_dataset import generate_spiral_data, generate_vertical_data

# In[] Run the App
if __name__ == "__main__":
    n_classes = 10
    # X, y = generate_spiral_data(n_points_per_class=200, n_classes=n_classes, visualization=False)
    X, y = generate_vertical_data(n_points_per_class=200, n_classes=n_classes, visualization=True)
    n_inputs_1 = X.ndim
    n_neurons_1 = 10
    n_inputs_2 = n_neurons_1
    n_neurons_2 = n_classes*2
    n_inputs_3 = n_neurons_2
    n_neurons_3 = n_classes

    dense_1 = Dense(n_inputs=n_inputs_1, n_neurons=n_neurons_1)
    activation_1 = ReLU()
    dense_2 = Dense(n_inputs=n_inputs_2, n_neurons=n_neurons_2)
    activation_2 = Softmax()
    dense_3 = Dense(n_inputs=n_inputs_3, n_neurons=n_neurons_3)
    activation_3 = Softmax()

    loss_func = CategoricalCrossentropy()
    acc_func = Accuracy()

    n_epochs = 100000
    loss_main = 9999
    best_dense_1_weights = dense_1.weights.copy()
    best_dense_1_biases = dense_1.biases.copy()
    best_dense_2_weights = dense_2.weights.copy()
    best_dense_2_biases = dense_2.biases.copy()
    best_dense_3_weights = dense_3.weights.copy()
    best_dense_3_biases = dense_3.biases.copy()


    for idx in range(n_epochs):
        tmp = np.random.randn(n_inputs_1, n_neurons_1)
        dense_1.weights += tmp / np.max(tmp)
        tmp = np.random.randn(1, n_neurons_1)
        dense_1.biases += tmp / np.max(tmp)
        tmp = np.random.randn(n_inputs_2, n_neurons_2)
        dense_2.weights += tmp / np.max(tmp)
        tmp = np.random.randn(1, n_neurons_2)
        dense_2.biases += tmp / np.max(tmp)
        tmp = np.random.randn(n_inputs_3, n_neurons_3)
        dense_3.weights += tmp / np.max(tmp)
        tmp = np.random.randn(1, n_neurons_3)
        dense_3.biases += tmp / np.max(tmp)

        dense_1.forward(X)
        activation_1.forward(dense_1.y)
        dense_2.forward(activation_1.y)
        activation_2.forward(dense_2.y)
        dense_3.forward(activation_2.y)
        activation_3.forward(dense_3.y)

        loss = loss_func.calculate(activation_3.y, y)
        acc = acc_func.calculate(activation_3.y, y)

        if loss < loss_main:
            print(f"Learning improved at round {idx}. Loss: {loss}, Acc: {acc}")
            best_dense_1_weights = dense_1.weights.copy()
            best_dense_1_biases = dense_1.biases.copy()
            best_dense_2_weights = dense_2.weights.copy()
            best_dense_2_biases = dense_2.biases.copy()
            best_dense_3_weights = dense_3.weights.copy()
            best_dense_3_biases = dense_3.biases.copy()
            loss_main = loss
        else:
            dense_1.weights = best_dense_1_weights.copy()
            dense_1.biases = best_dense_1_biases.copy()
            dense_2.weights = best_dense_2_weights.copy()
            dense_2.biases = best_dense_2_biases.copy()
            dense_3.weights = best_dense_3_weights.copy()
            dense_3.biases = best_dense_3_biases.copy()
